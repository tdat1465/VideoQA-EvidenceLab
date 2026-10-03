import random
import logging

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType

from .blip2.blip2 import Blip2Base, disabled_train
from transformers import LlamaTokenizer, LlamaConfig
from einops import rearrange, repeat
import torch.nn.functional as F
from .blip2.vit import get_sinusoid_encoding_table

from models.topk import HardtopK, HardtopK_segment

logger = logging.getLogger(__name__)


from transformers import StoppingCriteria, StoppingCriteriaList
class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


class Coffee_Mate(Blip2Base):
    """
    coffee_mate model.
    """
    def __init__(self, config):
        super().__init__()
        # pretrained_path
        vit_blip_model_path = config.get("vit_blip_model_path", None)
        llama_model_path = config.get("llama_model_path")
        videochat2_model_path = config.get("videochat2_model_path", "")  
        freeze_vit = config.get("freeze_vit", True)
        freeze_qformer = config.get("freeze_qformer", True)
        # vit
        low_resource = config.get("low_resource", False) # use 8 bit and put vit in cpu
        # qformer
        num_query_token = config.get("num_query_token")
        qformer_hidden_dropout_prob = config.get("qformer_hidden_dropout_prob", 0.1)
        qformer_attention_probs_dropout_prob = config.get("qformer_attention_probs_dropout_prob", 0.1)
        qformer_drop_path_rate = config.get("qformer_drop_path_rate", 0.1)
        extra_num_query_token = config.get("extra_num_query_token", 32)
        self.qformer_text_input = config.get("qformer_text_input", True)
        # prompt
        max_txt_len = config.get("max_txt_len", 32)
        self.begin_signal = "###"
        self.role = ("Human", "Assistant")
        self.start_token = config.get("start_token", "<Video>")
        self.end_token = config.get("end_token", "</Video>")
        self.img_start_token = config.get("img_start_token", "<Image>")
        self.img_end_token = config.get("img_end_token", "</Image>")
        logger.info(f"Add instruction in qformer: {self.qformer_text_input}")
        # debug
        debug = config.get("debug", False)
        use_flash_attention = config.get("use_flash_attention", False)
        self.use_lora = config.get("use_lora", False)
        lora_r = config.get("lora_r", 8)
        lora_alpha = config.get("lora_alpha", 32)
        lora_dropout = config.get("lora_dropout", 0.05)

        self.tokenizer = self.init_tokenizer(truncation_side="left")
        self.low_resource = low_resource
        self.vision_encoder, self.vision_layernorm, = self.init_vision_encoder_umt(config)

        self.qformer, self.query_tokens = self.init_Qformer(
            num_query_token, config.vision_encoder.encoder_embed_dim,
            qformer_hidden_dropout_prob=qformer_hidden_dropout_prob,
            qformer_attention_probs_dropout_prob=qformer_attention_probs_dropout_prob,
            qformer_drop_path_rate=qformer_drop_path_rate,
        )
        
        if not self.qformer_text_input:
            self.qformer.bert.embeddings.word_embeddings = None
            self.qformer.bert.embeddings.position_embeddings = None
            for layer in self.qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.qformer.resize_token_embeddings(len(self.tokenizer))
        self.qformer.cls = None

        if vit_blip_model_path:
            logger.info(f"Load ViT and QFormer from {vit_blip_model_path}")
            state_dict = torch.load(vit_blip_model_path, map_location="cpu")
            msg = self.load_state_dict(state_dict, strict=False)
            logger.info(msg)
            logger.info('Loading ViT and Q-Former Done')    
        
        self.extra_num_query_token = extra_num_query_token
        if extra_num_query_token > 0:
            logger.info(f"Add extra {extra_num_query_token} tokens in QFormer")
            self.extra_query_tokens = nn.Parameter(
                torch.zeros(1, extra_num_query_token, self.query_tokens.shape[-1])
            )

        self.method = config.get("method", "coma")
        # TODO  matching loss
        self.matching_head = nn.Sequential(
            nn.Linear(self.query_tokens.shape[-1], self.query_tokens.shape[-1]),
            nn.ReLU(),
            nn.Linear(self.query_tokens.shape[-1], 1))

        # TODO  different frames support
        self.frame_16_pos_embed = get_sinusoid_encoding_table(
            (224//16)**2*16,
            self.vision_encoder.encoder.num_features,
            ckpt_num_frame=4,
            cur_frame=16,
        )
        self.frame_32_pos_embed = get_sinusoid_encoding_table(
            (224//16)**2*32,
            self.vision_encoder.encoder.num_features,
            ckpt_num_frame=4,
            cur_frame=32,
        )
        self.frame_8_pos_embed = get_sinusoid_encoding_table(
            (224//16)**2*8,
            self.vision_encoder.encoder.num_features,
            ckpt_num_frame=4,
            cur_frame=8,
        )
        self.frame_4_pos_embed = get_sinusoid_encoding_table(
            (224//16)**2*4,
            self.vision_encoder.encoder.num_features,
            ckpt_num_frame=4,
            cur_frame=4,
        )
        self.frame_2_pos_embed = get_sinusoid_encoding_table(
            (224//16)**2*2,
            self.vision_encoder.encoder.num_features,
            ckpt_num_frame=4,
            cur_frame=2,
        )

        if freeze_vit:
            logger.info("freeze vision encoder")
            for _, param in self.vision_encoder.named_parameters():
                param.requires_grad = False
            self.vision_encoder = self.vision_encoder.eval()
            self.vision_encoder.train = disabled_train
            for _, param in self.vision_layernorm.named_parameters():
                param.requires_grad = False
            self.vision_layernorm = self.vision_layernorm.eval()
            self.vision_layernorm.train = disabled_train

        if freeze_qformer:
            logger.info("freeze Qformer")
            for _, param in self.qformer.named_parameters():
                param.requires_grad = False
            self.qformer = self.qformer.eval()
            self.qformer.train = disabled_train
            self.query_tokens.requires_grad = False

        logger.info('Loading LLAMA')
        # problem: do we need to set truncation_side="left"?
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if use_flash_attention:
            logger.info("Use flash attention")
            from .blip2.modeling_llama_mem import LlamaForCausalLM
        else:
            from .blip2.modeling_llama import LlamaForCausalLM
        if debug:
            logger.info("Debug mode, build small LLAMA")
            llama_config = LlamaConfig.from_pretrained(llama_model_path)
            llama_config.hidden_size = 512
            llama_config.intermediate_size = 2048
            llama_config.num_attention_heads = 8
            llama_config.num_hidden_layers = 12
            llama_config.torch_dtype = torch.float16
            self.llama_model = LlamaForCausalLM(llama_config)
        else:
            if self.low_resource:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model_path,
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                )
            else:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model_path,
                    torch_dtype=torch.float16,
                )

        logger.info("freeze LLAMA")
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        logger.info('Loading LLAMA Done')

        if self.use_lora:
            logger.info("Use lora")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, 
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            # self.llama_model.logger.info_trainable_parameters()

        self.llama_proj = nn.Linear(
            self.qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        self.max_txt_len = max_txt_len

        # load weights of VideoChat2
        if videochat2_model_path:
            logger.info(f"Load VideoChat2 from: {videochat2_model_path}")
            ckpt = torch.load(videochat2_model_path, map_location="cpu")
            if 'model' in ckpt.keys():
                state_dict = ckpt["model"]
            else:
                state_dict = ckpt
            msg = self.load_state_dict(state_dict, strict=False)
            logger.info(msg)

    def vit_to_cpu(self):
        self.vision_layernorm.to("cpu")
        self.vision_layernorm.float()
        self.vision_encoder.to("cpu")
        self.vision_encoder.float()

    def encode_img(self, image, instruction, question=None, neg_q=None, is_frame_select=False):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            T = image.shape[1]
            use_image = True if T == 1 else False
            image = image.permute(0, 2, 1, 3, 4)

            image_embeds = self.vision_encoder(image, use_image)
            B, T, L, C = image_embeds.shape
            image_embeds = image_embeds.reshape(B, -1, C)
            image_embeds = self.vision_layernorm(image_embeds).to(device)  # [B, T*L, C]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            if self.extra_num_query_token > 0:
                query_tokens = torch.cat([self.query_tokens, self.extra_query_tokens], dim=1)
            else:
                query_tokens = self.query_tokens
            query_tokens = query_tokens.expand(image_embeds.shape[0], -1, -1)
            if self.qformer_text_input:
                text_Qformer = self.tokenizer(
                    instruction,
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image_embeds.device)
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)

                if is_frame_select:
                    # image_seq = rearrange(image, 'b c t h w -> (b t) c h w')
                    # image_seq = rearrange(image_seq, '(b t) c h w -> b t c h w', t=T // 2).permute(0, 2, 1, 3, 4)
                    # # image_seq = rearrange(image_seq, '(b t) c h w -> b t c h w', t=T // 4).permute(0, 2, 1, 3, 4)
                    # # image_seq = rearrange(image_seq, '(b t) c h w -> b t c h w', t=T // 8).permute(0, 2, 1, 3, 4)
                    # self.vision_encoder.encoder.pos_embed = self.frame_8_pos_embed
                    # # self.vision_encoder.encoder.pos_embed = self.frame_4_pos_embed
                    # # self.vision_encoder.encoder.pos_embed = self.frame_2_pos_embed
                    # image_embeds_seq = self.vision_encoder(image_seq, use_image)  # B*4  T//4 L D
                    # image_embeds_seq = image_embeds_seq.reshape(B, -1, C)
                    # image_embeds_seq = self.vision_layernorm(image_embeds_seq).to(device)  # [B, T*L, C]

                    seq_Qformer = self.tokenizer(
                        question,
                        padding='longest',
                        truncation=True,
                        max_length=self.max_txt_len,
                        return_tensors="pt",
                    ).to(image_embeds.device)

                    # image_embeds_seq = rearrange(image_embeds_seq, 'b (t l) d -> (b t) l d', t=T)
                    image_embeds_seq = rearrange(image_embeds, 'b (t l) d -> (b t) l d', t=T)
                    image_atts_seq = torch.ones(image_embeds_seq.size()[:-1], dtype=torch.long).to(device)
                    query_tokens_seq = torch.cat([self.query_tokens, self.extra_query_tokens], dim=1)
                    query_tokens_seq = query_tokens_seq.expand(image_embeds_seq.shape[0], -1, -1)
                    text_Qformer_id = torch.repeat_interleave(seq_Qformer.input_ids, T, 0)
                    text_Qformer_mask = torch.repeat_interleave(seq_Qformer.attention_mask, T, 0)
                    query_atts_seq = torch.ones(query_tokens_seq.size()[:-1], dtype=torch.long).to(image_embeds.device)
                    Qformer_atts = torch.cat([query_atts_seq, text_Qformer_mask], dim=1)
                    query_output_seq = self.qformer.bert(
                        text_Qformer_id,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens_seq,
                        encoder_hidden_states=image_embeds_seq,
                        encoder_attention_mask=image_atts_seq,
                        return_dict=True,
                    )
                    seq_score = self.matching_head(
                        query_output_seq.last_hidden_state[:, :query_tokens.size(1), :].detach()).mean(dim=1).squeeze(1)
                    seq_score = rearrange(seq_score, '(b t) -> b t', t=T)

                # else:
                Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

                query_output = self.qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            else:
                query_output = self.qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_llama = self.llama_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])

            if neg_q is not None:
                image_embeds = repeat(image_embeds, 'b p h -> (n b) p h', n=10)
                image_atts = repeat(image_atts, 'b p -> (n b) p', n=10)
                query_tokens = repeat(query_tokens, 'b p h -> (n b) p h', n=10)
                candidate_questions = [txt for batch_txt in neg_q for txt in batch_txt]
                candidate_questions = self.tokenizer(
                    candidate_questions,
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image_embeds.device)
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)
                candidate_questions_atts = torch.cat([query_atts, candidate_questions.attention_mask], dim=1)

                matching_output = self.qformer.bert(
                    candidate_questions.input_ids,
                    attention_mask=candidate_questions_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
                prediction = self.matching_head(matching_output.last_hidden_state[:, :query_tokens.size(1), :]).mean(dim=1).squeeze(1)
                prediction = rearrange(prediction, '(n b) -> b n', n=10)  # n=10

        return inputs_llama, use_image, prediction if neg_q is not None else None, query_output.last_hidden_state[:, :query_tokens.size(1), :], seq_score if is_frame_select else None

    def _get_text_len(self, text):
        return self.llama_tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]

    def forward(self, image, text_input, instruction, question, neg_q, qsn_id, where):

        if self.method == 'coma':
            self.vision_encoder.encoder.pos_embed = self.frame_16_pos_embed
            # self.vision_encoder.encoder.pos_embed = self.frame_32_pos_embed
            _, _, teacher_prediction, teacher_output, seq_score = self.encode_img(
                image, instruction, question, neg_q, is_frame_select=True)
            teacher_output = teacher_output.detach()
            teacher_prediction = teacher_prediction.detach()

            with self.maybe_autocast():
                idx_frame = HardtopK_segment(seq_score, m=8, k=1).permute(0, 2, 1).detach()
                student_image = torch.einsum('bfchw,bkf->bkchw', image.float(), idx_frame).half()
                student_image = student_image.detach()

            self.vision_encoder.encoder.pos_embed = self.frame_8_pos_embed
            # self.vision_encoder.encoder.pos_embed = self.frame_4_pos_embed
            img_embeds, use_image, prediction, student_output, _ = self.encode_img(
                student_image, instruction, question, neg_q, is_frame_select=False)

        elif self.method == 'm2l-sd':
            self.vision_encoder.encoder.pos_embed = self.frame_16_pos_embed
            _, _, teacher_prediction, teacher_output, _ = self.encode_img(
                image, instruction, question, neg_q, is_frame_select=False)
            teacher_output = teacher_output.detach()
            teacher_prediction = teacher_prediction.detach()

            image = image[:, ::2, :, :, :]

            self.vision_encoder.encoder.pos_embed = self.frame_8_pos_embed
            img_embeds, use_image, prediction, student_output, _ = self.encode_img(
                image, instruction, question, neg_q, is_frame_select=False)

        batch_size, img_len, _ = img_embeds.shape
        alpha = min(0.4, 0.4 * where)  # up to 0.4 from 0 in epoch 0
        feature_sd_loss = alpha * F.mse_loss(student_output.mean(1), teacher_output.mean(1))
        logit_sd_loss = (1-alpha) * F.cross_entropy(prediction, qsn_id) + alpha * -torch.sum(F.log_softmax(
            prediction, dim=-1) * F.softmax(teacher_prediction,dim=-1), dim=-1).mean()

        # mark the largest length
        # when padding, the attention mask will be 0
        max_len = 0
        input_embed_list = []
        p_before_len_list = []
        target_list = []
        # handle each prompt individually
        for idx, prompt in enumerate(text_input):
            tmp_img_embeds = img_embeds[idx].unsqueeze(0)
            # split the prompt via END_TOKEN
            end_token = self.img_end_token if use_image else self.end_token
            p_before, p_after = prompt.split(end_token)
            p_after = end_token + p_after
            p_before_tokens = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(tmp_img_embeds.device)
            p_after_tokens = self.llama_tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(tmp_img_embeds.device)
            if self.use_lora:
                p_before_embeds = self.llama_model.base_model.model.model.embed_tokens(p_before_tokens.input_ids)
                p_after_embeds = self.llama_model.base_model.model.model.embed_tokens(p_after_tokens.input_ids)
            else:
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids)
            input_embeds = torch.cat([p_before_embeds, tmp_img_embeds, p_after_embeds], dim=1)

            # extract the answers and mask the target
            # the answers are only in the p_after
            sep1 = self.begin_signal + self.role[0] + ": "
            sep2 = self.begin_signal + self.role[1] + ": "
            raw_text = p_after.split(sep2)
            for idx in range(1, len(raw_text)):
                raw_text[idx] = sep2 + raw_text[idx]
            # the first raw_text contains system and question
            # the last raw_text only contains answer
            # rstrip() for the extra " "
            answer_targets = p_after_tokens.input_ids.clone()
            # target: "###Human:       ###Assistant: xxxxx. ###"
            system = raw_text[0].split(sep1)[0]
            system_len = self._get_text_len(system.rstrip())
            sep_len = self._get_text_len(sep1.rstrip())
            cur_len = self._get_text_len(raw_text[0].rstrip())
            answer_targets[:, :system_len] = -100
            answer_targets[:, (system_len+sep_len):cur_len] = -100
            for text in raw_text[1:-1]: 
                total_len = self._get_text_len(text.rstrip())
                ans_len = self._get_text_len((text.split(sep1)[0]+sep1).rstrip())
                answer_targets[:, (cur_len+ans_len):(cur_len+total_len)] = -100
                cur_len += total_len
            cur_len += self._get_text_len(raw_text[-1].rstrip())
            assert cur_len == answer_targets.shape[1], f"The final length ({cur_len}) is not equal to the original prompt ({answer_targets.shape[1]}): {prompt}"

            max_len = max(max_len, input_embeds.shape[1])  # select the max sample length
            input_embed_list.append(input_embeds)
            p_before_len_list.append(p_before_tokens.input_ids.shape[1])
            target_list.append(answer_targets)
        
        # plus one for bos
        # max_txt_len plus num_query_token is the max len
        txt_len = min(max_len + 1, self.max_txt_len + img_len)
        inputs_embeds = torch.ones([batch_size, txt_len], dtype=torch.long).to(img_embeds.device) * self.llama_tokenizer.pad_token_id
        if self.use_lora:
            inputs_embeds = self.llama_model.base_model.model.model.embed_tokens(inputs_embeds)
        else:
            inputs_embeds = self.llama_model.model.embed_tokens(inputs_embeds)
        attention_mask = torch.zeros([batch_size, txt_len], dtype=torch.long).to(img_embeds.device)
        targets = torch.ones([batch_size, txt_len], dtype=torch.long).to(img_embeds.device).fill_(-100)
        # set bos_token
        inputs_embeds[:, :1] = self.llama_tokenizer.bos_token_id
        for idx in range(batch_size):
            input_len = min(input_embed_list[idx].shape[1], txt_len - 1)
            # if less than txt_len, the input will be padding
            # if more than txt_len, the input will be truncated
            inputs_embeds[idx, 1:(input_len+1)] = input_embed_list[idx][:, :input_len]
            # the attention_mask is 0 when padding
            attention_mask[idx, :(input_len+1)] = 1
            # the target is -100 when padding
            p_before_len = p_before_len_list[idx]
            targets[idx, (p_before_len+img_len+1):(input_len+1)] = target_list[idx][0, :(input_len-p_before_len-img_len)]

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        return dict(
            loss=outputs.loss + feature_sd_loss + logit_sd_loss,
        )

    @torch.no_grad()
    def generate(self, image, text_input, instruction, question, generate, answer):
        if self.method == 'coma':
            self.vision_encoder.encoder.pos_embed = self.frame_16_pos_embed
            # self.vision_encoder.encoder.pos_embed = self.frame_32_pos_embed
            _, _, _, _, seq_score = self.encode_img(image, instruction, question, is_frame_select=True)
            # seq_score = seq_score.detach()
            with self.maybe_autocast():
                idx_frame = HardtopK_segment(seq_score, m=8, k=1).permute(0, 2, 1).detach()
                image = torch.einsum('bfchw,bkf->bkchw', image.float(), idx_frame).half()

        self.vision_encoder.encoder.pos_embed = self.frame_8_pos_embed
        # self.vision_encoder.encoder.pos_embed = self.frame_4_pos_embed
        img_embeds, use_image, _, _, _ = self.encode_img(image, instruction, question, is_frame_select=False)
        batch_size, img_len, _ = img_embeds.shape

        output_list = []
        # handle each prompt individually
        stop_words_ids = [
            torch.tensor([835]).to(img_embeds.device),
            torch.tensor([2277, 29937]).to(img_embeds.device)]  # '###' can be encoded in two different ways.

        for idx, prompt in enumerate(generate):
            tmp_img_embeds = img_embeds[idx].unsqueeze(0)
            # split the prompt via END_TOKEN
            end_token = self.img_end_token if use_image else self.end_token
            p_before, p_after = prompt.split(end_token)
            p_after = end_token + p_after
            # p_before_tokens = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(tmp_img_embeds.device)
            # add eos
            p_before_tokens = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=True).to(tmp_img_embeds.device)
            p_after_tokens = self.llama_tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(tmp_img_embeds.device)

            if self.use_lora:
                p_before_embeds = self.llama_model.base_model.model.model.embed_tokens(p_before_tokens.input_ids)
                p_after_embeds = self.llama_model.base_model.model.model.embed_tokens(p_after_tokens.input_ids)
            else:
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids)
            input_embeds = torch.cat([p_before_embeds, tmp_img_embeds, p_after_embeds], dim=1)

            with self.maybe_autocast():
                outputs = self.llama_model.generate(
                    inputs_embeds=input_embeds,
                    max_new_tokens=100,
                    stopping_criteria=StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)]),
                    num_beams=1,
                    do_sample=False,
                    min_length=1,
                    top_p=0.9,
                    repetition_penalty=1.0,
                    length_penalty=1,
                    temperature=1.0,
                )
            output_token = outputs[0]
            if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
                output_token = output_token[1:]
            if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                output_token = output_token[1:]
            output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
            output_text = output_text.split('###')[0]  # remove the stop sign '###'
            output_text = output_text.split('Assistant:')[-1].strip()
            output_text = output_text.strip().split('\n')[0]
            # print('output_text:', output_text)

            output_list.append(output_text)
        return output_list