"""Frozen official LLaVA-Video-7B-Qwen2; run in its own pinned environment."""
import copy
from contextlib import contextmanager

from .backends import HFBackend, prompt
from .hardware import execution_profile
from .pipeline import Decision


@contextmanager
def pinned_siglip(config):
    # Official tower does not forward revision. Scope this hook to model loading
    # and preserve the official tower implementation and pretrained parameters.
    from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionModel
    original_descriptor = SigLipVisionModel.__dict__.get("from_pretrained")
    original = SigLipVisionModel.from_pretrained

    def load(cls, name, *args, **kwargs):
        if name != config.vision_id:
            raise ValueError("Unexpected unpinned vision tower")
        kwargs["revision"] = config.vision_revision
        return original(name, *args, **kwargs)

    SigLipVisionModel.from_pretrained = classmethod(load)
    try:
        yield
    finally:
        if original_descriptor is None:
            delattr(SigLipVisionModel, "from_pretrained")
        else:
            SigLipVisionModel.from_pretrained = original_descriptor


class LlavaVideoBackend:
    reset_peak = HFBackend.reset_peak
    measurements = HFBackend.measurements

    def __init__(self, config):
        import torch
        from transformers import AutoTokenizer
        from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM
        self.torch, self.config = torch, config
        self.profile = execution_profile(config)
        self.dtype = getattr(torch, self.profile["resolved_dtype"])
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.revision)
        model_config = LlavaQwenConfig.from_pretrained(config.model_id, revision=config.revision)
        if model_config.mm_vision_tower != config.vision_id:
            raise ValueError("Model config has a different vision tower")
        model_config.mm_spatial_pool_stride = 2
        model_config.mm_spatial_pool_mode = "bilinear"
        # Upstream silently truncates expanded image embeddings if this is set.
        # Disable truncation and explicitly reject over-budget inputs below.
        model_config.tokenizer_model_max_length = None
        with pinned_siglip(config):
            self.model = LlavaQwenForCausalLM.from_pretrained(
                config.model_id, revision=config.revision, config=model_config,
                torch_dtype=self.dtype, low_cpu_mem_usage=True, device_map={"": 0},
                attn_implementation="sdpa")
            tower = self.model.get_vision_tower()
            if not tower.is_loaded:
                tower.load_model(device_map={"": 0})
        tower.to(device="cuda:0", dtype=self.dtype)
        self.processor = tower.image_processor
        self.model.eval().requires_grad_(False)
        self.letter_ids = []
        for label in "ABCDE":
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError("Option label must encode to one token")
            self.letter_ids.append(ids[0])
        self.video_duration = None

    def generate_text(self, text, max_new_tokens=16):
        from llava.conversation import conv_templates
        torch = self.torch
        conversation = copy.deepcopy(conv_templates["qwen_1_5"])
        conversation.append_message(conversation.roles[0], text)
        conversation.append_message(conversation.roles[1], None)
        inputs = self.tokenizer(conversation.get_prompt(), return_tensors="pt").to("cuda:0")
        count = int(inputs["input_ids"].shape[-1])
        if count + max_new_tokens > self.config.max_input_tokens:
            raise ValueError("LENS allocation prompt exceeds input budget")
        with torch.inference_mode(), torch.autocast("cuda", dtype=self.dtype):
            # Official LLaVA generates from inputs_embeds, so returned IDs are
            # new tokens only (unlike HFBackend's input_ids generation).
            output = self.model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                         images=None, modalities=["text"],
                                         max_new_tokens=max_new_tokens, do_sample=False)
        return {"text": self.tokenizer.decode(output[0], skip_special_tokens=True),
                "input_tokens": count, "output_tokens": int(output.shape[-1])}

    def answer(self, question, frames, times):
        import numpy as np
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        torch = self.torch
        duration = self.video_duration
        if duration is None:
            raise ValueError("Set actual video duration before answering")
        timestamps = ", ".join(f"{t:.2f}s" for t in times)
        instruction = (f"The video lasts for {duration:.2f} seconds, and {len(frames)} frames are sampled "
                       f"from it. These frames are located at {timestamps}. "
                       "Please answer the following questions related to this video.")
        conversation = copy.deepcopy(conv_templates["qwen_1_5"])
        conversation.append_message(conversation.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + instruction
                                    + "\n" + prompt(question))
        conversation.append_message(conversation.roles[1], None)
        input_ids = tokenizer_image_token(conversation.get_prompt(), self.tokenizer,
                                         IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda:0")
        video = np.stack([np.asarray(frame) for frame in frames])
        pixels = self.processor.preprocess(video, return_tensors="pt")["pixel_values"].to("cuda:0", self.dtype)
        with torch.inference_mode(), torch.autocast("cuda", dtype=self.dtype):
            _, positions, mask, _, embeddings, _ = self.model.prepare_inputs_labels_for_multimodal(
                input_ids, None, torch.ones_like(input_ids), None, None, [pixels], ["video"])
            tokens = int(embeddings.shape[1])
            if tokens + 1 > min(self.config.max_input_tokens, self.model.config.max_position_embeddings):
                raise ValueError(f"Expanded LLaVA input has {tokens} tokens; no silent truncation")
            # Only the first answer token is needed. Compute the LM head at the
            # final position, avoiding a [video_tokens, 152064] logit allocation.
            hidden = self.model.model(inputs_embeds=embeddings, attention_mask=mask,
                                      position_ids=positions, use_cache=False, return_dict=True).last_hidden_state
            next_logits = self.model.lm_head(hidden[:, -1, :]).float()
        logits = next_logits[0, self.letter_ids[:len(question.choices)]]
        return Decision(torch.softmax(logits, dim=-1).cpu().tolist(), tokens)
