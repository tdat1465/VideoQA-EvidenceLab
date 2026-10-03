"""Run the unchanged, SHA-verified upstream FOCUS selector over a full video.

Scoring is an explicitly identified HF port of BLIP ITM, not a claim of
numerical parity with the paper's legacy LAVIS environment.
"""
import hashlib
import importlib.util
import math
import re
from pathlib import Path

from .data import file_hash
from .longvideo_config import FOCUS_SHA256


def load_focus(path: Path):
    if file_hash(path) != FOCUS_SHA256:
        raise ValueError("FOCUS source hash mismatch")
    spec = importlib.util.spec_from_file_location("evidencelab_upstream_focus", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FOCUS


def question_seed(seed, question_id):
    # Unlike rank-based upstream RNG this remains identical across resume/node/order.
    return int.from_bytes(hashlib.sha256(f"{seed}:{question_id}".encode()).digest()[:8], "big")


def blip_caption(text):
    text = re.sub(r'([.!"()*#:;~])', " ", text.lower())
    text = re.sub(r"\s{2,}", " ", text).rstrip("\n").strip(" ")
    return " ".join(text.split(" ")[:50])


def prepare_blip_itm_tokenizer(tokenizer, text_config, embedding_rows):
    """Restore LAVIS special tokens to the existing checkpoint embedding rows.

    The pinned HF large-COCO tokenizer contains only the 30,522 BERT tokens,
    while the trained model has two extra rows: [DEC], then [ENC]. Register
    them in that order, as LAVIS BlipBase.init_tokenizer does. Never resize or
    initialize model embeddings, or silently use CLS/UNK for ITM scoring.
    """
    if text_config.bos_token_id != 30522 or text_config.vocab_size != 30524 or embedding_rows != 30524:
        raise ValueError("Unexpected BLIP checkpoint vocabulary; expected 30524 trained rows and BOS 30522")
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    dec = tokenizer.convert_tokens_to_ids("[DEC]")
    enc = tokenizer.convert_tokens_to_ids("[ENC]")
    if dec != 30522 or enc != 30523 or len(tokenizer) != embedding_rows:
        raise ValueError("BLIP special-token IDs do not match the trained checkpoint embedding rows")
    return enc


class BlipITMScorer:
    def __init__(self, config):
        import torch
        from transformers import BlipForImageTextRetrieval, BlipProcessor
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode
        self.torch = torch
        self.batch_size = config.blip_batch_size
        self.processor = BlipProcessor.from_pretrained(config.blip_id, revision=config.blip_revision)
        self.model = BlipForImageTextRetrieval.from_pretrained(
            config.blip_id, revision=config.blip_revision, torch_dtype=torch.float32).to("cuda:0")
        self.model.eval().requires_grad_(False)
        # LAVIS evaluation uses torchvision/PIL bicubic resize (not HF image resize).
        self.transform = transforms.Compose([
            transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((.48145466, .4578275, .40821073), (.26862954, .26130258, .27577711)),
        ])
        self.enc_id = prepare_blip_itm_tokenizer(
            self.processor.tokenizer, self.model.config.text_config,
            self.model.text_encoder.get_input_embeddings().num_embeddings)
        self.reset()

    def reset(self):
        self.evaluated = []
        self.calls = 0

    def __call__(self, video, query, indices):
        from PIL import Image
        torch = self.torch
        values = []
        for start in range(0, len(indices), self.batch_size):
            batch = [int(i) for i in indices[start:start + self.batch_size]]
            # Decode only one small batch. Never materialize all source frames.
            pixels = torch.stack([self.transform(Image.fromarray(video[i].asnumpy()).convert("RGB"))
                                  for i in batch]).to("cuda:0")
            inputs = self.processor.tokenizer([blip_caption(query)] * len(batch), padding="longest",
                                             truncation=True, max_length=35, return_tensors="pt").to("cuda:0")
            # HF retrieval forward does not replace CLS with ENC automatically.
            inputs["input_ids"][:, 0] = self.enc_id
            inputs.pop("token_type_ids", None)
            with torch.inference_mode():
                scores = self.model(pixel_values=pixels, **inputs, use_itm_head=True).itm_score
                values.extend(torch.softmax(scores.float(), dim=-1)[:, 1].cpu().tolist())
            self.evaluated.extend(batch)
            self.calls += 1
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
            raise ValueError("Nonfinite/out-of-range BLIP scores")
        return values


def select_focus(video, question, config, scorer, focus_class):
    import numpy as np
    scorer.reset()
    fps = float(video.get_avg_fps())
    spacing = (len(video) / max(1.0, fps)) / config.frames
    gap = 0.0 if spacing <= .2 else min(.25 * spacing, 1.0)
    indices, details = focus_class(similarity_fn=scorer).select_keyframes(
        video, question.question, config.frames, min_gap_sec=gap,
        rng=np.random.default_rng(question_seed(config.selector_seed, question.id)))
    indices = [int(i) for i in indices]
    if not indices or indices != sorted(set(indices)) or len(indices) > config.frames:
        raise ValueError("Invalid upstream FOCUS selection")
    if min(indices) < 0 or max(indices) >= len(video):
        raise ValueError("FOCUS selected an out-of-range frame")
    # Do not silently pad if upstream returns fewer than k frames.
    return indices, {"scorer_frame_evaluations": len(scorer.evaluated),
                     "scorer_unique_frames": len(set(scorer.evaluated)), "scorer_batches": scorer.calls,
                     "focus_details": details, "selector_rng_seed": question_seed(config.selector_seed, question.id)}
