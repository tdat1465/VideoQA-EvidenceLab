"""Frozen Hugging Face backends. Real GPU evaluation is not a CPU smoke test."""
from __future__ import annotations

import hashlib
import math
from collections import OrderedDict

from .pipeline import Decision


def prompt(question):
    choices = "\n".join(f"{chr(65+i)}. {x}" for i, x in enumerate(question.choices))
    return ("Answer the multiple-choice question using the video evidence. "
            "Timestamps are seconds in the source video. Select exactly one option letter.\n"
            f"Question: {question.question}\n{choices}\nAnswer:")


def molmo_metadata(times):
    # HF VideoMetadata.timestamps = frames_indices / fps. Fractional indices with
    # a virtual 1 Hz timeline preserve irregular real timestamps without resampling.
    return {"total_num_frames": len(times), "fps": 1.0, "frames_indices": list(times),
            "duration": max(times) + 0.001}


class HFBackend:
    def __init__(self, config):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Require exactly one allocated CUDA GPU; do not set CUDA_VISIBLE_DEVICES yourself")
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("This configuration requires native BF16")
        self.torch, self.config = torch, config
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        self.processor = AutoProcessor.from_pretrained(config.model_id, revision=config.revision,
                                                       use_fast=False,
                                                       trust_remote_code=config.backend == "molmo2")
        self.model = AutoModelForImageTextToText.from_pretrained(
            config.model_id, revision=config.revision, trust_remote_code=config.backend == "molmo2",
            torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0})
        self.model.eval()
        self.model.requires_grad_(False)
        tokenizer = self.processor.tokenizer
        self.letter_ids = []
        for label in "ABCDE":
            ids = tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Letter {label} is not a single token; change scoring protocol explicitly")
            self.letter_ids.append(ids[0])

    def _inputs(self, question, frames, times):
        import numpy as np
        text = prompt(question)
        if self.config.backend == "molmo2":
            messages = [{"role": "user", "content": [{"type": "video", "video": "predecoded"},
                                                        {"type": "text", "text": text}]}]
            rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            video = np.stack([np.asarray(frame) for frame in frames])
            return self.processor(text=[rendered], videos=[video],
                                  video_metadata=[molmo_metadata(times)], do_sample_frames=False,
                                  return_tensors="pt")
        # Ordered multi-image input exposes exact timestamps for nonuniform sampling.
        # This is deliberately labelled qwen_multi_image, not paper-native video eval.
        content = []
        for frame, t in zip(frames, times):
            content.extend([{"type": "text", "text": f"Frame at {t:.3f}s:"},
                            {"type": "image", "image": frame}])
        content.append({"type": "text", "text": text})
        rendered = self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                       tokenize=False, add_generation_prompt=True)
        return self.processor(text=[rendered], images=frames, return_tensors="pt",
                              max_pixels=self.config.max_image_pixels, min_pixels=56*56)

    def answer(self, question, frames, times):
        torch = self.torch
        inputs = self._inputs(question, frames, times)
        tokens = int(inputs["input_ids"].shape[-1])
        if tokens > self.config.max_input_tokens:
            raise RuntimeError(f"Input has {tokens} tokens > configured cap {self.config.max_input_tokens}; "
                               "create a NEW run with a smaller frame budget, never silently truncate")
        inputs = inputs.to("cuda:0")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            result = self.model.generate(**inputs, max_new_tokens=1, do_sample=False,
                                         return_dict_in_generate=True, output_scores=True)
        logits = result.scores[0][0, self.letter_ids[:len(question.choices)]].float()
        probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
        return Decision(probabilities, tokens)

    def reset_peak(self):
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()

    def measurements(self):
        self.torch.cuda.synchronize()
        return {"peak_vram_allocated_gib": self.torch.cuda.max_memory_allocated() / 1024**3,
                "peak_vram_reserved_gib": self.torch.cuda.max_memory_reserved() / 1024**3}


class ClipScorer:
    def __init__(self, config):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch, self.config = torch, config
        self.processor = CLIPProcessor.from_pretrained(config.clip_id, revision=config.clip_revision, use_fast=False)
        self.model = CLIPModel.from_pretrained(config.clip_id, revision=config.clip_revision).to(config.clip_device)
        self.model.eval().requires_grad_(False)
        self.cache = OrderedDict()

    def score(self, frames, texts):
        torch, device = self.torch, self.config.clip_device
        key = hashlib.sha256()
        for image in frames:
            key.update(str(image.size).encode())
            key.update(image.tobytes())
        key = key.hexdigest()
        with torch.inference_mode():
            if key in self.cache:
                image_features = self.cache.pop(key).to(device)
            else:
                chunks = []
                for offset in range(0, len(frames), self.config.clip_batch_size):
                    data = self.processor(images=frames[offset:offset+self.config.clip_batch_size],
                                          return_tensors="pt").to(device)
                    chunks.append(self.model.get_image_features(**data))
                image_features = torch.cat(chunks)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            self.cache[key] = image_features.cpu()
            while len(self.cache) > 32:
                self.cache.popitem(last=False)
            encoded = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(device)
            text_features = self.model.get_text_features(**encoded)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            # Cosine scores, not unnormalized logits. This is the AKS+CLIP variant.
            result = (text_features @ image_features.T).float().cpu().tolist()
        return result


class MockBackend:
    """Deterministic mechanics test. Scores have NO scientific meaning."""
    def answer(self, question, frames, times):
        key = f"{question.id}:{len(frames)}".encode()
        raw = hashlib.sha256(key).digest()
        values = [1.0 + raw[i]/255 for i in range(len(question.choices))]
        total = sum(values)
        return Decision([x/total for x in values], 100 + len(frames)*10)

    def reset_peak(self):
        pass

    def measurements(self):
        return {"peak_vram_allocated_gib": None, "peak_vram_reserved_gib": None}


class MockScorer:
    def score(self, frames, texts):
        return [[math.sin(i * 0.7 + len(text)) for i in range(len(frames))] for text in texts]
