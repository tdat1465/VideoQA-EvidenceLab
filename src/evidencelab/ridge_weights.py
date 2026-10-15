"""Durable, question-only RIDGE preferences, generated before loading the answerer."""
from __future__ import annotations

import gc
import math
from pathlib import Path
import shutil
import time

from .data import digest
from .ridge_adapter import IMPLEMENTATION, RIDGE_FILES, RIDGE_REVISION, validate_weights
from .store import atomic_json


def question_hash(question):
    # Options, labels, category and duration do not enter the weight-generator prompt.
    return digest({"id": question.id, "question": question.question})


def weight_contract(config, contract, author):
    return {
        "schema": 1, "implementation": IMPLEMENTATION, "ridge_revision": RIDGE_REVISION,
        "weight_source_sha256": RIDGE_FILES["weight.py"],
        "prompt_sha256": digest([author.SYSTEM_PROMPT, author.USER_TEMPLATE]),
        "model_id": config.ridge_weight_model_id, "model_revision": config.ridge_weight_revision,
        "dtype": contract["resolved_dtype"], "query": "question-only-no-labels-v1",
        "max_input_tokens": 2048, "max_new_tokens": 512, "batch_size": 1, "do_sample": False,
        "enable_thinking": False, "seed": config.selector_seed,
        "source_sha256": contract["source_sha256"], "runtime": contract["runtime"],
        "annotation_sha256": contract["annotation_sha256"],
        "selected_ids_hash": contract["selected_ids_hash"],
    }


def validate_record(record, question):
    if (not isinstance(record, dict) or record.get("question_sha256") != question_hash(question)
            or record.get("question_id") != question.id):
        raise ValueError("RIDGE cached question/ID changed")
    validate_weights(record.get("weights"))
    for key in ("input_tokens", "output_tokens"):
        if type(record.get(key)) is not int or record[key] < 1:
            raise ValueError("Invalid RIDGE token accounting")
    seconds = record.get("generation_seconds")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid RIDGE generation time")
    if record.get("record_sha256") != digest({k: v for k, v in record.items() if k != "record_sha256"}):
        raise ValueError("RIDGE weight record checksum mismatch")
    return record


class RidgeWeightCache:
    def __init__(self, path, contract, questions):
        import json
        self.path, self.contract = Path(path), contract
        self.questions = {q.id: q for q in questions}
        self.records = {}
        if self.path.exists():
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if value.get("contract") != contract:
                raise ValueError("RIDGE weight cache contract mismatch; use a new run label")
            self.records = value["records"]
            if not isinstance(self.records, dict) or not self.records.keys() <= self.questions.keys():
                raise ValueError("Unknown question IDs in RIDGE weight cache")
            for qid, record in self.records.items():
                validate_record(record, self.questions[qid])

    def save(self):
        atomic_json(self.path, {"contract": self.contract, "records": self.records})

    def add(self, question, weights, input_tokens, output_tokens, generation_seconds, response_sha256):
        if question.id not in self.questions or question != self.questions[question.id]:
            raise ValueError("Unselected question in RIDGE weight generator")
        record = {"question_id": question.id, "question_sha256": question_hash(question),
                  "weights": validate_weights(weights), "input_tokens": input_tokens,
                  "output_tokens": output_tokens, "generation_seconds": generation_seconds,
                  "response_sha256": response_sha256}
        record["record_sha256"] = digest(record)
        validate_record(record, question)
        if question.id in self.records and self.records[question.id] != record:
            raise ValueError("Conflicting RIDGE weight cache record")
        self.records[question.id] = record
        self.save()
        return record

    def check_completed(self, results):
        for result in results:
            record = self.records.get(result["id"])
            if record is None or result.get("ridge_weight_record_sha256") != record["record_sha256"]:
                raise ValueError("RIDGE completed results do not match the durable weight cache")


class QwenWeightGenerator:
    """Original author prompt and HF generate recipe, pinned and fail-closed.

    Adaptations: no multi-GPU scheduler/CUDA override, immutable model revision,
    per-question checkpointing, and rejection of author parser default fallbacks.
    """
    def __init__(self, config, resolved_dtype, author, ram_root):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.config, self.author, self.torch = config, author, torch
        self.ram_root = Path(ram_root).resolve()
        self.cache = self.ram_root / "ridge-qwen-cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.model, self.tokenizer = None, None
        torch.manual_seed(config.selector_seed)
        torch.cuda.reset_peak_memory_stats()
        try:
            options = {"revision": config.ridge_weight_revision, "cache_dir": str(self.cache),
                       "trust_remote_code": False}
            self.tokenizer = AutoTokenizer.from_pretrained(config.ridge_weight_model_id,
                                                           padding_side="left", **options)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(config.ridge_weight_model_id,
                torch_dtype=getattr(torch, resolved_dtype), **options).to("cuda:0")
            self.model.eval().requires_grad_(False)
        except BaseException:
            self.close()
            raise

    def __call__(self, question):
        import hashlib
        torch = self.torch
        messages = [{"role": "system", "content": self.author.SYSTEM_PROMPT},
                    {"role": "user", "content": self.author.USER_TEMPLATE.format(question=question.question)}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
        inputs = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True,
                                max_length=2048).to("cuda:0")
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=512, do_sample=False, num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        input_length = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True).strip()
        parsed = self.author.extract_json(response)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("reasoning"), str):
            raise ValueError(f"RIDGE Qwen returned invalid JSON for {question.id}; refusing default weights")
        weights = {key: parsed[key] for key in self.author.WEIGHT_KEYS if key in parsed}
        validate_weights(weights)
        return weights, int(input_length), int(outputs.shape[1] - input_length), seconds, \
            hashlib.sha256(response.encode()).hexdigest()

    def close(self):
        self.model, self.tokenizer = None, None
        gc.collect()
        self.torch.cuda.empty_cache()
        # Delete only our dedicated cache after releasing all model references.
        if self.cache.is_relative_to(self.ram_root) and self.cache.name == "ridge-qwen-cache":
            shutil.rmtree(self.cache, ignore_errors=False)

    def measurements(self):
        return {"weight_peak_vram_allocated_gib": self.torch.cuda.max_memory_allocated()/1024**3,
                "weight_peak_vram_reserved_gib": self.torch.cuda.max_memory_reserved()/1024**3}


def prepare_weights(cache, needed_questions, factory, stop=lambda: False, expired=lambda: False):
    pending = [q for q in needed_questions if q.id not in cache.records]
    if not pending:
        return {"new_weights": 0, "generation_seconds": 0., "model_setup_seconds": 0., "phase_seconds": 0.}
    if stop() or expired():
        raise InterruptedError("Stop requested before RIDGE weight generation")
    started = time.perf_counter()
    generator = factory()
    setup_seconds = time.perf_counter() - started
    generated, generation_seconds = 0, 0.
    measurements = {}
    try:
        for question in pending:
            if stop() or expired():
                raise InterruptedError("Stop requested during RIDGE weight generation")
            values = generator(question)
            cache.add(question, *values)
            generated += 1
            generation_seconds += values[3]
            print(f"RIDGE weights committed {len(cache.records)}/{len(cache.questions)}: {question.id}", flush=True)
        measurements = getattr(generator, "measurements", lambda: {})()
    finally:
        generator.close()
    return {"new_weights": generated, "generation_seconds": generation_seconds,
            "model_setup_seconds": setup_seconds, "phase_seconds": time.perf_counter()-started,
            **measurements}
