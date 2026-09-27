"""Optional real processor check (downloads tokenizer/processor, NEVER model weights).

Run after installing requirements-server.txt and CPU or CUDA torch/torchvision.
This does not validate full generation, GPU memory, or model accuracy.
"""
import argparse
import json

import numpy as np
from PIL import Image
from transformers import AutoProcessor

from evidencelab.backends import HFBackend
from evidencelab.config import Config
from evidencelab.data import Question

p = argparse.ArgumentParser()
p.add_argument("--backend", choices=["molmo2", "qwen"], default="molmo2")
a = p.parse_args()
cfg = Config() if a.backend == "molmo2" else Config(
    backend="qwen", model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    revision="cc594898137f460bfe9f0759e9844b3ce807cfb5")
backend = object.__new__(HFBackend)
backend.config = cfg
backend.processor = AutoProcessor.from_pretrained(cfg.model_id, revision=cfg.revision,
                                                 use_fast=False,
                                                 trust_remote_code=cfg.backend == "molmo2")
frames = [Image.fromarray(np.full((160, 224, 3), i*30, dtype=np.uint8)) for i in range(3)]
times = [11.12, 11.58, 15.4]
inputs = backend._inputs(Question("smoke", "What happens?", ("Sit", "Stand", "Walk", "Run", "Stop")),
                         frames, times)
assert inputs["input_ids"].shape[0] == 1
tokenizer = backend.processor.tokenizer
assert all(len(tokenizer.encode(x, add_special_tokens=False)) == 1 for x in "ABCDE")
decoded = tokenizer.decode(inputs["input_ids"][0])
if cfg.backend == "molmo2":
    # Molmo formats timestamps to tenths in its input string.
    assert all(f"{t:.1f}" in decoded for t in times), decoded
else:
    assert all(f"{t:.3f}s" in decoded for t in times)
print(json.dumps({"backend": cfg.backend, "revision": cfg.revision,
                  "input_shapes": {k: list(v.shape) for k, v in inputs.items() if hasattr(v, "shape")},
                  "timestamps_preserved": True, "full_model_loaded": False}, indent=2))
