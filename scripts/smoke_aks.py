"""Verify wrapper parity on ordinary inputs plus the documented fallback cases."""
import argparse
import heapq
import importlib.util
from pathlib import Path

import numpy as np

from evidencelab.data import file_hash
from evidencelab.selectors import AKS_SHA256, aks

p = argparse.ArgumentParser()
p.add_argument("source", type=Path)
a = p.parse_args()
assert file_hash(a.source) == AKS_SHA256
spec = importlib.util.spec_from_file_location("aks_original", a.source)
upstream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)
for seed in range(20):
    scores = np.random.default_rng(seed).random(64)
    normalized = (scores - scores.min()) / (scores.max() - scores.min())
    segments, ids = upstream.meanstd(64, [{"score": normalized, "depth": 0}], 16,
                                    [list(range(64))], .8, -100, 4)
    expected = sorted(indices[j] for segment, indices in zip(segments, ids)
                      for j in heapq.nlargest(int(16/2**segment["depth"]), range(len(segment["score"])),
                                             segment["score"].__getitem__))
    actual, status = aks(list(scores), 16, a.source, depth=4)
    assert actual == expected and status == "upstream"
assert aks([1.] * 64, 16, a.source)[1] == "short_or_constant_uniform"
assert aks([.1, .2, .3], 16, a.source)[0] == [0, 1, 2]
print("AKS parity: 20 ordinary arrays passed; constant and short-input guards passed.")
