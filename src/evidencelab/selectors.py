"""Selection uses public questions and model scores, never labels or scene graphs."""
from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path

AKS_COMMIT = "b0b8a58fedf1d05d78151e2969cecde60e83721d"
AKS_SHA256 = "594557130aa702fb1c3eafae9df120514d36e9684443b415e0483dce14b4f149"
AKS_URL = f"https://raw.githubusercontent.com/ncTimTang/AKS/{AKS_COMMIT}/frame_select.py"


def uniform(n: int, k: int) -> list[int]:
    if n <= 0 or k <= 0:
        return []
    k = min(n, k)
    if k == 1:
        return [n // 2]
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def topk(scores: list[float], k: int) -> list[int]:
    if not scores or any(not math.isfinite(x) for x in scores):
        raise ValueError("Invalid relevance scores")
    return sorted(sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k])


def aks(scores: list[float], k: int, upstream: Path, t1=0.8, t2=-100.0, depth=5):
    """Call the pinned upstream splitting function, reproducing its quota logic.

    Do not redistribute upstream code: the inspected repository has no LICENSE.
    Short/constant sequences bypass upstream's divide-by-zero/empty-segment case.
    These explicitly labelled fallbacks are NOT claimed as paper reproduction.
    """
    if hashlib.sha256(upstream.read_bytes()).hexdigest() != AKS_SHA256:
        raise ValueError("AKS source hash mismatch")
    if not scores or any(not math.isfinite(x) for x in scores):
        raise ValueError("Invalid AKS scores")
    if len(scores) <= k or max(scores) == min(scores):
        return uniform(len(scores), k), "short_or_constant_uniform"
    import numpy as np
    spec = importlib.util.spec_from_file_location("aks_upstream", upstream)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    array = np.asarray(scores, dtype=float)
    normalized = (array - array.min()) / (array.max() - array.min())
    # Official code otherwise recursively creates empty segments on very short clips.
    effective_depth = min(depth, int(math.log2(len(scores))))
    segments, indices = module.meanstd(len(scores), [{"score": normalized, "depth": 0}],
                                      k, [list(range(len(scores)))], t1, t2, effective_depth)
    selected = []
    for segment, ids in zip(segments, indices):
        quota = int(k / 2 ** segment["depth"])
        selected.extend(ids[i] for i in topk(list(segment["score"]), quota))
    flag = "depth_capped_short_clip" if effective_depth != depth else "upstream"
    if not selected:
        return uniform(len(scores), k), "zero_quota_uniform"
    return sorted(set(selected)), flag


def expand_uniform(n: int, initial: list[int], budget: int) -> list[int]:
    """Keep initial frames and fill the largest uncovered intervals."""
    selected = set(initial)
    while len(selected) < min(n, budget):
        remaining = [i for i in range(n) if i not in selected]
        picked = max(remaining, key=lambda i: (min(abs(i-j) for j in selected), -i))
        selected.add(picked)
    return sorted(selected)


def evidence_frames(times: list[float], initial: list[int], budget: int,
                    question_scores: list[float], option_scores: list[list[float]],
                    competing: tuple[int, int], neighbor_seconds: float, weight: float):
    """Prototype: discriminative anchors plus before/after context; no VLM planner.

    abs(option_a-option_b) encourages discriminating evidence, while two queues
    balance evidence favoring each contender. Not an implementation of A.I.R.
    """
    a, b = competing
    n = len(times)
    if len(question_scores) != n or len(option_scores[a]) != n or len(option_scores[b]) != n:
        raise ValueError("Score/frame shape mismatch")
    score = [abs(option_scores[a][i] - option_scores[b][i]) + weight * question_scores[i]
             for i in range(n)]
    if any(not math.isfinite(x) for x in score):
        raise ValueError("Non-finite evidence scores")
    selected = set(initial)
    queues = [sorted([i for i in range(n) if option_scores[a][i] >= option_scores[b][i]],
                     key=lambda i: (-score[i], i)),
              sorted([i for i in range(n) if option_scores[a][i] < option_scores[b][i]],
                     key=lambda i: (-score[i], i))]
    turn, anchors = 0, []
    while len(selected) < min(budget, n):
        pool = [i for i in queues[turn % 2] if i not in selected]
        if not pool:
            pool = sorted([i for i in range(n) if i not in selected], key=lambda i: (-score[i], i))
        anchor = pool[0]
        anchors.append(anchor)
        before = [i for i in range(anchor) if times[anchor] - times[i] <= neighbor_seconds]
        after = [i for i in range(anchor+1, n) if times[i] - times[anchor] <= neighbor_seconds]
        candidates = [anchor]
        if before:
            candidates.append(before[0])
        if after:
            candidates.append(after[-1])
        for i in candidates:
            if len(selected) < min(budget, n):
                selected.add(i)
        turn += 1
    return sorted(selected), anchors
