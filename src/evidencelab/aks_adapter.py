"""Pinned AKS selection with bounded online BLIP scoring, no upstream redistribution.

The HF BLIP scorer is shared with FOCUS; this is not a claim of numerical parity
with AKS's legacy LAVIS scoring environment or its published full evaluation.
"""
import math

from .selectors import aks


def aks_candidates(total, fps):
    if total < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid video metadata")
    # Official feature_extract.py uses int(fps), then floor(total / int(fps)).
    # For sub-1 fps or sub-second clips, explicitly keep at least one candidate.
    step = max(1, int(fps))
    return [i * step for i in range(max(1, total // step))]


def select_aks(video, question, config, scorer, upstream, stop=lambda: False):
    candidates = aks_candidates(len(video), float(video.get_avg_fps()))
    scorer.reset()
    scores = []
    for start in range(0, len(candidates), config.blip_batch_size):
        if stop():
            raise InterruptedError("Stop requested during AKS scoring")
        batch = candidates[start:start + config.blip_batch_size]
        values = scorer(video, question.question, batch)
        if len(values) != len(batch):
            raise ValueError("AKS BLIP score/frame count mismatch")
        scores.extend(values)
    positions, fallback = aks(scores, config.frames, upstream, t1=.8, t2=-100., depth=5)
    indices = [candidates[i] for i in positions]
    if not indices or indices != sorted(set(indices)) or len(indices) > config.frames:
        raise ValueError("Invalid AKS selection")
    return indices, {
        "scorer_frame_evaluations": len(scorer.evaluated),
        "scorer_unique_frames": len(set(scorer.evaluated)), "scorer_batches": scorer.calls,
        "aks_candidate_count": len(candidates), "aks_fallback": fallback,
        "aks_candidate_implementation": "upstream-floor-fps-subsecond-guard-v1",
        "aks_params": {"t1": .8, "t2": -100., "depth": 5},
    }
