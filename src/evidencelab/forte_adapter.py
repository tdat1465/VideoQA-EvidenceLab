"""Independent FORTE reconstruction from arXiv:2610.00573v1, Appendices A-C.

No author implementation is vendored. Only original frame indices and the public
question reach the scorer. NumPy/SciPy are imported lazily for stdlib-only login
planning. A heap of gap maxima implements the same exact greedy acquisition
decisions as the paper's heap over individual candidates.
"""
from __future__ import annotations

from collections import deque
import heapq
import math
import time

from .selectors import uniform

IMPLEMENTATION = "paper-v1-independent-hf-blip-gap-heap-v1"
PAPER = "https://arxiv.org/abs/2610.00573v1"


def reference_grid(n, fps, rho):
    if (type(n) is not int or n < 1 or not math.isfinite(fps) or fps <= 0
            or not math.isfinite(rho) or rho <= 0):
        raise ValueError("Invalid FORTE video/scoring allowance")
    stride = max(1, math.floor(fps / rho))
    # The appendix specifies count, but not integer mapping. Use EvidenceLab's
    # endpoint-spanning uniform grid (midpoint when count=1), without duplicates.
    return uniform(n, max(1, n // stride))


def normalize(scores):
    import numpy as np
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid measured relevance scores")
    lo, hi = float(values.min()), float(values.max())
    return np.zeros_like(values) if hi == lo else (values - lo) / (hi - lo)


def gp_predict(observed_times, normalized_scores, query_times, length_scale=10.,
               initial_width=128, max_width=1024, stop=lambda: False):
    """FP64 banded RBF fit with the paper's jitter/width recovery schedule."""
    import numpy as np
    from scipy.linalg import cholesky_banded, cho_solve_banded

    t = np.asarray(observed_times, dtype=np.float64)
    y = np.asarray(normalized_scores, dtype=np.float64)
    query = np.asarray(query_times, dtype=np.float64)
    if (t.ndim != 1 or y.shape != t.shape or not len(t) or query.ndim != 1
            or not np.isfinite(t).all() or not np.isfinite(y).all()
            or not np.isfinite(query).all() or np.any(np.diff(t) <= 0)
            or length_scale <= 0 or initial_width < 0 or max_width < initial_width):
        raise ValueError("Invalid GP input")
    mean = float(y.mean())
    width = min(initial_width, len(t) - 1)
    cap = min(max_width, len(t) - 1)
    attempts = []
    fit_start = time.perf_counter()
    while True:
        if stop():
            raise InterruptedError("Stop requested during FORTE GP fit")
        band = np.zeros((width + 1, len(t)), dtype=np.float64)
        band[0] = 1.
        for offset in range(1, width + 1):
            band[offset, :len(t) - offset] = np.exp(
                -.5 * ((t[offset:] - t[:-offset]) / length_scale) ** 2)
        alpha = None
        for exponent in range(-12, -2):
            epsilon = 10. ** exponent
            band[0] = 1. + epsilon
            try:
                factor = cholesky_banded(band, lower=True, check_finite=False)
                alpha = cho_solve_banded((factor, True), y - mean, check_finite=False)
                attempts.append({"half_width": width, "epsilon": epsilon, "success": True})
                break
            except np.linalg.LinAlgError:
                attempts.append({"half_width": width, "epsilon": epsilon, "success": False})
        if alpha is not None:
            break
        if width >= cap:
            raise RuntimeError("FORTE banded GP failed at the recovery cap")
        width = min(max(1, 2 * width), cap)
    fit_seconds = time.perf_counter() - fit_start
    predict_start = time.perf_counter()
    predictions = np.empty(len(query), dtype=np.float64)
    offsets = np.arange(-width, width + 1)
    # Vectorized binary insertion lookup; chunks bound temporary matrices to
    # 1024*(2w+1) entries instead of all source frames times all observations.
    for start in range(0, len(query), 1024):
        if stop():
            raise InterruptedError("Stop requested during FORTE GP prediction")
        q = query[start:start + 1024]
        centers = np.minimum(np.searchsorted(t, q, side="left"), len(t) - 1)
        positions = centers[:, None] + offsets[None, :]
        valid = (positions >= 0) & (positions < len(t))
        safe = np.clip(positions, 0, len(t) - 1)
        weights = np.exp(-.5 * ((q[:, None] - t[safe]) / length_scale) ** 2)
        weights *= valid
        predictions[start:start + len(q)] = mean + (weights * alpha[safe]).sum(axis=1)
    np.clip(predictions, float(y.min()), float(y.max()), out=predictions)
    return predictions, {"half_width": width, "epsilon": epsilon, "attempts": attempts,
                         "fit_seconds": fit_seconds,
                         "predict_seconds": time.perf_counter() - predict_start}


def acquire_batch(predictions, observed_indices, count, fps, budget, beta=3.5,
                  stop=lambda: False):
    """Exact relevance+coverage greedy batch, with GP predictions held fixed."""
    import numpy as np
    mu = np.asarray(predictions, dtype=np.float64)
    n = len(mu)
    observed = sorted(set(int(i) for i in observed_indices))
    if (n < 1 or not np.isfinite(mu).all() or not math.isfinite(fps) or fps <= 0
            or beta < 0 or count < 0 or count > n - len(observed) or budget < 1
            or any(i < 0 or i >= n for i in observed)):
        raise ValueError("Invalid acquisition inputs")
    if count == 0:
        return []
    if n == 1:
        return [0]
    # Express gaps in source-index units: dividing both gap and tau by fps
    # cancels exactly. Virtual endpoints permit unobserved boundary frames.
    tau = (n - 1) / (budget + 1)
    heap = []

    def add_gap(left, right):
        if right - left <= 1:
            return
        ids = np.arange(left + 1, right)
        l, r = max(0, left), min(n - 1, right)
        gains = (np.log1p((ids - l) / tau) + np.log1p((r - ids) / tau)
                 - math.log1p((r - l) / tau))
        values = mu[ids] + beta * gains
        position = int(np.argmax(values))  # Earlier source frame breaks ties.
        heapq.heappush(heap, (-float(values[position]), int(ids[position]), left, right))

    cuts = [-1, *observed, n]
    for left, right in zip(cuts, cuts[1:]):
        if stop():
            raise InterruptedError("Stop requested during FORTE acquisition")
        add_gap(left, right)
    result = []
    for _ in range(count):
        if stop():
            raise InterruptedError("Stop requested during FORTE acquisition")
        _, index, left, right = heapq.heappop(heap)
        result.append(index)
        # Recompute only candidates in the split gap after each acquisition.
        add_gap(left, index)
        add_gap(index, right)
    return result


def coverage(times, duration, k):
    return sum(math.log1p((right - left) / (duration / (k + 1)))
               for left, right in zip([0., *times], [*times, duration])) if duration else 0.


def exact_keyframes(times, scores, k, duration, beta=1., stop=lambda: False):
    """O(KN) monotone-envelope DP over measured frames, Appendix A.3."""
    import numpy as np
    t = np.asarray(times, dtype=np.float64)
    relevance = np.asarray(scores, dtype=np.float64)
    n = len(t)
    if (t.ndim != 1 or relevance.shape != t.shape or not n or not 1 <= k <= n
            or not np.isfinite(t).all() or not np.isfinite(relevance).all()
            or np.any(np.diff(t) <= 0) or not math.isfinite(duration) or duration < t[-1]
            or t[0] < 0 or not math.isfinite(beta) or beta < 0):
        raise ValueError("Invalid exact keyframe optimization inputs")
    if beta == 0 or duration == 0 or k == n:
        chosen = sorted(sorted(range(n), key=lambda i: (-relevance[i], i))[:k])
        return chosen, float(relevance[chosen].sum() + beta * coverage(t[chosen].tolist(), duration, k))
    tau = duration / (k + 1)
    previous = relevance + beta * np.log1p(t / tau)
    predecessors = np.full((k, n), -1, dtype=np.int32)
    for layer in range(1, k):
        if stop():
            raise InterruptedError("Stop requested during FORTE exact optimization")
        current = np.full(n, -np.inf)
        hull = deque()
        record = -math.inf

        def intersection(p, q):
            delta = (previous[q] - previous[p]) / beta
            denominator = -math.expm1(-delta)
            return t[p] - tau + (t[q] - t[p]) / denominator

        def value(j, x):
            return previous[j] + beta * math.log1p((x - t[j]) / tau)

        for i in range(layer, n):
            j = i - 1
            if previous[j] > record:
                record = previous[j]
                while len(hull) >= 2 and intersection(hull[-2], hull[-1]) >= intersection(hull[-1], j):
                    hull.pop()
                hull.append(j)
            while len(hull) >= 2 and value(hull[1], t[i]) >= value(hull[0], t[i]):
                hull.popleft()
            best = hull[0]
            current[i] = relevance[i] + value(best, t[i])
            predecessors[layer, i] = best
        previous = current
    terminal = previous + beta * np.log1p((duration - t) / tau)
    index = int(np.argmax(terminal))
    objective = float(terminal[index])
    chosen = [index]
    for layer in range(k - 1, 0, -1):
        index = int(predecessors[layer, index])
        chosen.append(index)
    return list(reversed(chosen)), objective


def supplement(indices, n, k):
    """Preserve scored selection; add farthest original frames, earlier on ties."""
    import numpy as np
    chosen = list(indices)
    target = min(n, k)
    if not chosen or chosen != sorted(set(chosen)) or any(i < 0 or i >= n for i in chosen):
        raise ValueError("Invalid selection to supplement")
    if len(chosen) >= target:
        return chosen
    ids = np.arange(n, dtype=np.int64)
    distances = np.full(n, n, dtype=np.int64)
    for index in chosen:
        np.minimum(distances, np.abs(ids - index), out=distances)
    while len(chosen) < target:
        index = int(np.argmax(distances))
        chosen.append(index)
        np.minimum(distances, np.abs(ids - index), out=distances)
    return sorted(chosen)


def select_forte(video, question, config, scorer, stop=lambda: False):
    import numpy as np
    n, fps = len(video), float(video.get_avg_fps())
    reference = reference_grid(n, fps, config.forte_rho)
    budget = len(reference)
    initial = reference_grid(n, fps, config.forte_rho / 2)
    observed = np.zeros(n, dtype=bool)
    scores = np.full(n, np.nan, dtype=np.float64)
    scorer.reset()
    scorer_seconds = 0.

    def measure(indices):
        nonlocal scorer_seconds
        for start in range(0, len(indices), config.blip_batch_size):
            if stop():
                raise InterruptedError("Stop requested during FORTE scoring")
            batch = indices[start:start + config.blip_batch_size]
            if any(observed[i] for i in batch):
                raise ValueError("FORTE attempted duplicate scorer evaluation")
            began = time.perf_counter()
            values = np.asarray(scorer(video, question.question, batch), dtype=np.float64)
            scorer_seconds += time.perf_counter() - began
            if values.shape != (len(batch),) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise ValueError("Invalid FORTE BLIP score count/probability")
            scores[batch] = values
            observed[batch] = True

    measure(initial)
    remaining = budget - len(initial)
    if remaining < 0:
        raise ValueError("FORTE initialization exceeded reference budget")
    base, extra = divmod(remaining, config.forte_rounds)
    rounds = []
    duration = (n - 1) / fps
    for round_index in range(config.forte_rounds):
        allowance = base + (round_index < extra)
        if allowance == 0:
            continue
        obs = np.flatnonzero(observed)
        unscored = np.flatnonzero(~observed)
        means, gp = gp_predict(obs / fps, normalize(scores[obs]), unscored / fps,
                               length_scale=config.forte_length_scale, stop=stop)
        predictions = np.zeros(n, dtype=np.float64)
        predictions[unscored] = means
        began = time.perf_counter()
        batch = acquire_batch(predictions, obs, allowance, fps, budget,
                              beta=config.forte_beta_acq, stop=stop)
        acquisition_seconds = time.perf_counter() - began
        measure(batch)
        rounds.append({"round": round_index + 1, "observed_before": len(obs),
                       "acquired_indices": batch, "acquisition_seconds": acquisition_seconds, **gp})
    pool = np.flatnonzero(observed)
    if len(pool) != budget or len(scorer.evaluated) != budget or len(set(scorer.evaluated)) != budget:
        raise ValueError("FORTE scorer accounting does not match budget")
    final_scores = normalize(scores[pool])
    if stop():
        raise InterruptedError("Stop requested before FORTE final selection")
    began = time.perf_counter()
    if len(pool) >= config.frames:
        positions, objective = exact_keyframes(pool / fps, final_scores, config.frames,
                                               duration, beta=config.forte_beta_sel, stop=stop)
        chosen = [int(pool[i]) for i in positions]
    else:
        chosen, objective = pool.tolist(), None
    optimized_count = len(chosen)
    chosen = supplement(chosen, n, config.frames)
    optimization_seconds = time.perf_counter() - began
    if chosen != sorted(set(chosen)) or len(chosen) != min(n, config.frames):
        raise ValueError("Invalid FORTE final selection")
    return chosen, {
        "scorer_frame_evaluations": budget, "scorer_unique_frames": budget,
        "scorer_batches": scorer.calls, "forte_implementation": IMPLEMENTATION,
        "forte_rho": config.forte_rho, "forte_timestamp_span_seconds": duration,
        "forte_scoring_budget": budget, "forte_initial_indices": initial,
        "forte_scored_indices": pool.tolist(), "forte_raw_scores": scores[pool].tolist(),
        "forte_rounds": rounds, "forte_selection_objective": objective,
        "forte_supplemental_frames": len(chosen) - optimized_count,
        "forte_source_shorter_than_k": n < config.frames,
        "forte_scorer_wall_seconds": scorer_seconds,
        "forte_optimization_seconds": optimization_seconds,
    }
