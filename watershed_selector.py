"""Deterministic, NumPy-only 1D marker-controlled watershed selection.

AKS supplies terminal leaves and their existing quotas. Nothing here decides
where AKS splits, changes relevance scores, or reallocates unused leaf slots.
"""

from dataclasses import dataclass
import heapq
import math

import numpy as np


@dataclass(frozen=True)
class WatershedConfig:
    sigma: float = 1.0
    min_prominence: float = 0.05
    min_distance: int = 3
    prominence_weight: float = 0.25

    def __post_init__(self):
        for name in ("sigma", "min_prominence", "prominence_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if isinstance(self.min_distance, bool) or not isinstance(self.min_distance, int) or self.min_distance < 1:
            raise ValueError("min_distance must be a positive integer")


def smooth_curve(scores, sigma):
    """Gaussian convolution with replicated edges; sigma is candidate ticks."""
    scores = np.asarray(scores, dtype=float)
    if len(scores) < 2 or sigma == 0:
        return scores.copy()
    # A huge sigma cannot need a kernel larger than the signal's support.
    radius = min(len(scores) - 1, int(math.ceil(3 * min(sigma, len(scores) - 1))))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(np.pad(scores, radius, mode="edge"), kernel, mode="valid")


def peak_markers(curve, min_prominence):
    """One marker per maximal plateau, including one-sided endpoint peaks.

    Prominence is height minus the higher of the two basin minima before a
    strictly higher peak or the domain edge. At an endpoint only the available
    side is used. A flat domain has no prominent peak. All computations use
    AKS's *global* normalized scale, never per-leaf min/max normalization.
    """
    n = len(curve)
    if n == 0 or np.ptp(curve) <= 1e-12:
        return []
    # Nearest strictly higher sample on each side (equal peaks do not bound).
    left = np.full(n, -1, dtype=int)
    right = np.full(n, n, dtype=int)
    stack = []
    for i in range(n):
        while stack and curve[stack[-1]] <= curve[i]:
            stack.pop()
        if stack:
            left[i] = stack[-1]
        stack.append(i)
    stack = []
    for i in range(n - 1, -1, -1):
        while stack and curve[stack[-1]] <= curve[i]:
            stack.pop()
        if stack:
            right[i] = stack[-1]
        stack.append(i)

    # Range-minimum segment tree keeps prominence queries O(log T).
    size = 1 << (n - 1).bit_length()
    tree = np.full(2 * size, np.inf)
    tree[size:size + n] = curve
    for j in range(size - 1, 0, -1):
        tree[j] = min(tree[2 * j], tree[2 * j + 1])

    def minimum(lo, hi):
        lo, hi = lo + size, hi + size  # half-open
        value = np.inf
        while lo < hi:
            if lo & 1:
                value = min(value, tree[lo])
                lo += 1
            if hi & 1:
                hi -= 1
                value = min(value, tree[hi])
            lo //= 2
            hi //= 2
        return value

    peaks = []
    start = 0
    while start < n:
        end = start
        while end + 1 < n and curve[end + 1] == curve[start]:
            end += 1
        if (start == 0 or curve[start] > curve[start - 1]) and (end == n - 1 or curve[end] > curve[end + 1]):
            marker = (start + end) // 2
            minima = []
            if start > 0:
                minima.append(minimum(max(0, int(left[marker])), start + 1))
            if end < n - 1:
                minima.append(minimum(end, min(n, int(right[marker]) + 1)))
            prominence = float(curve[marker] - max(minima)) if minima else 0.0
            if prominence > 1e-12 and prominence >= min_prominence:
                peaks.append((marker, prominence))
        start = end + 1
    return peaks


def watershed_labels(curve, markers):
    """Flood -curve from peak markers on the 1D neighbor graph.

    Priority is the maximum elevation encountered along the flood path. FIFO
    ties give deterministic plateau ownership. This is a watershed operation,
    rather than merely taking local maxima or cutting fixed temporal bins.
    """
    labels = np.full(len(curve), -1, dtype=int)
    queue = []
    age = 0
    for label, (marker, _) in enumerate(markers):
        labels[marker] = label
        heapq.heappush(queue, (-float(curve[marker]), age, marker, label))
        age += 1
    while queue:
        level, _, index, label = heapq.heappop(queue)
        for neighbor in (index - 1, index + 1):
            if 0 <= neighbor < len(curve) and labels[neighbor] == -1:
                labels[neighbor] = label
                heapq.heappush(queue, (max(level, -float(curve[neighbor])), age, neighbor, label))
                age += 1
    return labels


def select_leaves(leaves, config=None):
    """Return global candidate positions and auditable diagnostics.

    leaves: dictionaries with globally normalized ``scores``, chronological
    candidate ``positions`` and AKS's integer ``quota``. NMS spans leaf borders.
    Shortfall refill stays in the same leaf, prefers maximum distance, then raw
    relevance, then earlier time. Distance is relaxed only if no remaining
    candidate in any unfinished leaf can meet it. No selected peak is discarded.
    """
    config = config or WatershedConfig()
    capacities = [min(leaf["quota"], len(leaf["scores"])) for leaf in leaves]
    counts = [0] * len(leaves)
    candidates = []
    diagnostics = {"leaves": [], "selections": [], "distance_relaxations": 0}
    remaining = {}
    for leaf_id, leaf in enumerate(leaves):
        scores = np.asarray(leaf["scores"], dtype=float)
        positions = leaf["positions"]
        curve = smooth_curve(scores, config.sigma)
        markers = peak_markers(curve, config.min_prominence)
        labels = watershed_labels(curve, markers)
        basin_details = []
        for label, (marker, prominence) in enumerate(markers):
            basin = np.flatnonzero(labels == label)
            # Highest *unsmoothed* relevance, then proximity to the marker.
            best = min(basin.tolist(), key=lambda i: (-scores[i], abs(i - marker), i))
            pos = int(positions[best])
            relevance = float(scores[best])
            utility = relevance + config.prominence_weight * prominence
            candidates.append((-utility, -relevance, -prominence, pos, leaf_id))
            basin_details.append({"marker": int(positions[marker]), "representative": pos,
                                  "prominence": prominence, "relevance": relevance,
                                  "start": int(positions[basin[0]]), "end": int(positions[basin[-1]])})
        diagnostics["leaves"].append({"quota": leaf["quota"], "capacity": capacities[leaf_id],
                                      "basins": basin_details})
        if capacities[leaf_id]:
            remaining.update({int(pos): (leaf_id, float(score)) for pos, score in zip(positions, scores)})

    selected = []

    def add(pos, leaf_id, reason):
        selected.append(pos)
        counts[leaf_id] += 1
        remaining.pop(pos)
        diagnostics["selections"].append({"position": pos, "leaf": leaf_id, "reason": reason})

    for _, _, _, pos, leaf_id in sorted(candidates):
        if counts[leaf_id] < capacities[leaf_id] and all(abs(pos - other) >= config.min_distance for other in selected):
            add(pos, leaf_id, "peak")

    while sum(counts) < sum(capacities):
        pool = []
        for pos, (leaf_id, score) in remaining.items():
            if counts[leaf_id] < capacities[leaf_id]:
                distance = min((abs(pos - other) for other in selected), default=float("inf"))
                pool.append((distance, score, -pos, leaf_id))
        allowed = [item for item in pool if item[0] >= config.min_distance]
        # No selected frames: highest relevance, stable earliest-time tie.
        best = max(allowed or pool)
        distance, _, neg_pos, leaf_id = best
        relaxed = distance < config.min_distance
        diagnostics["distance_relaxations"] += int(relaxed)
        add(-neg_pos, leaf_id, "relaxed_refill" if relaxed else "refill")

    return sorted(selected), diagnostics
