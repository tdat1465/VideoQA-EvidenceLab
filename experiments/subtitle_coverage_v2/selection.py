"""Replace subtitle frames while preserving temporal anchors and occupied bins."""
from collections import Counter
import math

POLICY = {
    "version": "subtitle-protected-replacement-v2",
    "time_bins": 16,
    "subtitle_budget": 16,
    "subtitle_matches_per_query": 3,
    "local_rule": "same-bin-nearest-then-lowest-blip",
    "fallback_rule": "lowest-blip-in-redundantly-covered-bin",
}


def subtitle_targets(count, fps, record):
    """Preserve v1 proposals and ordering to isolate the replacement policy."""
    if count < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Require positive source frame count and FPS")
    offset = float(record["offset"] or 0)
    if not math.isfinite(offset):
        raise ValueError("Invalid subtitle offset")
    anchors = []
    for matches in record["matches"]:
        possible = []
        for _, _, start, end in matches["candidates"]:
            if not all(math.isfinite(v) for v in (start, end)) or start > end:
                raise ValueError("Invalid subtitle interval")
            timestamp = (start + end) / 2 - offset
            if 0 <= timestamp < count / fps:
                possible.append(timestamp)
        anchors.append(possible[:3])
    times = list(dict.fromkeys(
        t for rank in range(3) for group in anchors for t in group[rank:rank + 1]
    ))
    return list(dict.fromkeys(
        max(0, min(count - 1, round((t + shift) * fps)))
        for shift in (0, 1, -1, 2, -2, 4, -4, 8, -8)
        for t in times
    ))[:16]


def protected_select(base, targets, count, candidates, relevance):
    """Keep grid representatives; replace original frames one-for-one.

    Relevance is question-only BLIP probability on each base frame. All accepted
    targets are retained. A target is skipped when no safe replacement exists.
    """
    if (not base or base != sorted(set(base)) or len(base) > 64
            or base[0] < 0 or base[-1] >= count):
        raise ValueError("Require 1-64 unique chronological source-frame indices")
    if (not candidates or candidates != sorted(set(candidates))
            or candidates[0] < 0 or candidates[-1] >= count):
        raise ValueError("Invalid candidate grid")
    if len(targets) > 16 or len(set(targets)) != len(targets):
        raise ValueError("Expected at most 16 unique subtitle targets")
    if any(type(i) is not int or not 0 <= i < count for i in targets):
        raise ValueError("Subtitle target is outside the source video")
    if set(relevance) != set(base) or any(
        not math.isfinite(v) or not 0 <= v <= 1 for v in relevance.values()
    ):
        raise ValueError("Require one finite BLIP probability per base frame")

    # Same endpoint-inclusive 16-position grid as event coverage. If its exact
    # frame is absent (RIDGE fallback), protect the nearest existing base frame.
    grid_size = min(16, len(candidates))
    grid = [candidates[i * (len(candidates) - 1) // max(1, grid_size - 1)]
            for i in range(grid_size)]
    mandatory = {min(base, key=lambda f: (abs(f - anchor), f)) for anchor in grid}
    mandatory.update((base[0], base[-1]))
    protected = mandatory | (set(base) & set(targets))
    selected = set(base)
    original = set(base)
    bin_count = min(16, len(base))

    def time_bin(frame):
        return min(bin_count - 1, frame * bin_count // count)

    original_bins = {time_bin(f) for f in base}
    swaps, skipped = [], []
    for target in targets:
        if target in selected:
            continue
        augmented = selected | {target}
        occupancy = Counter(time_bin(f) for f in augmented)
        removable = [f for f in selected & original - protected
                     if occupancy[time_bin(f)] > 1]
        local = [f for f in removable if time_bin(f) == time_bin(target)]
        if local:
            victim = min(local, key=lambda f: (abs(f - target), relevance[f], f))
            reason = "nearby-same-bin"
        elif removable:
            victim = min(removable, key=lambda f: (
                relevance[f], min(abs(f - other) for other in augmented if other != f), f
            ))
            reason = "low-relevance-covered-bin"
        else:
            skipped.append(target)
            continue
        selected.remove(victim)
        selected.add(target)
        protected.add(target)
        swaps.append({"added": target, "removed": victim, "reason": reason,
                      "removed_blip": relevance[victim]})
    if (len(selected) != len(base) or not mandatory <= selected
            or not original_bins <= {time_bin(f) for f in selected}):
        raise AssertionError("Replacement violated the frame budget or coverage")
    return sorted(selected), {"mandatory": sorted(mandatory), "swaps": swaps,
                              "skipped_targets": skipped,
                              "occupied_bins_before": len(original_bins),
                              "occupied_bins_after": len({time_bin(f) for f in selected})}
