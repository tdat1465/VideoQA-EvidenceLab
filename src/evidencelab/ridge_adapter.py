"""SHA-verified author RIDGE selector; no reimplementation or uniform fallback."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import urllib.request

from .aks_adapter import aks_candidates
from .data import file_hash

RIDGE_REVISION = "0f710c7304e806ccedc27b0d177241cb14d77bba"
RIDGE_FILES = {
    "frame_select.py": "7897d51979def23f83f4ce8e406b27b981f44ed6d8b1886eaff41241c1c83744",
    "weight.py": "0048dfc74e9a26febb14c97207bc3da98f04fb8b84202f9c40fcbe88128e744c",
}
IMPLEMENTATION = "author-0f710c7-hf-blip-floor-fps-qonly-v1"
WEIGHT_KEYS = ("peak_similarity", "slope_abs", "rising_slope", "falling_slope",
               "boundary_change", "context_density")


def fetch_ridge(directory: Path):
    """Fetch two immutable upstream files into allocation RAM, not our bundle."""
    import hashlib
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in RIDGE_FILES.items():
        path = directory / name
        if path.exists():
            if file_hash(path) != expected:
                raise ValueError(f"RIDGE source hash mismatch: {name}")
            continue
        url = f"https://raw.githubusercontent.com/Xssq999/RIDGE/{RIDGE_REVISION}/{name}"
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                payload = response.read(1024 * 1024)
        except Exception as error:
            raise RuntimeError(f"Cannot fetch pinned RIDGE {name}: {type(error).__name__}") from None
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"RIDGE downloaded source hash mismatch: {name}")
        temporary = path.with_suffix(".download")
        temporary.write_bytes(payload)
        temporary.replace(path)
    return directory


def load_ridge(directory: Path):
    modules = {}
    for name, expected in RIDGE_FILES.items():
        path = directory / name
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"RIDGE source hash mismatch: {name}")
        spec = importlib.util.spec_from_file_location("evidencelab_author_ridge_" + path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[path.stem] = module
    return modules["frame_select"], modules["weight"]


def validate_weights(weights):
    if not isinstance(weights, dict) or set(weights) != set(WEIGHT_KEYS):
        raise ValueError("RIDGE requires exactly six named evidence weights")
    if any(type(weights[key]) is not int or not 0 <= weights[key] <= 10 for key in WEIGHT_KEYS):
        raise ValueError("RIDGE weights must be integer values in [0, 10]")
    return {key: weights[key] for key in WEIGHT_KEYS}


def selector_args(config):
    # All other defaults are exactly those in the pinned author argument parser.
    return SimpleNamespace(sigma=config.ridge_sigma, adaptive_sigma=config.ridge_adaptive_sigma,
                           peak_prominence=.15, base_peak_expand=3, max_peak_expand=10,
                           slope_extend_k=.5, max_extend_factor=4., boundary_k=2.)


def select_ridge(video, question, config, scorer, author, weight_record, stop=lambda: False):
    weights = validate_weights(weight_record["weights"])
    candidates = aks_candidates(len(video), float(video.get_avg_fps()))
    scorer.reset()
    scores = []
    for start in range(0, len(candidates), config.blip_batch_size):
        if stop():
            raise InterruptedError("Stop requested during RIDGE scoring")
        batch = candidates[start:start + config.blip_batch_size]
        values = scorer(video, question.question, batch)
        if len(values) != len(batch) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("RIDGE BLIP scores do not match candidate frames")
        scores.extend(values)
    # The official core has a T<5 edge case when K<T; do not silently patch it.
    if config.frames < len(scores) < 5:
        raise ValueError("Pinned author RIDGE does not support K<T<5; use K>=5")
    positions = [int(i) for i in author.gradselect(scores, config.frames, weights, selector_args(config))]
    if (positions != sorted(set(positions)) or len(positions) != min(config.frames, len(candidates))
            or not positions or positions[0] < 0 or positions[-1] >= len(candidates)):
        raise ValueError("Invalid author RIDGE selected positions; no fallback was applied")
    indices = [candidates[i] for i in positions]
    return indices, {
        "scorer_frame_evaluations": len(scorer.evaluated),
        "scorer_unique_frames": len(set(scorer.evaluated)), "scorer_batches": scorer.calls,
        "ridge_revision": RIDGE_REVISION, "ridge_implementation": IMPLEMENTATION,
        "ridge_candidate_count": len(candidates), "ridge_candidate_implementation": "aks-floor-fps-v1",
        "ridge_candidate_positions": positions, "ridge_fallback": False,
        "ridge_weights": weights, "ridge_weight_record_sha256": weight_record["record_sha256"],
        "ridge_weight_generation_seconds": weight_record["generation_seconds"],
        "ridge_weight_output_tokens": weight_record["output_tokens"],
        "auxiliary_input_tokens": weight_record["input_tokens"],
        "ridge_sigma": config.ridge_sigma, "ridge_adaptive_sigma": config.ridge_adaptive_sigma,
    }
