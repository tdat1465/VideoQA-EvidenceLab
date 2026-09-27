from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from .store import atomic_json, read_run


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def percentile(values, fraction):
    return sorted(values)[round((len(values)-1)*fraction)] if values else None


def summarize(contract, results):
    labelled = [r for r in results if r["answer"] is not None]
    groups = defaultdict(list)
    for r in labelled:
        groups[r["question_type"]].append(r["correct"])
    category = {key: {"n": len(values), "accuracy": mean(values)} for key, values in sorted(groups.items())}
    nextqa = defaultdict(list)
    for r in labelled:
        if r["dataset"] == "nextqa":
            nextqa[r["question_type"][:1].upper()].append(r["correct"])
    latency = [r["elapsed_seconds"] for r in results]
    vram = [r["peak_vram_allocated_gib"] for r in results if r["peak_vram_allocated_gib"] is not None]
    flips = {"wrong_to_right": 0, "right_to_wrong": 0}
    for row in labelled:
        initial = row["calls"][0]["scores"]
        initial_correct = max(range(len(initial)), key=initial.__getitem__) == row["answer"]
        flips["wrong_to_right"] += int(not initial_correct and row["correct"])
        flips["right_to_wrong"] += int(initial_correct and not row["correct"])
    return {"synthetic": contract["config"]["backend"] == "mock",
            "resolved_dtype": contract.get("resolved_dtype"),
            "nodes": sorted({r["execution"]["node"] for r in results if "execution" in r}),
            "complete": len(results) == contract["selected_count"],
            "selected_count": contract["selected_count"], "completed_count": len(results),
            "labelled_completed": len(labelled), "accuracy_completed": mean([r["correct"] for r in labelled]),
            "per_type": category,
            "macro_type_accuracy": mean([v["accuracy"] for v in category.values()]),
            "nextqa_CTD": {k: {"n": len(v), "accuracy": mean(v)} for k, v in sorted(nextqa.items())},
            "latency_mean_seconds": mean(latency), "latency_p50_seconds": percentile(latency, .5),
            "latency_p95_seconds": percentile(latency, .95), "measured_question_seconds": sum(latency),
            "mean_input_tokens": mean([r["input_tokens"] for r in results]),
            "mean_vlm_calls": mean([len(r["calls"]) for r in results]),
            "mean_frame_presentations": mean([r["frame_presentations"] for r in results]),
            "mean_final_frames": mean([r["final_frames"] for r in results]),
            "refinement_rate": mean([r["refined"] for r in results]), "answer_changes": flips,
            "peak_vram_allocated_gib": max(vram) if vram else None,
            "timing_scope": "video verification/decode + selection + all answer calls; excludes model load/download"}


def export(directory: Path, contract, results):
    summary = summarize(contract, results)
    atomic_json(directory / "summary.json", summary)
    with (directory / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    with (directory / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["id", "video", "dataset", "split", "question_type", "prediction", "answer", "correct",
                  "elapsed_seconds", "input_tokens", "final_frames", "refined", "peak_vram_allocated_gib"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    return summary


def compare(left: Path, right: Path, iterations=2000, seed=18):
    if iterations < 1:
        raise ValueError("iterations must be positive")
    a, ra = read_run(left)
    b, rb = read_run(right)
    if a.get("resolved_dtype") != b.get("resolved_dtype"):
        raise ValueError("Paired comparison requires the same resolved inference precision")
    if a["source_sha256"] != b["source_sha256"] or a["runtime"] != b["runtime"]:
        raise ValueError("Use the same source and runtime for paired experiments")
    if a["selected_ids_hash"] != b["selected_ids_hash"] or a["manifest_sha256"] != b["manifest_sha256"]:
        raise ValueError("Paired comparison requires the same dataset and selected IDs")
    for key in ("backend", "model_id", "revision", "prompt_version", "max_image_pixels", "candidate_frames"):
        if a["config"][key] != b["config"][key]:
            raise ValueError(f"Backbone/protocol mismatch: {key}")
    if len(ra) != a["selected_count"] or len(rb) != b["selected_count"]:
        raise ValueError("Complete both runs before paired comparison")
    by_id = {r["id"]: r for r in rb}
    clusters = defaultdict(lambda: [0, 0])
    for r in ra:
        if r["answer"] is None:
            raise ValueError("Paired accuracy requires labels for every selected question")
        other = by_id[r["id"]]
        if r["answer"] != other["answer"]:
            raise ValueError("Labels differ")
        # Bootstrap whole source videos, not correlated questions independently.
        group = clusters[(r["dataset"], r["video"])]
        group[0] += int(other["correct"]) - int(r["correct"])
        group[1] += 1
    values = list(clusters.values())
    rng, bootstrap = random.Random(seed), []
    for _ in range(iterations):
        draws = rng.choices(values, k=len(values))
        bootstrap.append(sum(x[0] for x in draws)/sum(x[1] for x in draws))
    return {"synthetic": a["config"]["backend"] == "mock", "videos": len(values), "questions": len(ra),
            "delta_accuracy_right_minus_left": sum(x[0] for x in values)/len(ra),
            "video_cluster_bootstrap_95_ci": [percentile(bootstrap, .025), percentile(bootstrap, .975)],
            "left": summarize(a, ra), "right": summarize(b, rb)}
