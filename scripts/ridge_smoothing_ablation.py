"""Separate 200-question RIDGE smoothing experiment; frozen baseline stays read-only.

Author source is fetched and SHA-verified in job RAM. Small score/curve metadata
and results persist; videos, libraries and model weights remain in tmpfs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import gc
import json
import math
from pathlib import Path
import random
import signal
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evidencelab.data import digest, file_hash
from evidencelab.store import Store, atomic_json, run_lock
from evidencelab.ridge_adapter import RIDGE_REVISION, RIDGE_FILES, selector_args, validate_weights

VARIANTS = ("sigma2", "sigma1", "sigma4", "adaptive", "multiscale_v1")
BASE_SOURCE = "e92cfc762bde69a511beb494a4b5305bd43e89ccdb72d599de938366da1be057"
PROTOTYPE = {"scales": [1., 2., 4.], "vote": "majority-else-sigma2",
             "rescue_radius": 1, "contrast_radius": 12, "contrast_exclude": 3,
             "minimum_contrast": .15, "minimum_support": 2,
             "support_normalized_score": .5, "ranking": "sigma2-with-sigma1-rescue"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    return value


def choose_development(baseline_rows, count=200, seed=18):
    """Only IDs/video/candidate count enter sampling; no outcomes or answer labels."""
    groups = defaultdict(list)
    for row in baseline_rows:
        if row["ridge_candidate_count"] >= 64:
            groups[row["video"]].append(row["id"])
    videos = sorted(groups)
    random.Random(seed).shuffle(videos)
    chosen, chosen_videos = [], []
    for video in videos:
        ids = sorted(groups[video])
        if len(chosen) + len(ids) <= count:
            chosen.extend(ids)
            chosen_videos.append(video)
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise ValueError("Cannot select the requested whole-video question budget")
    return {"schema": 1, "seed": seed, "count": count, "ids": sorted(chosen),
            "videos": sorted(chosen_videos), "sampling": "shuffled-video-groups-candidates-ge64-v1",
            "note": "Development data; exclude all these videos from any later video-disjoint split."}


def args_for(variant):
    args = selector_args(SimpleNamespace(ridge_sigma=2., ridge_adaptive_sigma=False))
    if variant == "adaptive":
        args.adaptive_sigma = True
    elif variant in ("sigma1", "sigma2", "sigma4"):
        args.sigma = float(variant[-1])
    else:
        raise ValueError(variant)
    return args


def select_variant(author, raw_scores, weights, variant, k=64):
    import numpy as np
    scores = np.asarray(raw_scores, dtype=np.float64)
    if len(scores) < k or k < 5 or not np.isfinite(scores).all():
        raise ValueError("This development experiment requires at least K>=5 scored candidates")
    weights = validate_weights(weights)
    span = float(scores.max() - scores.min())
    normalized = (scores - scores.min()) / span if span > 1e-9 else scores.copy()
    intent = author.compute_coverage_intent(weights)
    if variant != "multiscale_v1":
        args = args_for(variant)
        positions = [int(x) for x in author.gradselect(scores, k, weights, args)]
        labels, curve = author.analyze_curve(normalized, args, intent)
        details = {"labels": labels, "curve": curve,
                   "allocation": author.allocate_frames(weights, labels, k), "rescued_peaks": []}
    else:
        analyses = [author.analyze_curve(normalized, args_for(name), intent)
                    for name in ("sigma1", "sigma2", "sigma4")]
        mid_labels, mid_curve = analyses[1]
        labels = []
        for i in range(len(scores)):
            votes = Counter(item[0][i] for item in analyses)
            role, votes_count = votes.most_common(1)[0]
            labels.append(role if votes_count >= 2 else mid_labels[i])
        # A fine-only short peak can be real: retain a small core only with
        # raw local contrast and two adjacent high-score samples. Frozen v1
        # thresholds are a hypothesis, not an author method or a proven gain.
        rescued = []
        ranking = np.array(mid_curve["s_smooth"], copy=True)
        fine_curve = analyses[0][1]
        for peak in fine_curve["peaks"]:
            if any(abs(peak - p) <= 2 for curve in analyses[1:] for p in curve[1]["peaks"]):
                continue
            radius, outside, exclude = (PROTOTYPE[key] for key in
                                        ("rescue_radius", "contrast_radius", "contrast_exclude"))
            lo, hi = max(0, peak-radius), min(len(scores), peak+radius+1)
            flank = np.r_[normalized[max(0, peak-outside):max(0, peak-exclude)],
                          normalized[min(len(scores), peak+exclude+1):min(len(scores), peak+outside+1)]]
            if not len(flank):
                continue
            contrast = float(normalized[lo:hi].max() - np.median(flank))
            support = int(np.sum(normalized[lo:hi] >= PROTOTYPE["support_normalized_score"]))
            if contrast >= PROTOTYPE["minimum_contrast"] and support >= PROTOTYPE["minimum_support"]:
                rescued.append(int(peak))
                labels[lo:hi] = ["peak"] * (hi-lo)
                ranking[lo:hi] = fine_curve["s_smooth"][lo:hi]
        allocation = author.allocate_frames(weights, labels, k)
        chosen = []
        abs_slope = np.abs(mid_curve["s_prime"])
        for role, budget in allocation.items():
            region = [i for i, name in enumerate(labels) if name == role]
            if role in ("rising", "falling") and budget >= 3 and len(region) >= 3:
                part = author.select_by_temporal_bins(region, ranking, budget)
            else:
                values = abs_slope if role == "boundary" else ranking
                part = author.select_by_score(region, values, budget, min_gap=2)
            chosen.extend(int(x) for x in part)
        chosen = set(chosen)
        for i in sorted(range(len(scores)), key=lambda t: ranking[t], reverse=True):
            if len(chosen) >= k:
                break
            chosen.add(i)
        positions = sorted(chosen)[:k]
        details = {"labels": labels, "curve": {**mid_curve, "ranking": ranking},
                   "allocation": allocation, "rescued_peaks": rescued,
                   "scale_analyses": [{"labels": roles, "curve": curve} for roles, curve in analyses],
                   "prototype": PROTOTYPE}
    if positions != sorted(set(positions)) or len(positions) != k or not 0 <= positions[0] <= positions[-1] < len(scores):
        raise ValueError("Invalid selection; no uniform fallback")
    return positions, jsonable(details)


def cache_key(contract, sample, video_sha, candidates, fps, source_frames):
    from evidencelab.ridge_weights import question_hash
    return {"schema": 1, "experiment": digest(contract), "question_id": sample.id,
            "question_sha256": question_hash(sample.public_question()), "video_sha256": video_sha,
            "candidate_indices": candidates, "fps": fps, "source_frames": source_frames}


def score_record(key, scores, seconds, batches):
    if (len(scores) != len(key["candidate_indices"]) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores)
            or not math.isfinite(seconds) or seconds < 0 or type(batches) is not int or batches < 1):
        raise ValueError("Invalid BLIP score vector")
    record = {"key": key, "scores": scores, "original_scoring_seconds": seconds, "scorer_batches": batches}
    record["sha256"] = digest(record)
    return record


def read_score_cache(path, key):
    record = load(path)
    if record["key"] != key or record["sha256"] != digest({k:v for k,v in record.items() if k != "sha256"}):
        raise ValueError("Score cache contract/checksum changed; use a new run label")
    score_record(key, record["scores"], record["original_scoring_seconds"], record["scorer_batches"])
    return record


def evaluate_question(sample, candidates, fps, weights, score, author, store,
                      diagnostic_path, answer, stop=lambda: False, reference=None):
    """One cache, five serial variants; each answer is committed independently."""
    done, diagnostics = store.done(), {}
    if diagnostic_path.exists():
        diagnostics = load(diagnostic_path)
        if diagnostics.get("score_sha256") != score["sha256"]:
            raise ValueError("Diagnostics do not match the cached scores")
    diagnostics.update(question_id=sample.id, score_sha256=score["sha256"], variants=diagnostics.get("variants", {}))
    for variant in VARIANTS:
        key = sample.id + "::" + variant
        if key in done:
            continue
        if stop():
            raise InterruptedError("Stop between variants; previous answers preserved")
        started = time.perf_counter()
        positions, details = select_variant(author, score["scores"], weights["weights"], variant)
        select_seconds = time.perf_counter() - started
        indices, times = [candidates[i] for i in positions], [candidates[i]/fps for i in positions]
        diagnostics["variants"][variant] = {**details, "selected_positions": positions,
                                             "frame_indices": indices, "timestamps": times}
        atomic_json(diagnostic_path, diagnostics)
        decision, answer_seconds, measurements = answer(indices, times)
        if (len(decision.scores) != len(sample.choices) or any(not math.isfinite(x) or not 0 <= x <= 1 for x in decision.scores)
                or abs(sum(decision.scores)-1) > 1e-4):
            raise ValueError("Invalid answer scores")
        row = {"id": key, "question_id": sample.id, "variant": variant,
               "prediction": decision.prediction, "answer": sample.answer,
               "correct": decision.prediction == sample.answer, "video": sample.video,
               "question_type": sample.question_type, "duration_group": sample.duration_group,
               "frame_indices": indices, "timestamps": times, "final_frames": len(indices),
               "scores": decision.scores, "input_tokens": decision.input_tokens,
               "selector_only_seconds": select_seconds, "answer_decode_seconds": answer_seconds,
               "cached_question_seconds": select_seconds + answer_seconds,
               "score_sha256": score["sha256"], "weight_sha256": weights["record_sha256"],
               "prior_qwen_generation_seconds": weights["generation_seconds"], **measurements}
        if reference is not None and variant == "sigma2":
            row.update(baseline_frame_match=indices == reference["calls"][0]["frame_indices"],
                       baseline_prediction_match=decision.prediction == reference["prediction"])
        store.put(row)
        print(f"Committed {sample.id}: {variant}, correct={row['correct']}", flush=True)


def export(output, contract, store):
    rows = store.results()
    stats = {}
    for variant in VARIANTS:
        values = [r for r in rows if r["variant"] == variant]
        stats[variant] = {"n": len(values), "correct": sum(v["correct"] for v in values),
                          "accuracy": sum(v["correct"] for v in values)/len(values) if values else None,
                          "mean_cached_question_seconds": sum(v["cached_question_seconds"] for v in values)/len(values) if values else None}
    scores = [load(path) for path in sorted((output/"scores").glob("*.json"))]
    summary = {"complete": len(rows) == contract["selected_count"], "selected_questions": len(contract["manifest"]["ids"]),
               "completed_variant_answers": len(rows), "variants": stats,
               "one_time_scored_questions": len(scores),
               "one_time_scoring_seconds": sum(v["original_scoring_seconds"] for v in scores),
               "timing_scope": "Cached timings exclude shared BLIP scoring, prior Qwen, setup/indexing/transfer and diagnostic/result I/O; scoring charged once separately.",
               "interpretation": "Exploratory development ablation, not held-out benchmark evidence."}
    controls = [v for v in rows if v["variant"] == "sigma2" and "baseline_frame_match" in v]
    summary["baseline_reproduction"] = {"n": len(controls),
        "same_frames": sum(v["baseline_frame_match"] for v in controls),
        "same_prediction": sum(v["baseline_prediction_match"] for v in controls)}
    events = [json.loads(value) for (value,) in store.db.execute("SELECT value FROM events")]
    summary["recorded_sessions"] = {
        "model_setup_seconds": sum(v["seconds"] for v in events if v["event"] == "models_ready"),
        "video_transfer_hash_seconds": sum(v["transfer_hash_seconds"] for v in events if v["event"] == "video_loaded"),
        "worker_wall_seconds": sum(v["wall_seconds"] for v in events if v["event"] == "session_end")}
    atomic_json(output/"summary.json", summary)
    with (output/"predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    return summary


def run(args):
    from evidencelab.aks_adapter import aks_candidates
    from evidencelab.backends import HFBackend
    from evidencelab.config import source_fingerprint
    from evidencelab.focus_adapter import BlipITMScorer
    from evidencelab.hardware import execution_profile
    from evidencelab.longvideo import longvideo_runtime, validate_ram_root
    from evidencelab.longvideo_config import LongVideoConfig
    from evidencelab.longvideo_data import (parse_annotations, archive_index, HTTPRangeSource,
        MultipartReader, fetch_video, TRANSFER_CHUNK_BYTES, check_space)
    from evidencelab.ridge_adapter import fetch_ridge, load_ridge
    from evidencelab.ridge_weights import RidgeWeightCache, weight_contract
    import decord
    from PIL import Image
    baseline, output, ram = args.baseline.resolve(), args.output.resolve(), args.ram_root.resolve()
    if output == baseline or baseline in output.parents or output in baseline.parents:
        raise ValueError("Ablation output must be separate from the frozen baseline")
    validate_ram_root(ram)
    config = LongVideoConfig.load(args.config)
    old, baseline_rows, old_summary = load(baseline/"contract.json"), [], load(baseline/"summary.json")
    with (baseline/"predictions.jsonl").open(encoding="utf-8") as handle:
        baseline_rows = [json.loads(line) for line in handle if line.strip()]
    if (source_fingerprint() != BASE_SOURCE or old["source_sha256"] != BASE_SOURCE
            or config.contract() != old["config"] or not old_summary["complete"]
            or len(baseline_rows) != 1337 or digest(sorted(v["id"] for v in baseline_rows)) != old["selected_ids_hash"]
            or file_hash(args.annotations) != old["annotation_sha256"]):
        raise ValueError("Frozen baseline/config/source/annotation mismatch")
    execution = execution_profile(config)
    runtime = longvideo_runtime(config)
    if execution["resolved_dtype"] != old["resolved_dtype"] or runtime != old["runtime"]:
        raise ValueError("Runtime/precision changed; do not silently reuse baseline preferences")
    author, weight_author = load_ridge(fetch_ridge(ram/"ridge-upstream"))
    samples = parse_annotations(load(args.annotations))
    weights = RidgeWeightCache(baseline/"ridge-weights.json", weight_contract(config, old, weight_author),
                               [q.public_question() for q in samples])
    weights.check_completed(baseline_rows)
    manifest = choose_development(baseline_rows)
    selected = sorted((q for q in samples if q.id in set(manifest["ids"])), key=lambda q:(q.video,q.id))
    if len(selected) != 200:
        raise ValueError("Development annotation coverage changed")
    contract = {"schema": "ridge-smoothing-ablation-v1", "source_sha256": BASE_SOURCE,
                "experiment_script_sha256": file_hash(Path(__file__)), "config": config.contract(),
                "runtime": runtime, "resolved_dtype": execution["resolved_dtype"],
                "baseline_contract_sha256": digest(old), "baseline_weights_sha256": file_hash(baseline/"ridge-weights.json"),
                "baseline_predictions_sha256": file_hash(baseline/"predictions.jsonl"),
                "manifest": manifest, "variants": list(VARIANTS), "prototype": PROTOTYPE,
                "selected_count": 200 * len(VARIANTS), "author_source": RIDGE_FILES}
    stopped, old_handlers = [], {}
    for name in ("SIGINT", "SIGTERM", "SIGUSR1"):
        number = getattr(signal, name, None)
        if number is not None:
            old_handlers[number] = signal.signal(number, lambda signum, frame: stopped.append(signum))
    started = time.monotonic()
    stop = lambda: bool(stopped) or time.monotonic()-started >= args.max_seconds
    video, reader, backend, scorer = None, None, None, None
    video_path = ram/"ablation-active-video.mp4"
    baseline_by_id = {v["id"]:v for v in baseline_rows}
    try:
        with run_lock(output):
            store = Store(output, contract, (output/"results.sqlite3").exists())
            try:
                (output/"scores").mkdir(exist_ok=True)
                (output/"curves").mkdir(exist_ok=True)
                atomic_json(output/"manifest.json", manifest)
                done = store.done()
                pending = [q for q in selected if any(q.id+"::"+v not in done for v in VARIANTS)]
                if not pending:
                    return 0 if export(output, contract, store)["complete"] else 75
                if stop():
                    raise InterruptedError("Stop before setup")
                source = HTTPRangeSource(stop=stop)
                reader = MultipartReader(source)
                members = archive_index(reader, args.index, {q.video for q in selected})
                reader.clear_cache()
                reader.block_size = TRANSFER_CHUNK_BYTES
                setup_start = time.perf_counter()
                backend, scorer = HFBackend(config), BlipITMScorer(config)
                store.event({"event":"models_ready", "seconds":time.perf_counter()-setup_start, "execution":execution})
                current_video, completed_questions = None, 0
                for sample in pending:
                    if stop() or (args.max_new_questions and completed_questions >= args.max_new_questions):
                        break
                    if sample.video != current_video:
                        video = None
                        gc.collect()
                        video_path.unlink(missing_ok=True)
                        transfer_start = time.perf_counter()
                        video_sha = fetch_video(reader, members["videos/"+sample.video], video_path, stop=stop)
                        reader.clear_cache()
                        store.event({"event":"video_loaded", "video":sample.video,
                                     "transfer_hash_seconds":time.perf_counter()-transfer_start})
                        video = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=2)
                        current_video = sample.video
                    prior = baseline_by_id[sample.id]
                    if video_sha != prior["video_sha256"]:
                        raise ValueError("Source video bytes changed")
                    fps = float(video.get_avg_fps())
                    if len(video) == 0 or not math.isfinite(fps) or fps <= 0:
                        raise ValueError("Invalid source video FPS/length")
                    candidates = aks_candidates(len(video), fps)
                    if len(candidates) != prior["ridge_candidate_count"] or len(candidates) < 64:
                        raise ValueError("Candidate grid differs from the frozen baseline")
                    key = cache_key(contract, sample, video_sha, candidates, fps, len(video))
                    name = digest(sample.id)+".json"
                    cache_path = output/"scores"/name
                    if cache_path.exists():
                        score = read_score_cache(cache_path, key)
                        store.event({"event":"score_cache_hit", "question_id":sample.id})
                    else:
                        scorer.reset()
                        score_start, scores = time.perf_counter(), []
                        for offset in range(0, len(candidates), config.blip_batch_size):
                            if stop():
                                raise InterruptedError("Stop during BLIP scoring")
                            scores.extend(scorer(video, sample.question, candidates[offset:offset+config.blip_batch_size]))
                        backend.torch.cuda.synchronize()
                        score = score_record(key, scores, time.perf_counter()-score_start, scorer.calls)
                        atomic_json(cache_path, score)
                    def answer(indices, times):
                        backend.reset_peak()
                        begin = time.perf_counter()
                        check_space(ram, 0)
                        frames = [Image.fromarray(video[i].asnumpy()).convert("RGB") for i in indices]
                        backend.video_duration = len(video)/fps
                        decision = backend.answer(sample.public_question(), frames, times)
                        measurements = backend.measurements()
                        measurements["execution"] = execution
                        return decision, time.perf_counter()-begin, measurements
                    evaluate_question(sample, candidates, fps, weights.records[sample.id], score, author,
                                      store, output/"curves"/name, answer, stop, reference=prior)
                    completed_questions += 1
                    store.event({"event":"question_done", "question_id":sample.id})
            except InterruptedError:
                store.event({"event":"controlled_stop", "signals":stopped})
            finally:
                video = None
                gc.collect()
                if reader is not None:
                    reader.clear_cache()
                video_path.unlink(missing_ok=True)
                try:
                    store.event({"event":"session_end", "wall_seconds":time.monotonic()-started, "signals":stopped})
                    summary = export(output, contract, store)
                finally:
                    store.close()
            return 0 if summary["complete"] else 75
    finally:
        for number, handler in old_handlers.items():
            signal.signal(number, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ram-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--max-new-questions", type=int, default=0)
    parser.add_argument("--max-seconds", type=int, default=165600)
    args = parser.parse_args()
    if args.max_new_questions < 0 or args.max_seconds <= 0:
        parser.error("Invalid session limit")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
