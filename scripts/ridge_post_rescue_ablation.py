"""Bounded post-selection rescue. Separate BF16 run; immutable v1 caches stay read-only."""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import json
import math
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import ridge_component_ablation as frozen
from evidencelab.data import digest,file_hash
from evidencelab.store import Store,atomic_json,run_lock
from evidencelab.ridge_adapter import RIDGE_FILES,validate_weights

VARIANTS=('sigma2','post_rescue1','post_rescue2')
BASE_SOURCE=frozen.BASE_SOURCE
COMPONENT_SHA='ee89d02c2cf37c8c130db877665792ab52302e41f80b1301140769e1399eb083'
PROTOTYPE={'schema':'bounded-post-rescue-v1','base':'sigma2','peak_detector':'frozen-rescue-only-v1',
           'max_swaps':[1,2],'temporal_bins':16,'donor_radius_candidates':12,
           'redundant_neighbor_gap_candidates':2,'minimum_gap_candidates':2,
           'preserve_original_rescue_core_coverage':True,'preserve_first_last':True,
           'pilot_questions':20,'pilot_seed':20261022}
load=frozen.load
read_score_cache=frozen.read_score_cache
cache_key=frozen.cache_key

def validate_source(source,baseline):
    if file_hash(Path(frozen.__file__))!=COMPONENT_SHA:
        raise ValueError('Frozen component helper changed')
    return frozen.validate_source(source,baseline)

def pilot_manifest(baseline_rows,manifest):
    """Only ID, video and candidate count enter selection; no outcomes or 7-case list."""
    allowed=set(manifest['ids'])
    return frozen.choose_development([r for r in baseline_rows if r['id'] in allowed],
        count=PROTOTYPE['pilot_questions'],seed=PROTOTYPE['pilot_seed'])

def select_variant(author,raw_scores,weights,variant,k=64):
    import numpy as np
    if variant not in VARIANTS or k!=64:raise ValueError('Frozen post-rescue variants require 64 frames')
    scores=np.asarray(raw_scores,dtype=np.float64)
    weights=validate_weights(weights)
    base,base_details=frozen.select_variant(author,scores,weights,'sigma2',k)
    if variant=='sigma2':
        return base,{**base_details,'swaps':[],'swap_count':0,'prototype':PROTOTYPE}
    _,detector=frozen.select_variant(author,scores,weights,'rescue_only',k)
    n=len(scores);span=float(scores.max()-scores.min())
    norm=(scores-scores.min())/span if span>1e-9 else scores.copy()
    fine=detector['scale_analyses'][0]['curve']['s_smooth']
    original=set(base);selected=set(base);swaps=[];proposals=[]
    bin_of=lambda i:min(PROTOTYPE['temporal_bins']-1,i*PROTOTYPE['temporal_bins']//n)
    cores={p:set(range(max(0,p-1),min(n,p+2))) for p in detector['rescued_peaks']}
    protected=[core for core in cores.values() if core&original]
    cap=int(variant[-1])
    peak_candidates=[]
    for p,core in cores.items():
        if core&original:
            proposals.append({'peak':p,'status':'already_covered'});continue
        flank=np.r_[norm[max(0,p-12):max(0,p-3)],norm[min(n,p+4):min(n,p+13)]]
        contrast=float(max(norm[i] for i in core)-np.median(flank))
        peak_candidates.append((p,contrast))
    peak_candidates.sort(key=lambda item:(-item[1],-fine[item[0]],item[0]))
    for p,contrast in peak_candidates:
        core=cores[p]
        if len(swaps)>=cap:
            proposals.append({'peak':p,'status':'swap_cap'});continue
        if core&selected:
            proposals.append({'peak':p,'status':'covered_after_swap'});continue
        donors=[]
        for donor in sorted(original&selected):
            if donor in (base[0],base[-1]) or bin_of(donor)!=bin_of(p) or abs(donor-p)>PROTOTYPE['donor_radius_candidates']:
                continue
            retained=selected-{donor}
            neighbors=[i for i in retained if bin_of(i)==bin_of(donor) and abs(i-donor)<=PROTOTYPE['redundant_neighbor_gap_candidates']]
            if not neighbors or any(abs(p-i)<PROTOTYPE['minimum_gap_candidates'] for i in retained):continue
            proposed=retained|{p}
            if any(not core_before&proposed for core_before in protected):continue
            donors.append(donor)
        if not donors:
            proposals.append({'peak':p,'status':'no_redundant_local_donor'});continue
        donor=min(donors,key=lambda i:(norm[i],abs(i-p),i))
        selected.remove(donor);selected.add(p)
        swaps.append({'added':p,'removed':donor,'bin':bin_of(p),'contrast':contrast,
                      'distance_candidates':abs(p-donor)})
        proposals.append({'peak':p,'status':'swapped','donor':donor})
    positions=sorted(selected)
    if len(positions)!=64 or len(original-selected)!=len(swaps) or len(selected-original)!=len(swaps):
        raise AssertionError('Swap budget/order invariant')
    if Counter(bin_of(i) for i in positions)!=Counter(bin_of(i) for i in base):raise AssertionError('Temporal bins changed')
    if not {base[0],base[-1]}<=selected or any(not c&selected for c in protected):raise AssertionError('Original coverage lost')
    return positions,{**base_details,'prototype':PROTOTYPE,'swaps':swaps,'swap_count':len(swaps),
        'eligible_rescue_peaks':detector['rescued_peaks'],'proposals':proposals,
        'selected_core_coverage_preserved':True,'temporal_bin_counts_preserved':True}

def evaluate_question(sample, candidates, fps, weights, score, author, store,
                      diagnostic_path, answer, stop=lambda: False, reference=None):
    """One read-only cache, three serial variants, independently committed answers."""
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
               "question":sample.question,"choices":list(sample.choices),
               "prediction": decision.prediction, "answer": sample.answer,
               "correct": decision.prediction == sample.answer, "video": sample.video,
               "question_type": sample.question_type, "duration_group": sample.duration_group,
               "frame_indices": indices, "timestamps": times, "final_frames": len(indices),
               "scores": decision.scores, "input_tokens": decision.input_tokens,
               "swap_count": details["swap_count"], "swaps": details["swaps"],
               "selector_only_seconds": select_seconds, "answer_decode_seconds": answer_seconds,
               "cached_question_seconds": select_seconds + answer_seconds,
               "score_sha256": score["sha256"], "weight_sha256": weights["record_sha256"],
               "prior_qwen_generation_seconds": weights["generation_seconds"], **measurements}
        if reference is not None:
            row.update(baseline_frame_match=indices == reference["calls"][0]["frame_indices"],
                       baseline_prediction_match=decision.prediction == reference["prediction"])
            if variant=='sigma2' and not (row['baseline_frame_match'] and row['baseline_prediction_match']):
                raise ValueError('Sigma2 control mismatch; stop before interpreting new selectors')
            if row['baseline_frame_match'] and not row['baseline_prediction_match']:
                raise ValueError('Identical input frames did not reproduce baseline prediction')
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
    scores = []  # Scores belong to the immutable source run, never recomputed here.
    summary = {"complete": len(rows) == contract["selected_count"], "selected_questions": len(contract["manifest"]["ids"]),
               "completed_variant_answers": len(rows), "variants": stats,
               "one_time_scored_questions": len(scores),
               "new_blip_calls":0,"new_qwen_calls":0,
               "one_time_scoring_seconds": sum(v["original_scoring_seconds"] for v in scores),
               "timing_scope": "Cached timings exclude prior v1 BLIP/Qwen, setup/indexing/transfer and diagnostic/result I/O. No new BLIP or Qwen inference.",
               "interpretation": "Exploratory development ablation, not held-out benchmark evidence."}
    controls = [v for v in rows if v.get("baseline_frame_match")]
    summary["reproduction_on_identical_sigma2_frames"] = {"n": len(controls),
        "same_frames": sum(v["baseline_frame_match"] for v in controls),
        "same_prediction": sum(v["baseline_prediction_match"] for v in controls)}
    events = [json.loads(value) for (value,) in store.db.execute("SELECT value FROM events")]
    summary["recorded_sessions"] = {
        "model_setup_seconds": sum(v["seconds"] for v in events if v["event"] == "models_ready"),
        "video_transfer_hash_seconds": sum(v["transfer_hash_seconds"] for v in events if v["event"] == "video_loaded"),
        "worker_wall_seconds": sum(v["wall_seconds"] for v in events if v["event"] == "session_end")}
    summary["available_score_questions"] = len(contract["manifest"]["ids"])
    summary["reused_score_questions"] = len({row["question_id"] for row in rows})
    summary["read_only_score_source"] = contract["score_source"]
    summary["read_only_reference_full_dev200"] = contract["reference_variants"]
    completed = {q for q in contract["manifest"]["ids"] if all(q+"::"+v in store.done() for v in VARIANTS)}
    summary["pilot20"] = {"target_questions":20, "completed_questions":len(set(contract["pilot20_manifest"]["ids"]) & completed), "complete":set(contract["pilot20_manifest"]["ids"]) <= completed}
    reference_rows = [json.loads(line) for line in (Path(contract["score_source"])/"predictions.jsonl").read_text().splitlines() if line.strip()]
    summary["matched_reference_variants"] = {v:{"n":len(completed), "correct":sum(r["correct"] for r in reference_rows if r["variant"]==v and r["question_id"] in completed)} for v in ("sigma2","sigma1","sigma4")}
    summary["swap_counts"] = {v:dict(Counter(r["swap_count"] for r in rows if r["variant"]==v)) for v in VARIANTS}
    atomic_json(output/"summary.json", summary)
    with (output/"predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    return summary



def run(args):
    from evidencelab.aks_adapter import aks_candidates
    from evidencelab.backends import HFBackend
    from evidencelab.config import source_fingerprint
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
    cached = validate_source(args.source_cache, baseline)
    source_cache = args.source_cache.resolve()
    if output == source_cache or source_cache in output.parents or output in source_cache.parents:
        raise ValueError("Component output must be separate from read-only score source")
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
    manifest = cached["contract"]["manifest"]
    pilot = pilot_manifest(baseline_rows, manifest)
    selected = sorted((q for q in samples if q.id in set(manifest["ids"])), key=lambda q:(q.video,q.id))
    if len(selected) != 200:
        raise ValueError("Development annotation coverage changed")
    contract = {"schema": "ridge-post-rescue-ablation-v1", "source_sha256": BASE_SOURCE,
                "experiment_script_sha256": file_hash(Path(__file__)), "config": config.contract(),
                "runtime": runtime, "resolved_dtype": execution["resolved_dtype"],
                "baseline_contract_sha256": digest(old), "baseline_weights_sha256": file_hash(baseline/"ridge-weights.json"),
                "baseline_predictions_sha256": file_hash(baseline/"predictions.jsonl"),
                "manifest": manifest, "pilot20_manifest":pilot, "component_helper_sha256":COMPONENT_SHA, "variants": list(VARIANTS), "prototype": PROTOTYPE,
                "selected_count": 200 * len(VARIANTS), "author_source": RIDGE_FILES,
                "score_source": str(source_cache), "score_source_contract_sha256": file_hash(source_cache/"contract.json"),
                "score_source_predictions_sha256": file_hash(source_cache/"predictions.jsonl"),
                "score_inventory": cached["inventory"], "reference_variants": cached["summary"]["variants"]}
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
                (output/"curves").mkdir(exist_ok=True)
                atomic_json(output/"manifest.json", manifest)
                done = store.done()
                scope = set(pilot["ids"]) if args.mode == "pilot20" else set(manifest["ids"])
                selected = [q for q in selected if q.id in scope]
                pending = [q for q in selected if any(q.id+"::"+v not in done for v in VARIANTS)]
                if not pending:
                    mode_summary=export(output,contract,store)
                    return 0 if mode_summary['complete'] or (args.mode=='pilot20' and mode_summary['pilot20']['complete']) else 75
                if stop():
                    raise InterruptedError("Stop before setup")
                source = HTTPRangeSource(stop=stop)
                reader = MultipartReader(source)
                members = archive_index(reader, args.index, {q.video for q in selected})
                reader.clear_cache()
                reader.block_size = TRANSFER_CHUNK_BYTES
                setup_start = time.perf_counter()
                backend = HFBackend(config)
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
                    key = cache_key(cached["contract"], sample, video_sha, candidates, fps, len(video))
                    name = digest(sample.id)+".json"
                    score = read_score_cache(source_cache/"scores"/name, key)
                    if score["sha256"] != cached["inventory"][sample.id]:
                        raise ValueError("Read-only score cache changed during this job")
                    store.event({"event":"source_score_cache_hit", "question_id":sample.id})
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
                    print(json.dumps(summary,ensure_ascii=False),flush=True)
                finally:
                    store.close()
            return 0 if summary['complete'] or (args.mode=='pilot20' and summary['pilot20']['complete']) else 75
    finally:
        for number, handler in old_handlers.items():
            signal.signal(number, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    if "--preflight" in sys.argv:
        parser.add_argument("--preflight", action="store_true")
        parser.add_argument("--baseline", type=Path, required=True)
        parser.add_argument("--source-cache", type=Path, required=True)
        args = parser.parse_args()
        data = validate_source(args.source_cache, args.baseline)
        print(json.dumps({"preflight":"passed", "cached_questions":len(data["inventory"]),
                          "variants":list(VARIANTS), "new_answers":600, "new_blip_calls":0}))
        return 0
    parser.add_argument("--mode", choices=["pilot20","dev200"], default="pilot20")
    parser.add_argument("--source-cache", type=Path, required=True)
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
