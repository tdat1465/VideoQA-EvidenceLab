"""Separate event decomposition and bounded coverage ablation. New run; all frozen baseline files stay read-only."""
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



import ridge_query_options_ablation as query_frozen
import ridge_event_coverage as events
from evidencelab.pipeline import Decision

VARIANTS = events.VARIANTS
PROTOTYPE = events.PROTOTYPE
BASE_SOURCE = frozen.BASE_SOURCE
COMPONENT_SHA = query_frozen.COMPONENT_SHA
QUERY_HELPER_SHA = 'fccf37f6daebf4198597d51bda1d1470f3a29010b8722b74fe248ef3f0f04245'
load = frozen.load
read_score_cache = frozen.read_score_cache
cache_key = frozen.cache_key


def helper_hash():
    return file_hash(Path(events.__file__))


def validate_source(source, baseline):
    if file_hash(Path(query_frozen.__file__)) != QUERY_HELPER_SHA:
        raise ValueError('Frozen BLIP query helper changed')
    return query_frozen.validate_source(source, baseline)


def pilot_manifest(baseline_rows, manifest):
    return query_frozen.pilot_manifest(baseline_rows, manifest)


def event_score(key, question_scores, per_event, plan, seconds, batches):
    events.validate_vectors(question_scores, per_event, plan)
    if not math.isfinite(seconds) or seconds < 0 or type(batches) is not int or batches < 0:
        raise ValueError('Invalid event scoring counters')
    if bool(plan['queries']) != bool(batches):
        raise ValueError('Scorer call count inconsistent with queries')
    record = {'key': key, 'question_scores': question_scores, 'per_event_scores': per_event,
              'query_plan': plan, 'original_scoring_seconds': seconds, 'scorer_batches': batches}
    record['sha256'] = digest(record)
    return record


def read_event_score(path, key, question_scores, plan):
    record = load(path)
    if (record['key'] != key or record['question_scores'] != question_scores or record['query_plan'] != plan
            or record['sha256'] != digest({k: v for k, v in record.items() if k != 'sha256'})):
        raise ValueError('Event cache/query/source checksum changed')
    checked = event_score(key, question_scores, record['per_event_scores'], plan,
                          record['original_scoring_seconds'], record['scorer_batches'])
    if checked != record:
        raise ValueError('Invalid event score record')
    return record


select_variant = events.select_variant


def evaluate_question(sample, candidates, fps, weights, score, author, store, diagnostic_path,
                      answer, stop=lambda: False, reference=None, reference_sigma1=None):
    diagnostics = load(diagnostic_path) if diagnostic_path.exists() else {'variants': {}}
    if diagnostics.get('score_sha256', score['sha256']) != score['sha256']:
        raise ValueError('Curve score changed')
    diagnostics.update(question_id=sample.id, score_sha256=score['sha256'])
    # Same public question + identical ordered frames/timestamps is the same
    # deterministic forward. Record reuse explicitly; never borrow old answers.
    same_session = {tuple(r['frame_indices']): r for r in store.results() if r['question_id'] == sample.id}
    for variant in VARIANTS:
        key = sample.id+'::'+variant
        if key in store.done():
            continue
        if stop():
            raise InterruptedError('Stop between event variants')
        start = time.perf_counter()
        positions, details = select_variant(author, score, weights['weights'], variant)
        seconds = time.perf_counter()-start
        indices = [candidates[i] for i in positions]
        times = [i/fps for i in indices]
        diagnostics['variants'][variant] = {**details, 'selected_positions': positions,
                                            'frame_indices': indices, 'timestamps': times}
        atomic_json(diagnostic_path, diagnostics)
        prior = same_session.get(tuple(indices))
        if prior is None:
            decision, answer_seconds, measurements = answer(indices, times)
        else:
            decision = Decision(prior['scores'], prior['input_tokens'])
            answer_seconds, measurements = 0., {'execution': prior['execution']}
        if (len(decision.scores) != len(sample.choices)
                or any(not math.isfinite(x) or not 0 <= x <= 1 for x in decision.scores)
                or abs(sum(decision.scores)-1) > 1e-4):
            raise ValueError('Invalid answer scores')
        fm = indices == reference['calls'][0]['frame_indices']
        pm = decision.prediction == reference['prediction']
        f1 = indices == reference_sigma1['frame_indices']
        p1 = decision.prediction == reference_sigma1['prediction']
        if ((variant == 'sigma2' and not fm) or (fm and not pm)
                or (variant == 'sigma1' and not f1) or (f1 and not p1)):
            raise ValueError('Frozen sigma1/sigma2 control mismatch; investigate before continuing')
        row = {'id': key, 'question_id': sample.id, 'variant': variant, 'question': sample.question,
               'choices': list(sample.choices), 'answer': sample.answer, 'prediction': decision.prediction,
               'correct': sample.answer == decision.prediction, 'video': sample.video,
               'question_type': sample.question_type, 'duration_group': sample.duration_group,
               'frame_indices': indices, 'timestamps': times, 'final_frames': len(indices),
               'scores': decision.scores, 'input_tokens': decision.input_tokens,
               'selector_only_seconds': seconds, 'answer_decode_seconds': answer_seconds,
               'cached_question_seconds': seconds+answer_seconds, 'score_sha256': score['sha256'],
               'weight_sha256': weights['record_sha256'], 'baseline_frame_match': fm,
               'baseline_prediction_match': pm, 'sigma1_frame_match': f1, 'sigma1_prediction_match': p1,
               'parser_eligible': score['query_plan']['eligible'],
               'accepted_events': details.get('accepted_events'),
               'answer_inference_performed': prior is None,
               'reused_answer_from': None if prior is None else prior['id'], **measurements}
        store.put(row)
        same_session[tuple(indices)] = row
        print(f'Committed {sample.id}: {variant}, correct={row["correct"]}, reused={prior is not None}', flush=True)


def export(output, contract, store):
    rows = store.results()
    groups = {}
    for row in rows:
        groups.setdefault(row['question_id'], {})[row['variant']] = row
    stats = {}
    for variant in VARIANTS:
        rr = [r for r in rows if r['variant'] == variant]
        stats[variant] = {'n': len(rr), 'correct': sum(r['correct'] for r in rr),
                         'accuracy': sum(r['correct'] for r in rr)/len(rr) if rr else None}
    completed = {q for q, g in groups.items() if set(g) == set(VARIANTS)}
    pilot = set(contract['pilot20_manifest']['ids'])
    scores = [load(p) for p in sorted((output/'scores').glob('*.json'))]
    recorded = [json.loads(x) for x, in store.db.execute('SELECT value FROM events')]
    paired = {}
    for control in ('sigma2', 'sigma1', 'time16_sigma1', 'eventmean_sigma1'):
        paired[control] = {'n': len(completed),
            'gains': sum(groups[q]['eventcover_sigma1']['correct'] and not groups[q][control]['correct'] for q in completed),
            'losses': sum(groups[q][control]['correct'] and not groups[q]['eventcover_sigma1']['correct'] for q in completed)}
    summary = {'complete': len(rows) == contract['selected_count'], 'selected_questions': 200,
        'completed_variant_answers': len(rows), 'variants': stats,
        'pilot20': {'target_questions': 20, 'completed_questions': len(completed & pilot), 'complete': pilot <= completed},
        'new_blip_batches': sum(x['scorer_batches'] for x in scores), 'new_qwen_calls': 0,
        'event_cache_questions': len(scores), 'parser_eligible_questions': sum(x['query_plan']['eligible'] for x in scores),
        'supported_event_questions': sum(r['variant'] == 'eventcover_sigma1' and bool(r['accepted_events']) for r in rows),
        'eventcover_same_frames_as_sigma1': sum(r['variant'] == 'eventcover_sigma1' and r['sigma1_frame_match'] for r in rows),
        'new_event_queries': sum(len(x['query_plan']['queries']) for x in scores),
        'new_event_chunks': sum(len(q['content_ids']) for x in scores for q in x['query_plan']['queries']),
        'one_time_scoring_seconds': sum(x['original_scoring_seconds'] for x in scores),
        'actual_answer_forwards': sum(r['answer_inference_performed'] for r in rows),
        'same_run_answer_reuses': sum(not r['answer_inference_performed'] for r in rows),
        'paired_eventcover_vs': paired,
        'control_reproduction': {v: {'n': stats[v]['n'],
            'same_frames': sum(r['variant'] == v and r[field+'_frame_match'] for r in rows),
            'same_prediction': sum(r['variant'] == v and r[field+'_prediction_match'] for r in rows)}
            for v, field in [('sigma2', 'baseline'), ('sigma1', 'sigma1')]},
        'recorded_sessions': {name: sum(e[field] for e in recorded if e['event'] == kind)
            for name, kind, field in [('worker_wall_seconds', 'session_end', 'wall_seconds'),
                ('model_setup_seconds', 'models_ready', 'seconds'), ('video_transfer_hash_seconds', 'video_loaded', 'transfer_hash_seconds')]},
        'timing_scope': 'New event BLIP decode/preprocess/forward separate; reused identical-frame answers have zero forward seconds.',
        'interpretation': 'Development prototype, not validated improvement. Gate uses public text; no dataset type or answer labels.'}
    atomic_json(output/'summary.json', summary)
    with (output/'predictions.jsonl').open('w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    return summary


def run(args):
    from evidencelab.aks_adapter import aks_candidates
    from evidencelab.backends import HFBackend
    from evidencelab.focus_adapter import BlipITMScorer
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
    contract = {"schema": "ridge-event-coverage-ablation-v1", "source_sha256": BASE_SOURCE,
                "experiment_script_sha256": file_hash(Path(__file__)), "config": config.contract(),
                "runtime": runtime, "resolved_dtype": execution["resolved_dtype"],
                "baseline_contract_sha256": digest(old), "baseline_weights_sha256": file_hash(baseline/"ridge-weights.json"),
                "baseline_predictions_sha256": file_hash(baseline/"predictions.jsonl"),
                "manifest": manifest, "pilot20_manifest":pilot, "component_helper_sha256":COMPONENT_SHA, "event_helper_sha256":helper_hash(), "query_helper_sha256":QUERY_HELPER_SHA, "variants": list(VARIANTS), "prototype": PROTOTYPE,
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
    sigma1_by_id = {r["question_id"]: r for r in (json.loads(x) for x in (source_cache/"predictions.jsonl").read_text().splitlines()) if r["variant"] == "sigma1"}
    try:
        with run_lock(output):
            store = Store(output, contract, (output/"results.sqlite3").exists())
            try:
                (output/"curves").mkdir(exist_ok=True)
                (output/"scores").mkdir(exist_ok=True)
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
                if any(events.event_descriptions(q.public_question())["eligible"] for q in pending):
                    scorer = BlipITMScorer(config)
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
                    plan=events.event_plan(sample.public_question(),scorer.processor.tokenizer if scorer is not None else None)
                    new_key={**cache_key(contract,sample,video_sha,candidates,fps,len(video)),
                             'public_query_sha256':digest({'question':sample.question,'choices':sorted(sample.choices)}),
                             'query_plan_sha256':digest(plan),'source_question_score_sha256':score['sha256']}
                    query_path=output/'scores'/name
                    if query_path.exists():
                        score=read_event_score(query_path,new_key,score['scores'],plan)
                    else:
                        begin=time.perf_counter()
                        event_scores=query_frozen.score_options(scorer,video,candidates,plan["queries"],config.blip_batch_size,stop) if plan["queries"] else []
                        backend.torch.cuda.synchronize()
                        score=event_score(new_key,score['scores'],event_scores,plan,time.perf_counter()-begin,scorer.calls if plan["queries"] else 0)
                        atomic_json(query_path,score)
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
                                      store, output/"curves"/name, answer, stop, reference=prior, reference_sigma1=sigma1_by_id[sample.id])
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
                          "variants":list(VARIANTS), "new_answers":1000, "new_blip_scoring":"deduplicated events for text-eligible sequences only", "new_qwen_calls":0}))
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
