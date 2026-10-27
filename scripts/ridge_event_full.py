"""Frozen event-coverage v1 on all 1337 validation questions, in a separate run."""
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
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import ridge_event_coverage_ablation as dev
import ridge_event_coverage as events
from evidencelab.config import source_fingerprint
from evidencelab.data import digest, file_hash
from evidencelab.pipeline import Decision
from evidencelab.store import Store, atomic_json, run_lock

VARIANTS = events.VARIANTS
SCHEMA = 'ridge-event-coverage-full1337-v1'
load = dev.load


def read_rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def validate_inputs(source, baseline, development):
    """Standard-library preflight; no model/library downloads on the login node."""
    cached = dev.validate_source(source, baseline)
    base = load(baseline / 'contract.json')
    bases = read_rows(baseline / 'predictions.jsonl')
    summary = load(baseline / 'summary.json')
    dc, ds = load(development / 'contract.json'), load(development / 'summary.json')
    if (len(bases) != 1337 or len({r['id'] for r in bases}) != 1337 or not summary['complete']
            or digest(sorted(r['id'] for r in bases)) != base['selected_ids_hash']
            or not ds['complete'] or ds['completed_variant_answers'] != 1000
            or not ds['pilot20']['complete'] or dc['schema'] != 'ridge-event-coverage-ablation-v1'
            or dc['prototype'] != events.PROTOTYPE or dc['variants'] != list(VARIANTS)
            or dc['manifest'] != cached['contract']['manifest'] or dc['score_inventory'] != cached['inventory']):
        raise ValueError('Require completed full baseline and frozen event dev200')
    for name in ('source_sha256', 'config', 'runtime', 'resolved_dtype'):
        if dc[name] != base[name]:
            raise ValueError('Development/baseline runtime or configuration changed')
    for field, path in [
        ('experiment_script_sha256', Path(dev.__file__)),
        ('event_helper_sha256', Path(events.__file__)),
        ('query_helper_sha256', Path(dev.query_frozen.__file__)),
        ('component_helper_sha256', Path(dev.frozen.__file__)),
        ('baseline_predictions_sha256', baseline / 'predictions.jsonl'),
        ('baseline_weights_sha256', baseline / 'ridge-weights.json'),
        ('score_source_contract_sha256', source / 'contract.json'),
        ('score_source_predictions_sha256', source / 'predictions.jsonl'),
    ]:
        if file_hash(path) != dc[field]:
            raise ValueError('Frozen input changed: ' + field)
    if dc['baseline_contract_sha256'] != digest(base):
        raise ValueError('Baseline contract changed')
    refs = {(r['question_id'], r['variant']): r for r in read_rows(development / 'predictions.jsonl')}
    source_refs = {(r['question_id'], r['variant']): r for r in read_rows(source / 'predictions.jsonl')}
    base_by_id = {r['id']: r for r in bases}
    expected = {(q, v) for q in dc['manifest']['ids'] for v in VARIANTS}
    if len(refs) != 1000 or set(refs) != expected:
        raise ValueError('Incomplete development predictions')
    event_inventory = {}
    for q in dc['manifest']['ids']:
        name = digest(q) + '.json'
        oldscore = load(source / 'scores' / name)
        score = load(development / 'scores' / name)
        if (score['key']['experiment'] != digest(dc) or score['key']['question_id'] != q
                or score['key']['source_question_score_sha256'] != oldscore['sha256']):
            raise ValueError('Development event cache provenance changed')
        dev.read_event_score(development / 'scores' / name, score['key'], oldscore['scores'], score['query_plan'])
        event_inventory[q] = score['sha256']
        for v in VARIANTS:
            r = refs[q, v]
            b = base_by_id[q]
            if (r['id'] != q + '::' + v or r['score_sha256'] != score['sha256']
                    or r['weight_sha256'] != b['ridge_weight_record_sha256']
                    or r['answer'] != b['answer'] or r['video'] != b['video']
                    or digest({'id': q, 'question': r['question']}) != oldscore['key']['question_sha256']
                    or r['correct'] != (r['prediction'] == r['answer'])):
                raise ValueError('Development result/cache mismatch')
            if v in ('sigma1', 'sigma2') and any(r[k] != source_refs[q, v][k] for k in ('frame_indices', 'prediction')):
                raise ValueError('Development controls no longer reproduce')
    manifest = {'schema': 1, 'count': 1337, 'ids': sorted(r['id'] for r in bases),
                'videos': sorted({r['video'] for r in bases}),
                'sampling': 'all-pinned-longvideobench-validation-questions',
                'development_ids': dc['manifest']['ids'], 'development_videos': dc['manifest']['videos'],
                'evaluation_note': 'Report all validation and video-disjoint-from-dev200 subsets separately.'}
    inputs = {'baseline_contract_sha256': digest(base),
              'baseline_predictions_sha256': file_hash(baseline / 'predictions.jsonl'),
              'baseline_weights_sha256': file_hash(baseline / 'ridge-weights.json'),
              'score_source_contract_sha256': file_hash(source / 'contract.json'),
              'score_source_predictions_sha256': file_hash(source / 'predictions.jsonl'),
              'development_contract_sha256': file_hash(development / 'contract.json'),
              'development_predictions_sha256': file_hash(development / 'predictions.jsonl'),
              'question_score_inventory': cached['inventory'], 'development_event_inventory': event_inventory}
    return {'cached': cached, 'base': base, 'bases': bases, 'development_contract': dc,
            'references': refs, 'manifest': manifest, 'inputs': inputs}


def code_contract():
    return {'worker_sha256': file_hash(Path(__file__)), 'development_worker_sha256': file_hash(Path(dev.__file__)),
            'event_helper_sha256': file_hash(Path(events.__file__)),
            'query_helper_sha256': file_hash(Path(dev.query_frozen.__file__)),
            'component_helper_sha256': file_hash(Path(dev.frozen.__file__))}


def validate_score(record, key):
    if record['key'] != key or record['sha256'] != digest({k: v for k, v in record.items() if k != 'sha256'}):
        raise ValueError('Full-run score key/checksum changed')
    n = len(key['candidate_indices'])
    if not n or key['candidate_indices'] != sorted(set(key['candidate_indices'])) or not math.isfinite(key['fps']) or key['fps'] <= 0:
        raise ValueError('Invalid cached candidate grid or FPS')
    if n <= 64:
        if (record['question_scores'] is not None or record['per_event_scores']
                or record['question_origin'] != 'all-candidates' or record['query_plan'] is not None):
            raise ValueError('Short video must use every candidate without fabricated scores')
    else:
        events.validate_vectors(record['question_scores'], record['per_event_scores'], record['query_plan'])
        if len(record['question_scores']) != n:
            raise ValueError('Question score/grid length mismatch')
    for field in ('question_scoring_seconds', 'event_scoring_seconds', 'question_batches', 'event_batches'):
        if not math.isfinite(record[field]) or record[field] < 0:
            raise ValueError('Invalid scoring counter')
    return record


def select_full(author, score, weights, variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown variant')
    n = len(score['key']['candidate_indices'])
    if n <= 64:
        return list(range(n)), {'fallback': True, 'selection_mode': 'all-candidates', 'accepted_events': 0}
    positions, details = events.select_variant(author, score, weights, variant)
    return positions, dev.frozen.jsonable(details)


def evaluate(sample, score, weights, author, store, curve_path, answer, baseline, references, stop=lambda: False):
    candidates, fps = score['key']['candidate_indices'], score['key']['fps']
    diagnostics = load(curve_path) if curve_path.exists() else {'question_id': sample.id, 'score_sha256': score['sha256'], 'variants': {}}
    if diagnostics['score_sha256'] != score['sha256']:
        raise ValueError('Curve/score mismatch')
    previous = [r for r in store.results() if r['question_id'] == sample.id]
    same_frames = {tuple(r['frame_indices']): r for r in previous}
    done = {r['variant'] for r in previous}
    sigma1_positions, _ = select_full(author, score, weights['weights'], 'sigma1')
    sigma1_frames = [candidates[i] for i in sigma1_positions]
    for variant in VARIANTS:
        if variant in done:
            continue
        if stop():
            raise InterruptedError('Stop between full variants')
        start = time.perf_counter()
        positions, details = select_full(author, score, weights['weights'], variant)
        indices = [candidates[i] for i in positions]
        times = [i / fps for i in indices]
        selector_seconds = time.perf_counter() - start
        if (len(indices) != min(64, len(candidates)) or indices != sorted(set(indices))):
            raise ValueError('Frame budget/order mismatch')
        fm = indices == baseline['calls'][0]['frame_indices']
        reference = references.get((sample.id, variant))
        if variant == 'sigma2' and not fm:
            raise ValueError('Recomputed sigma2 frames differ from full baseline; investigate, do not tune this run')
        if reference is not None and indices != reference['frame_indices']:
            raise ValueError('Frozen development selection did not reproduce')
        diagnostics['variants'][variant] = {**details, 'selected_positions': positions, 'frame_indices': indices, 'timestamps': times}
        atomic_json(curve_path, diagnostics)
        prior = same_frames.get(tuple(indices))
        if prior is None:
            decision, answer_seconds, measurements = answer(indices, times)
        else:
            decision, answer_seconds = Decision(prior['scores'], prior['input_tokens']), 0.
            measurements = {'execution': prior['execution']}
        if (len(decision.scores) != len(sample.choices)
                or any(not math.isfinite(x) or not 0 <= x <= 1 for x in decision.scores)
                or abs(sum(decision.scores) - 1) > 1e-4):
            raise ValueError('Invalid answer probabilities')
        if fm and decision.prediction != baseline['prediction']:
            raise ValueError('Same-frame full-baseline prediction mismatch')
        if reference is not None and decision.prediction != reference['prediction']:
            raise ValueError('Same-frame development prediction mismatch')
        row = {'id': sample.id + '::' + variant, 'question_id': sample.id, 'variant': variant,
               'question': sample.question, 'choices': list(sample.choices), 'answer': sample.answer,
               'prediction': decision.prediction, 'correct': sample.answer == decision.prediction,
               'video': sample.video, 'question_type': sample.question_type, 'duration_group': sample.duration_group,
               'frame_indices': indices, 'timestamps': times, 'final_frames': len(indices),
               'scores': decision.scores, 'input_tokens': decision.input_tokens,
               'selector_only_seconds': selector_seconds, 'answer_decode_seconds': answer_seconds,
               'score_sha256': score['sha256'], 'weight_sha256': weights['record_sha256'],
               'baseline_frame_match': fm, 'baseline_prediction_match': decision.prediction == baseline['prediction'],
               'sigma1_frame_match': indices == sigma1_frames,
               'development_reference_match': None if reference is None else True,
               'parser_eligible': events.event_descriptions(sample.public_question())['eligible'],
               'accepted_events': details.get('accepted_events', 0),
               'all_candidates': len(candidates) <= 64, 'answer_inference_performed': prior is None,
               'reused_answer_from': None if prior is None else prior['id'], **measurements}
        store.put(row)
        same_frames[tuple(indices)] = row
        print(f'Committed {sample.id}: {variant}, correct={row["correct"]}, reused={prior is not None}', flush=True)


def export(output, contract, store):
    rows = store.results()
    grouped = {}
    for r in rows:
        grouped.setdefault(r['question_id'], {})[r['variant']] = r
    complete = {q for q, g in grouped.items() if set(g) == set(VARIANTS)}
    stats = {v: {'n': sum(r['variant'] == v for r in rows),
                 'correct': sum(r['variant'] == v and r['correct'] for r in rows)} for v in VARIANTS}
    for v in stats:
        stats[v]['accuracy'] = stats[v]['correct'] / stats[v]['n'] if stats[v]['n'] else None
    score_records = [load(p) for p in sorted((output / 'scores').glob('*.json'))]
    events_log = [json.loads(x) for x, in store.db.execute('SELECT value FROM events')]
    dev_videos = set(contract['manifest']['development_videos'])
    subsets = {}
    for name, keep in [('development_videos', True), ('unseen_videos', False)]:
        ids = {q for q in complete if (grouped[q]['sigma1']['video'] in dev_videos) == keep}
        subsets[name] = {'completed_questions': len(ids), 'videos': len({grouped[q]['sigma1']['video'] for q in ids}),
                         'correct': {v: sum(grouped[q][v]['correct'] for q in ids) for v in VARIANTS}}
    summary = {'complete': len(rows) == contract['selected_count'], 'selected_questions': 1337,
               'target_variant_answers': contract['selected_count'], 'completed_variant_answers': len(rows),
               'completed_questions': len(complete), 'variants': stats, 'subsets': subsets,
               'question_score_origins': dict(Counter(x['question_origin'] for x in score_records)),
               'new_question_scoring_seconds': sum(x['question_scoring_seconds'] for x in score_records),
               'new_event_scoring_seconds': sum(x['event_scoring_seconds'] for x in score_records),
               'new_question_batches': sum(x['question_batches'] for x in score_records),
               'new_event_batches': sum(x['event_batches'] for x in score_records),
               'new_qwen_calls': 0, 'actual_answer_forwards': sum(r['answer_inference_performed'] for r in rows),
               'same_run_answer_reuses': sum(not r['answer_inference_performed'] for r in rows),
               'parser_eligible_questions': sum(r['variant'] == 'eventcover_sigma1' and r['parser_eligible'] for r in rows),
               'supported_event_questions': sum(r['variant'] == 'eventcover_sigma1' and bool(r['accepted_events']) for r in rows),
               'eventcover_same_frames_as_sigma1': sum(r['variant'] == 'eventcover_sigma1' and r['sigma1_frame_match'] for r in rows),
               'development_reproduction': {v: {'n': sum(r['variant'] == v and r['development_reference_match'] is not None for r in rows),
                    'matched': sum(r['variant'] == v and r['development_reference_match'] is True for r in rows)} for v in VARIANTS},
               'control_reproduction': {'sigma2': {'n': stats['sigma2']['n'],
                   'same_frames': sum(r['variant'] == 'sigma2' and r['baseline_frame_match'] for r in rows),
                   'same_prediction': sum(r['variant'] == 'sigma2' and r['baseline_prediction_match'] for r in rows)}},
               'paired_eventcover_vs': {v: {'n': len(complete),
                   'gains': sum(grouped[q]['eventcover_sigma1']['correct'] and not grouped[q][v]['correct'] for q in complete),
                   'losses': sum(grouped[q][v]['correct'] and not grouped[q]['eventcover_sigma1']['correct'] for q in complete)} for v in VARIANTS[:-1]},
               'recorded_sessions': {'worker_wall_seconds': sum(x['wall_seconds'] for x in events_log if x['event'] == 'session_end')},
               'interpretation': 'Report development videos separately from unseen videos. Cached scores may be reused; every answer originates in this new run.'}
    atomic_json(output / 'summary.json', summary)
    with (output / 'predictions.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
        for r in rows:
            stream.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n')
    return summary


def make_score(sample, contract, candidates, fps, source_frames, video_sha, cached, source, development, video, scorer, batch_size, stop):
    key = {**dev.cache_key(contract, sample, video_sha, candidates, fps, source_frames),
           'public_query_sha256': digest({'question': sample.question, 'choices': sorted(sample.choices)})}
    record = {'key': key, 'question_scores': None, 'per_event_scores': [], 'query_plan': None,
              'question_origin': 'all-candidates', 'event_origin': 'not-needed',
              'question_scoring_seconds': 0., 'event_scoring_seconds': 0., 'question_batches': 0, 'event_batches': 0}
    if len(candidates) > 64:
        name = digest(sample.id) + '.json'
        if sample.id in cached['inventory']:
            oldkey = dev.cache_key(cached['contract'], sample, video_sha, candidates, fps, source_frames)
            old = dev.read_score_cache(source / 'scores' / name, oldkey)
            if old['sha256'] != cached['inventory'][sample.id]:
                raise ValueError('Frozen question scores changed')
            record.update(question_scores=old['scores'], question_origin='development-cache')
        else:
            scorer.reset()
            started, raw = time.perf_counter(), []
            for offset in range(0, len(candidates), batch_size):
                if stop():
                    raise InterruptedError('Stop during question scoring')
                raw.extend(scorer(video, sample.question, candidates[offset:offset + batch_size]))
            record.update(question_scores=raw, question_origin='fresh-blip',
                          question_scoring_seconds=time.perf_counter() - started, question_batches=scorer.calls)
        plan = events.event_plan(sample.public_question(), scorer.processor.tokenizer)
        record['query_plan'] = plan
        if sample.id in contract['development_event_inventory']:
            old = load(development / 'scores' / name)
            dev.read_event_score(development / 'scores' / name, old['key'], record['question_scores'], plan)
            if (old['sha256'] != contract['development_event_inventory'][sample.id]
                    or old['query_plan'] != plan or old['question_scores'] != record['question_scores']):
                raise ValueError('Frozen development event scores changed')
            record.update(per_event_scores=old['per_event_scores'], event_origin='development-cache')
        elif plan['queries']:
            started = time.perf_counter()
            per_event = dev.query_frozen.score_options(scorer, video, candidates, plan['queries'], batch_size, stop)
            record.update(per_event_scores=per_event, event_origin='fresh-blip',
                          event_scoring_seconds=time.perf_counter() - started, event_batches=scorer.calls)
    record['sha256'] = digest(record)
    return validate_score(record, key)


def run(args):
    from evidencelab.aks_adapter import aks_candidates
    from evidencelab.backends import HFBackend
    from evidencelab.focus_adapter import BlipITMScorer
    from evidencelab.hardware import execution_profile
    from evidencelab.longvideo import longvideo_runtime, validate_ram_root
    from evidencelab.longvideo_config import LongVideoConfig
    from evidencelab.longvideo_data import (parse_annotations, archive_index, HTTPRangeSource, MultipartReader,
                                          fetch_video, TRANSFER_CHUNK_BYTES, check_space)
    from evidencelab.ridge_adapter import fetch_ridge, load_ridge
    from evidencelab.ridge_weights import RidgeWeightCache, weight_contract
    import decord
    from PIL import Image
    baseline, source, development, output, ram = [p.resolve() for p in (args.baseline, args.source_cache, args.development_run, args.output, args.ram_root)]
    inputs = validate_inputs(source, baseline, development)
    for original in (baseline, source, development):
        if output == original or output in original.parents or original in output.parents:
            raise ValueError('Full run must be separate from every read-only input')
    validate_ram_root(ram)
    config = LongVideoConfig.load(args.config)
    old = inputs['base']
    execution, runtime = execution_profile(config), longvideo_runtime(config)
    if (config.contract() != old['config'] or runtime != old['runtime']
            or execution['resolved_dtype'] != old['resolved_dtype'] or file_hash(args.annotations) != old['annotation_sha256']):
        raise ValueError('Frozen configuration/runtime/annotation changed')
    author, weight_author = load_ridge(fetch_ridge(ram / 'ridge-upstream'))
    samples = sorted(parse_annotations(load(args.annotations)), key=lambda q: (q.video, q.id))
    if sorted(q.id for q in samples) != inputs['manifest']['ids']:
        raise ValueError('Full annotation coverage mismatch')
    weights = RidgeWeightCache(baseline / 'ridge-weights.json', weight_contract(config, old, weight_author), [q.public_question() for q in samples])
    weights.check_completed(inputs['bases'])
    contract = {'schema': SCHEMA, 'source_sha256': source_fingerprint(), **code_contract(), **inputs['inputs'],
                'config': config.contract(), 'runtime': runtime, 'resolved_dtype': execution['resolved_dtype'],
                'manifest': inputs['manifest'], 'variants': list(VARIANTS), 'prototype': events.PROTOTYPE,
                'selected_count': 1337 * len(VARIANTS), 'small_grid_policy': 'all-candidates-if-count-le64',
                'question_score_source': str(source), 'development_run': str(development)}
    stopped, old_handlers = [], {}
    for name in ('SIGINT', 'SIGTERM', 'SIGUSR1'):
        number = getattr(signal, name, None)
        if number is not None:
            old_handlers[number] = signal.signal(number, lambda signum, frame: stopped.append(signum))
    started = time.monotonic()
    stop = lambda: bool(stopped) or time.monotonic() - started >= args.max_seconds
    reader, video, backend, scorer = None, None, None, None
    video_path = ram / 'event-full-active-video.mp4'
    bases = {r['id']: r for r in inputs['bases']}
    try:
        with run_lock(output):
            store = Store(output, contract, (output / 'results.sqlite3').exists())
            try:
                for name in ('scores', 'curves'):
                    (output / name).mkdir(exist_ok=True)
                atomic_json(output / 'manifest.json', inputs['manifest'])
                done = store.done()
                pending = [q for q in samples if any(q.id + '::' + v not in done for v in VARIANTS)]
                if not pending:
                    return 0 if export(output, contract, store)['complete'] else 75
                if stop():
                    raise InterruptedError('Stop before setup')
                reader = MultipartReader(HTTPRangeSource(stop=stop))
                members = archive_index(reader, args.index, {q.video for q in pending})
                reader.clear_cache()
                reader.block_size = TRANSFER_CHUNK_BYTES
                backend = HFBackend(config)
                scorer = BlipITMScorer(config) if any(bases[q.id]['ridge_candidate_count'] > 64 for q in pending) else None
                current_video, completed = None, 0
                for sample in pending:
                    if stop() or (args.max_new_questions and completed >= args.max_new_questions):
                        break
                    if sample.video != current_video:
                        video = None
                        gc.collect()
                        video_path.unlink(missing_ok=True)
                        video_sha = fetch_video(reader, members['videos/' + sample.video], video_path, stop=stop)
                        reader.clear_cache()
                        video = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=2)
                        current_video = sample.video
                    prior = bases[sample.id]
                    fps = float(video.get_avg_fps())
                    if not len(video) or not math.isfinite(fps) or fps <= 0 or video_sha != prior['video_sha256']:
                        raise ValueError('Invalid or changed source video')
                    candidates = aks_candidates(len(video), fps)
                    if len(candidates) != prior['ridge_candidate_count']:
                        raise ValueError('Candidate grid changed')
                    score_path = output / 'scores' / (digest(sample.id) + '.json')
                    key = {**dev.cache_key(contract, sample, video_sha, candidates, fps, len(video)),
                           'public_query_sha256': digest({'question': sample.question, 'choices': sorted(sample.choices)})}
                    if score_path.exists():
                        score = validate_score(load(score_path), key)
                    else:
                        score = make_score(sample, contract, candidates, fps, len(video), video_sha,
                                           inputs['cached'], source, development, video, scorer, config.blip_batch_size, stop)
                        atomic_json(score_path, score)
                    def answer(indices, times):
                        backend.reset_peak()
                        begin = time.perf_counter()
                        check_space(ram, 0)
                        frames = [Image.fromarray(video[i].asnumpy()).convert('RGB') for i in indices]
                        backend.video_duration = len(video) / fps
                        decision = backend.answer(sample.public_question(), frames, times)
                        measurements = backend.measurements()
                        measurements['execution'] = execution
                        return decision, time.perf_counter() - begin, measurements
                    evaluate(sample, score, weights.records[sample.id], author, store,
                             output / 'curves' / (digest(sample.id) + '.json'), answer, prior, inputs['references'], stop)
                    completed += 1
                    store.event({'event': 'question_done', 'question_id': sample.id})
            except InterruptedError:
                store.event({'event': 'controlled_stop', 'signals': stopped})
            finally:
                video = None
                gc.collect()
                if reader is not None:
                    reader.clear_cache()
                video_path.unlink(missing_ok=True)
                try:
                    store.event({'event': 'session_end', 'wall_seconds': time.monotonic() - started, 'signals': stopped})
                    summary = export(output, contract, store)
                    print(json.dumps(summary, ensure_ascii=False), flush=True)
                finally:
                    store.close()
            return 0 if summary['complete'] else 75
    finally:
        for number, handler in old_handlers.items():
            signal.signal(number, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--mode', choices=['full1337'], default='full1337')
    for name in ('baseline', 'source-cache', 'development-run'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('config', 'annotations', 'output', 'ram-root', 'index'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--max-new-questions', type=int, default=0)
    parser.add_argument('--max-seconds', type=int, default=165600)
    args = parser.parse_args()
    if args.max_new_questions < 0 or args.max_seconds <= 0:
        parser.error('Invalid session limit')
    if args.preflight:
        data = validate_inputs(args.source_cache, args.baseline, args.development_run)
        print(json.dumps({'preflight': 'passed', 'questions': 1337, 'variants': list(VARIANTS), 'target_records': 6685,
                          'cached_question_scores': len(data['cached']['inventory']),
                          'all_candidate_questions': sum(r['ridge_candidate_count'] <= 64 for r in data['bases'])}))
        return 0
    if any(getattr(args, name) is None for name in ('config', 'annotations', 'output', 'ram_root', 'index')):
        parser.error('Worker requires config, annotations, output, ram-root and index')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
