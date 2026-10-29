"""Canonical RGB v2: fresh Uniform64 and five frozen RIDGE methods."""
from __future__ import annotations
import argparse
import gc
import json
import math
from pathlib import Path
import shutil
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import ridge_event_full as full
import ridge_canonical_frames as pixels
from evidencelab.data import digest, file_hash
from evidencelab.pipeline import Decision
from evidencelab.selectors import uniform
from evidencelab.store import Store, atomic_json, run_lock

SCHEMA = 'ridge-event-canonical-rgb-v2'
VARIANTS = ('uniform64', *full.VARIANTS)
REGRESSION_VIDEO = 'o2F-N42Ufo4.mp4'
load = full.load


def prepare(source, baseline, development):
    data = full.validate_inputs(source, baseline, development)
    manifest = data['manifest']
    pilot = full.dev.pilot_manifest(data['bases'], data['development_contract']['manifest'])
    regression = [r['id'] for r in data['bases'] if r['video'] == REGRESSION_VIDEO]
    pilot_ids = sorted(set(pilot['ids']) | set(regression))
    if len(regression) != 3 or len(pilot_ids) != 23:
        raise ValueError('Fixed pilot20 plus three decoder regression questions changed')
    manifest = {**manifest, 'pilot_ids': pilot_ids,
                'engineering_videos': sorted(set(manifest['development_videos']) | {REGRESSION_VIDEO}),
                'evaluation_note': 'New RGB protocol; old answers/curves are not comparable controls. '
                    'Report dev200 plus decoder-regression videos separately. Remaining validation is not a pristine test set.'}
    return {**data, 'manifest': manifest}


def code_contract():
    return {**full.code_contract(), 'worker_sha256': file_hash(Path(__file__)),
            'frozen_full_worker_sha256': file_hash(Path(full.__file__)),
            'canonical_pixels_sha256': file_hash(Path(pixels.__file__))}


def expected_contract(data):
    return {'schema': SCHEMA, 'source_sha256': full.source_fingerprint(), **code_contract(),
            **data['inputs'], 'config': data['base']['config'], 'runtime': data['base']['runtime'],
            'resolved_dtype': data['base']['resolved_dtype'], 'manifest': data['manifest'],
            'variants': list(VARIANTS), 'prototype': full.events.PROTOTYPE,
            'selected_count': 1337 * len(VARIANTS), 'pixel_policy': pixels.POLICY,
            'score_policy': 'fresh canonical RGB; no v1 score reuse',
            'answer_policy': 'fresh or identical frames inside this v2 journal only',
            'uniform_policy': '64 evenly spaced native source-frame ordinals',
            'small_grid_policy': 'RIDGE uses all candidates if count le64'}


def require_pilot(rows, manifest):
    actual = {r['id'] for r in rows}
    expected = {q + '::' + v for q in manifest['pilot_ids'] for v in VARIANTS}
    if not expected <= actual:
        raise ValueError(f'Complete canonical pilot23 first: {len(expected & actual)}/138 records')


def score_key(contract, sample, video, candidates):
    return {**full.dev.cache_key(contract, sample, video.manifest['video_sha256'], candidates,
                                video.get_avg_fps(), len(video)),
            'public_query_sha256': digest({'question': sample.question, 'choices': sorted(sample.choices)}),
            'pixel_manifest_sha256': digest(video.manifest)}


def make_score(key, sample, video, scorer, batch_size, stop):
    record = {'key': key, 'question_scores': None, 'per_event_scores': [], 'query_plan': None,
              'question_origin': 'all-candidates', 'event_origin': 'not-needed',
              'question_scoring_seconds': 0., 'event_scoring_seconds': 0., 'question_batches': 0, 'event_batches': 0}
    candidates = key['candidate_indices']
    if len(candidates) > 64:
        scorer.reset()
        begin, raw = time.perf_counter(), []
        for offset in range(0, len(candidates), batch_size):
            if stop():
                raise InterruptedError('Stop during canonical question scoring')
            raw.extend(scorer(video, sample.question, candidates[offset:offset + batch_size]))
        record.update(question_scores=raw, question_origin='fresh-canonical-blip',
                      question_scoring_seconds=time.perf_counter() - begin, question_batches=scorer.calls)
        plan = full.events.event_plan(sample.public_question(), scorer.processor.tokenizer)
        record['query_plan'] = plan
        if plan['queries']:
            begin = time.perf_counter()
            values = full.dev.query_frozen.score_options(scorer, video, candidates, plan['queries'], batch_size, stop)
            record.update(per_event_scores=values, event_origin='fresh-canonical-blip',
                          event_scoring_seconds=time.perf_counter() - begin, event_batches=scorer.calls)
    record['sha256'] = digest(record)
    return full.validate_score(record, key)


def selected_frames(author, score, weights, variant):
    if variant == 'uniform64':
        return uniform(score['key']['source_frames'], 64), {'selection_mode': 'native-source-uniform64'}
    positions, details = full.select_full(author, score, weights, variant)
    return [score['key']['candidate_indices'][p] for p in positions], details


def evaluate(sample, score, weights, author, store, curve_path, answer, baseline, stop):
    previous = [r for r in store.results() if r['question_id'] == sample.id]
    done = {r['variant'] for r in previous}
    stamp = score['key']['pixel_manifest_sha256']
    if any(r['pixel_manifest_sha256'] != stamp or r['score_sha256'] != score['sha256'] for r in previous):
        raise ValueError('Resume pixels or scores changed')
    same_frames = {tuple(r['frame_indices']): r for r in previous}
    diagnostics = {'question_id': sample.id, 'score_sha256': score['sha256'],
                   'pixel_manifest_sha256': stamp, 'variants': {}}
    for variant in VARIANTS:
        indices, details = selected_frames(author, score, weights['weights'], variant)
        expected_n = min(64, score['key']['source_frames'] if variant == 'uniform64' else len(score['key']['candidate_indices']))
        if len(indices) != expected_n or indices != sorted(set(indices)):
            raise ValueError('Frame budget/order mismatch')
        diagnostics['variants'][variant] = {**details, 'frame_indices': indices}
        if variant in done:
            old = next(r for r in previous if r['variant'] == variant)
            if old['frame_indices'] != indices:
                raise ValueError('Resume selector replay changed')
            continue
        if stop():
            raise InterruptedError('Stop between canonical variants')
        times = [i / score['key']['fps'] for i in indices]
        prior = same_frames.get(tuple(indices))
        if prior is None:
            decision, seconds, measurements = answer(indices, times)
        else:
            decision, seconds = Decision(prior['scores'], prior['input_tokens']), 0.
            measurements = {'execution': prior['execution']}
        if (len(decision.scores) != len(sample.choices) or
                any(not math.isfinite(x) or not 0 <= x <= 1 for x in decision.scores) or
                abs(sum(decision.scores) - 1) > 1e-4):
            raise ValueError('Invalid answer probabilities')
        row = {'id': sample.id + '::' + variant, 'question_id': sample.id, 'variant': variant,
               'question': sample.question, 'choices': list(sample.choices), 'answer': sample.answer,
               'prediction': decision.prediction, 'correct': decision.prediction == sample.answer,
               'video': sample.video, 'question_type': sample.question_type, 'duration_group': sample.duration_group,
               'frame_indices': indices, 'timestamps': times, 'final_frames': len(indices),
               'scores': decision.scores, 'input_tokens': decision.input_tokens,
               'score_sha256': score['sha256'], 'weight_sha256': weights['record_sha256'],
               'pixel_manifest_sha256': stamp, 'answer_decode_seconds': seconds,
               'answer_inference_performed': prior is None, 'reused_answer_from': None if prior is None else prior['id'],
               'old_baseline_frame_match': indices == baseline['calls'][0]['frame_indices'],
               'old_baseline_prediction_match': decision.prediction == baseline['prediction'], **measurements}
        store.put(row)
        same_frames[tuple(indices)] = row
        atomic_json(curve_path, diagnostics)
        print(f'Committed canonical {sample.id}: {variant}, correct={row["correct"]}, reused={prior is not None}', flush=True)
    atomic_json(curve_path, diagnostics)


def export(output, contract, store):
    rows, grouped = store.results(), {}
    allowed = set(contract['manifest']['ids'])
    for r in rows:
        if (r['question_id'] not in allowed or r['variant'] not in VARIANTS or
                r['id'] != r['question_id'] + '::' + r['variant']):
            raise ValueError('Foreign canonical row')
        grouped.setdefault(r['question_id'], {})[r['variant']] = r
    completed = {q for q, g in grouped.items() if set(g) == set(VARIANTS)}
    engineering = set(contract['manifest']['engineering_videos'])
    subsets = {}
    for name, ids in [('all', completed), ('engineering_videos', {q for q in completed if grouped[q]['sigma2']['video'] in engineering}),
                      ('outside_engineering_videos', {q for q in completed if grouped[q]['sigma2']['video'] not in engineering})]:
        subsets[name] = {'questions': len(ids), 'correct': {v: sum(grouped[q][v]['correct'] for q in ids) for v in VARIANTS}}
    pilot_n = len(set(contract['manifest']['pilot_ids']) & completed)
    summary = {'complete': len(rows) == contract['selected_count'], 'completed_variant_answers': len(rows),
               'completed_questions': len(completed), 'target_variant_answers': contract['selected_count'],
               'pilot23': {'completed_questions': pilot_n, 'target_questions': 23, 'complete': pilot_n == 23},
               'variants': {v: {'n': sum(r['variant'] == v for r in rows),
                                'correct': sum(r['variant'] == v and r['correct'] for r in rows)} for v in VARIANTS},
               'subsets': subsets, 'same_run_answer_reuses': sum(not r['answer_inference_performed'] for r in rows),
               'actual_answer_forwards': sum(r['answer_inference_performed'] for r in rows),
               'new_qwen_calls': 0, 'interpretation': contract['manifest']['evaluation_note']}
    atomic_json(output / 'summary.json', summary)
    with (output / 'predictions.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
        for r in rows:
            stream.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n')
    return summary


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
    baseline, source, development, output, ram = [p.resolve() for p in
        (args.baseline, args.source_cache, args.development_run, args.output, args.ram_root)]
    data = prepare(source, baseline, development)
    for original in (baseline, source, development):
        if output == original or output in original.parents or original in output.parents:
            raise ValueError('Canonical run must be separate from read-only inputs')
    validate_ram_root(ram)
    config = LongVideoConfig.load(args.config)
    execution, runtime = execution_profile(config), longvideo_runtime(config)
    if (config.contract() != data['base']['config'] or runtime != data['base']['runtime'] or
            execution['resolved_dtype'] != data['base']['resolved_dtype'] or
            file_hash(args.annotations) != data['base']['annotation_sha256']):
        raise ValueError('Pinned runtime/config/annotation changed')
    author, weight_author = load_ridge(fetch_ridge(ram / 'ridge-upstream'))
    samples = sorted(parse_annotations(load(args.annotations)), key=lambda q: (q.video, q.id))
    if sorted(q.id for q in samples) != data['manifest']['ids']:
        raise ValueError('Annotation coverage changed')
    weights = RidgeWeightCache(baseline / 'ridge-weights.json', weight_contract(config, data['base'], weight_author),
                               [q.public_question() for q in samples])
    weights.check_completed(data['bases'])
    contract, bases = expected_contract(data), {r['id']: r for r in data['bases']}
    stopped, old_handlers = [], {}
    for name in ('SIGINT', 'SIGTERM', 'SIGUSR1'):
        number = getattr(signal, name, None)
        if number is not None:
            old_handlers[number] = signal.signal(number, lambda signum, frame: stopped.append(signum))
    started = time.monotonic()
    stop = lambda: bool(stopped) or time.monotonic() - started >= args.max_seconds
    reader, video, current_video = None, None, None
    video_path, cache_root = ram / 'canonical-active-video.mp4', ram / 'canonical-active-pixels'
    if cache_root.exists() or video_path.exists():
        raise FileExistsError('Use a fresh session tmpfs directory')
    try:
        with run_lock(output):
            store = Store(output, contract, (output / 'results.sqlite3').exists())
            try:
                for name in ('scores', 'curves', 'pixels'):
                    (output / name).mkdir(exist_ok=True)
                atomic_json(output / 'manifest.json', data['manifest'])
                if args.mode == 'full1337':
                    require_pilot(store.results(), data['manifest'])
                scope = set(data['manifest']['pilot_ids'] if args.mode == 'pilot23' else data['manifest']['ids'])
                done = store.done()
                pending = [q for q in samples if q.id in scope and any(q.id + '::' + v not in done for v in VARIANTS)]
                if not pending:
                    export(output, contract, store)
                    return 0
                if stop():
                    raise InterruptedError('Stop before setup')
                reader = MultipartReader(HTTPRangeSource(stop=stop))
                members = archive_index(reader, args.index, {q.video for q in pending})
                reader.clear_cache()
                reader.block_size = TRANSFER_CHUNK_BYTES
                backend = HFBackend(config)
                scorer = BlipITMScorer(config) if any(bases[q.id]['ridge_candidate_count'] > 64 for q in pending) else None
                completed = 0
                for sample in pending:
                    if stop() or (args.max_new_questions and completed >= args.max_new_questions):
                        break
                    if current_video != sample.video:
                        video = None
                        gc.collect()
                        if cache_root.exists():
                            shutil.rmtree(cache_root)
                        video_path.unlink(missing_ok=True)
                        video_sha = fetch_video(reader, members['videos/' + sample.video], video_path, stop=stop)
                        reader.clear_cache()
                        if video_sha != bases[sample.id]['video_sha256']:
                            raise ValueError('Source video changed')
                        metadata = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
                        count, fps = len(metadata), float(metadata.get_avg_fps())
                        del metadata
                        gc.collect()
                        candidates = aks_candidates(count, fps)
                        needed = sorted(set(candidates) | set(uniform(count, 64)))
                        print(f'Canonical decode {sample.video}: {count} source frames, {len(needed)} cached RGB frames, two passes', flush=True)
                        video = pixels.build(video_path, cache_root, count, fps, needed, video_sha, check_space, stop)
                        manifest_path = output / 'pixels' / (digest(sample.video) + '.json')
                        if manifest_path.exists() and load(manifest_path) != video.manifest:
                            raise ValueError('Canonical resume pixel manifest changed')
                        atomic_json(manifest_path, video.manifest)
                        current_video = sample.video
                    if len(candidates) != bases[sample.id]['ridge_candidate_count']:
                        raise ValueError('Candidate grid changed')
                    key = score_key(contract, sample, video, candidates)
                    path = output / 'scores' / (digest(sample.id) + '.json')
                    if path.exists():
                        score = full.validate_score(load(path), key)
                    else:
                        score = make_score(key, sample, video, scorer, config.blip_batch_size, stop)
                        atomic_json(path, score)
                    def answer(indices, times):
                        backend.reset_peak()
                        begin = time.perf_counter()
                        check_space(ram, 0)
                        frames = [Image.fromarray(video[i].asnumpy()) for i in indices]
                        backend.video_duration = len(video) / video.get_avg_fps()
                        decision = backend.answer(sample.public_question(), frames, times)
                        measurements = {**backend.measurements(), 'execution': execution}
                        return decision, time.perf_counter() - begin, measurements
                    evaluate(sample, score, weights.records[sample.id], author, store,
                             output / 'curves' / (digest(sample.id) + '.json'), answer, bases[sample.id], stop)
                    completed += 1
                    store.event({'event': 'question_done', 'question_id': sample.id})
            except InterruptedError:
                store.event({'event': 'controlled_stop', 'signals': stopped})
            finally:
                video = None
                gc.collect()
                if reader is not None:
                    reader.clear_cache()
                if cache_root.exists():
                    shutil.rmtree(cache_root)
                video_path.unlink(missing_ok=True)
                try:
                    store.event({'event': 'session_end', 'wall_seconds': time.monotonic() - started})
                    summary = export(output, contract, store)
                    print(json.dumps(summary, ensure_ascii=False), flush=True)
                finally:
                    store.close()
            return 0 if summary['complete'] or (args.mode == 'pilot23' and summary['pilot23']['complete']) else 75
    finally:
        for number, handler in old_handlers.items():
            signal.signal(number, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--mode', choices=['pilot23', 'full1337'], default='pilot23')
    for name in ('baseline', 'source-cache', 'development-run'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('config', 'annotations', 'output', 'ram-root', 'index'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--max-new-questions', type=int, default=0)
    parser.add_argument('--max-seconds', type=int, default=165600)
    args = parser.parse_args()
    if args.max_new_questions < 0 or args.max_seconds <= 0:
        parser.error('Invalid limits')
    if args.preflight:
        data = prepare(args.source_cache, args.baseline, args.development_run)
        print(json.dumps({'preflight': 'passed', 'pilot_questions': len(data['manifest']['pilot_ids']),
                          'pilot_records': 138, 'full_records': 8022,
                          'contract_sha256': digest(expected_contract(data))}))
        return 0
    if any(getattr(args, n) is None for n in ('config', 'annotations', 'output', 'ram_root', 'index')):
        parser.error('Worker requires config, annotations, output, ram-root and index')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
