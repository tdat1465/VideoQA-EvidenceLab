"""Isolated, instrumented replay of one full1337 control mismatch; never writes evaluation results."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import ridge_event_full as full
from evidencelab.data import digest, file_hash
from evidencelab.store import atomic_json

TARGET = 'longvideobench:val:o2F-N42Ufo4_2'
VIDEO = 'o2F-N42Ufo4.mp4'
CASES = ('cold_a', 'cold_b', 'baseline_history', 'full_history')


def validate_inputs(source, baseline, development, full_run):
    data = full.validate_inputs(source, baseline, development)
    contract = full.load(full_run / 'contract.json')
    if contract['schema'] != full.SCHEMA or any(contract[k] != v for k, v in full.code_contract().items()):
        raise ValueError('Full worker/helper contract changed')
    if any(contract[k] != data['base'][k] for k in ('config', 'runtime', 'resolved_dtype')):
        raise ValueError('Full runtime/configuration changed')
    if contract['baseline_predictions_sha256'] != file_hash(baseline / 'predictions.jsonl'):
        raise ValueError('Baseline changed')
    rows = full.read_rows(full_run / 'predictions.jsonl')
    if any(r['question_id'] == TARGET for r in rows):
        raise ValueError('Target already has full-run answers; investigate current status')
    bases = {r['id']: r for r in data['bases']}
    video_ids = sorted(q for q, r in bases.items() if r['video'] == VIDEO)
    if video_ids != [TARGET[:-1] + str(i) for i in range(3)]:
        raise ValueError('Unexpected video question group')
    caches, history = {}, {}
    for q in video_ids:
        path = full_run / 'scores' / (digest(q) + '.json')
        score = full.load(path)
        full.validate_score(score, score['key'])
        if (score['key']['question_id'] != q or score['key']['experiment'] != digest(contract)
                or score['key']['video_sha256'] != bases[q]['video_sha256']):
            raise ValueError('Full score provenance mismatch')
        caches[q] = score
        if q != TARGET:
            history[q] = {r['variant']: r for r in rows if r['question_id'] == q}
            if set(history[q]) != set(full.VARIANTS):
                raise ValueError('Need both preceding full-run questions complete')
    return {'data': data, 'contract': contract, 'bases': bases, 'video_ids': video_ids,
            'caches': caches, 'history': history,
            'snapshot': {'full_contract_sha256': file_hash(full_run / 'contract.json'),
                         'full_predictions_sha256': file_hash(full_run / 'predictions.jsonl'),
                         'target_score_file_sha256': file_hash(full_run / 'scores' / (digest(TARGET) + '.json'))}}


def snapshot_unchanged(snapshot, full_run):
    paths = {'full_contract_sha256': full_run / 'contract.json',
             'full_predictions_sha256': full_run / 'predictions.jsonl',
             'target_score_file_sha256': full_run / 'scores' / (digest(TARGET) + '.json')}
    return all(file_hash(paths[k]) == v for k, v in snapshot.items())


def pixel_fingerprint(array):
    return {'sha256': hashlib.sha256(array.tobytes()).hexdigest(),
            'shape': list(array.shape), 'dtype': str(array.dtype)}


class TrackedFrame:
    def __init__(self, array):
        self.array = array

    def asnumpy(self):
        return self.array


class TrackedReader:
    """Same ordered decoder reads; hash the actual array passed to frozen BLIP."""
    def __init__(self, reader):
        self.reader = reader
        self.capture = False
        self.raw_pixels = []
        self.transformed = []
        self.last_index = None

    def __getitem__(self, index):
        frame = self.reader[index]
        if not self.capture:
            return frame
        array = frame.asnumpy()
        self.last_index = int(index)
        self.raw_pixels.append({'frame_index': int(index), **pixel_fingerprint(array)})
        return TrackedFrame(array)

    def __len__(self):
        return len(self.reader)

    def get_avg_fps(self):
        return self.reader.get_avg_fps()


def compare_vectors(a, b):
    if len(a) != len(b):
        raise ValueError('Different score lengths')
    return {'exact_equal': a == b, 'different_samples': sum(x != y for x, y in zip(a, b)),
            'max_abs_difference': max((abs(x - y) for x, y in zip(a, b)), default=0.)}


def compare_cases(a, b):
    result = {'scores': compare_vectors(a['question_scores'], b['question_scores']),
              'selected_frames_equal': a['frame_indices'] == b['frame_indices'],
              'common_selected_frames': len(set(a['frame_indices']) & set(b['frame_indices']))}
    for key in ('raw_pixels', 'transformed_pixels'):
        if len(a[key]) != len(b[key]) or [x['frame_index'] for x in a[key]] != [x['frame_index'] for x in b[key]]:
            raise ValueError('Fingerprint grid/order changed')
        different = [x['frame_index'] for x, y in zip(a[key], b[key]) if x != y]
        result[key] = {'different_frames': len(different), 'frame_indices': different}
    return result


def run(args):
    from evidencelab.aks_adapter import aks_candidates
    from evidencelab.backends import HFBackend
    from evidencelab.focus_adapter import BlipITMScorer, blip_caption
    from evidencelab.hardware import execution_profile
    from evidencelab.longvideo import longvideo_runtime, validate_ram_root
    from evidencelab.longvideo_config import LongVideoConfig
    from evidencelab.longvideo_data import (parse_annotations, archive_index, HTTPRangeSource, MultipartReader,
                                          fetch_video, TRANSFER_CHUNK_BYTES)
    from evidencelab.ridge_adapter import fetch_ridge, load_ridge
    from PIL import Image
    import decord
    baseline, source, development, full_run, output, ram = [p.resolve() for p in
        (args.baseline, args.source_cache, args.development_run, args.full_run, args.output, args.ram_root)]
    for original in (baseline, source, development, full_run):
        if output == original or output in original.parents or original in output.parents:
            raise ValueError('Diagnostic output must be separate from all evaluation inputs')
    if (output / 'report.json').exists():
        raise FileExistsError('Use a new diagnostic session directory')
    context = validate_inputs(source, baseline, development, full_run)
    validate_ram_root(ram)
    config = LongVideoConfig.load(args.config)
    execution, runtime = execution_profile(config), longvideo_runtime(config)
    base_contract = context['data']['base']
    if (config.contract() != base_contract['config'] or runtime != base_contract['runtime']
            or execution['resolved_dtype'] != base_contract['resolved_dtype']
            or file_hash(args.annotations) != base_contract['annotation_sha256']):
        raise ValueError('Diagnostic runtime/configuration/annotation changed')
    samples = {s.id: s for s in parse_annotations(full.load(args.annotations)) if s.video == VIDEO}
    if set(samples) != set(context['video_ids']):
        raise ValueError('Annotation group mismatch')
    for q, sample in samples.items():
        key = context['caches'][q]['key']
        if (key['question_sha256'] != digest({'id': q, 'question': sample.question})
                or key['public_query_sha256'] != digest({'question': sample.question, 'choices': sorted(sample.choices)})):
            raise ValueError('Public question/choices changed')
    weights = full.load(baseline / 'ridge-weights.json')['records']
    author, _ = load_ridge(fetch_ridge(ram / 'ridge-upstream'))
    output.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'ridge-event-drift-diagnostic-v1', 'target': TARGET, 'video': VIDEO,
              'evaluation_results': False, 'complete': False, 'planned_cases': list(CASES), 'cases': {},
              'snapshot_before': context['snapshot'], 'execution': execution, 'runtime': runtime,
              'script_sha256': file_hash(Path(__file__)), 'full_code_contract': full.code_contract(),
              'limitations': ['No historical baseline raw-score or pixel fingerprints are available.',
                  'Decoder histories use the recorded frame accesses within this video; preceding videos and global allocator history are not reproduced.',
                  'Instrumented timing includes hashing; this is not a latency benchmark.'],
              'history': {}}
    started, stopped, old_handlers = time.monotonic(), [], {}
    for name in ('SIGINT', 'SIGTERM', 'SIGUSR1'):
        number = getattr(signal, name, None)
        if number is not None:
            old_handlers[number] = signal.signal(number, lambda signum, frame: stopped.append(signum))
    stop = lambda: bool(stopped) or time.monotonic() - started >= args.max_seconds
    network, decoder, status = None, None, 0
    video_path = ram / 'diagnostic-video.mp4'
    try:
        network = MultipartReader(HTTPRangeSource(stop=stop))
        members = archive_index(network, args.index, {VIDEO})
        network.clear_cache()
        network.block_size = TRANSFER_CHUNK_BYTES
        video_sha = fetch_video(network, members['videos/' + VIDEO], video_path, stop=stop)
        network.clear_cache()
        if video_sha != context['bases'][TARGET]['video_sha256']:
            raise ValueError('Video bytes differ from baseline')
        backend = HFBackend(config)
        scorer = BlipITMScorer(config)
        torch = backend.torch
        report['torch_flags'] = {'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32, 'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'float32_matmul_precision': torch.get_float32_matmul_precision(), 'num_threads': torch.get_num_threads(),
            'gpu_uuid': str(getattr(torch.cuda.get_device_properties(0), 'uuid', 'unavailable'))}
        original_transform = scorer.transform
        def tracked_transform(image):
            tensor = original_transform(image)
            if decoder is not None and decoder.capture:
                decoder.transformed.append({'frame_index': decoder.last_index, **pixel_fingerprint(tensor.numpy())})
            return tensor
        scorer.transform = tracked_transform
        grids = context['caches'][TARGET]['key']['candidate_indices']
        fps = context['caches'][TARGET]['key']['fps']
        target = samples[TARGET]
        tokenized = scorer.processor.tokenizer(blip_caption(target.question), truncation=True, max_length=35)
        ids = list(tokenized['input_ids'])
        ids[0] = scorer.enc_id
        report['target_input'] = {'question': target.question, 'choices': list(target.choices),
                                  'blip_caption': blip_caption(target.question), 'input_ids': ids,
                                  'weights': weights[TARGET]['weights'], 'video_sha256': video_sha,
                                  'candidate_indices': grids, 'fps': fps}
        def score_question(sample):
            scorer.reset()
            raw = []
            for offset in range(0, len(grids), config.blip_batch_size):
                if stop():
                    raise InterruptedError('Diagnostic stop during scoring')
                raw.extend(scorer(decoder, sample.question, grids[offset:offset + config.blip_batch_size]))
            return raw
        for case in CASES:
            if stop():
                raise InterruptedError('Diagnostic stop between cases')
            decoder = None
            gc.collect()
            decoder = TrackedReader(decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=2))
            actual_grid = aks_candidates(len(decoder), float(decoder.get_avg_fps()))
            if actual_grid != grids or float(decoder.get_avg_fps()) != fps:
                raise ValueError('Diagnostic decoder grid/FPS differs from failing cache')
            history = []
            if case in ('baseline_history', 'full_history'):
                for q in context['video_ids'][:-1]:
                    sample = samples[q]
                    raw = score_question(sample)
                    phase = {'question_id': q, 'question_score_vs_full_cache': compare_vectors(raw, context['caches'][q]['question_scores']), 'answer_forwards': []}
                    if case == 'full_history':
                        plan = full.events.event_plan(sample.public_question(), scorer.processor.tokenizer)
                        if plan != context['caches'][q]['query_plan']:
                            raise ValueError('History query plan differs from recorded full run')
                        if plan['queries']:
                            full.dev.query_frozen.score_options(scorer, decoder, grids, plan['queries'], config.blip_batch_size, stop)
                        accesses = [context['history'][q][v] for v in full.VARIANTS if context['history'][q][v]['answer_inference_performed']]
                    else:
                        accesses = [context['bases'][q]]
                    for r in accesses:
                        if stop():
                            raise InterruptedError('Diagnostic stop during history replay')
                        indices = r['frame_indices'] if 'frame_indices' in r else r['calls'][0]['frame_indices']
                        frames = [Image.fromarray(decoder[i].asnumpy()).convert('RGB') for i in indices]
                        backend.video_duration = len(decoder) / fps
                        decision = backend.answer(sample.public_question(), frames, [i / fps for i in indices])
                        frames = None
                        phase['answer_forwards'].append({'variant': r.get('variant', 'sigma2'),
                            'prediction': decision.prediction, 'historical_prediction': r['prediction'],
                            'match': decision.prediction == r['prediction'], 'frame_indices': indices})
                    history.append(phase)
            report['history'][case] = history
            decoder.capture = True
            begin = time.perf_counter()
            raw = score_question(target)
            decoder.capture = False
            score = {**context['caches'][TARGET], 'question_scores': raw}
            positions, _ = full.select_full(author, score, weights[TARGET]['weights'], 'sigma2')
            selected = [grids[i] for i in positions]
            if [x['frame_index'] for x in decoder.raw_pixels] != grids or [x['frame_index'] for x in decoder.transformed] != grids:
                raise ValueError('Incomplete fingerprint trace')
            expected = context['bases'][TARGET]['calls'][0]['frame_indices']
            report['cases'][case] = {'question_scores': raw, 'frame_indices': selected,
                'matches_old_baseline': selected == expected, 'common_with_old_baseline': len(set(selected) & set(expected)),
                'scores_vs_failed_cache': compare_vectors(raw, context['caches'][TARGET]['question_scores']),
                'raw_pixels': decoder.raw_pixels, 'transformed_pixels': decoder.transformed,
                'instrumented_seconds': time.perf_counter() - begin}
            print(f'Diagnostic {case}: baseline overlap {len(set(selected) & set(expected))}/64', flush=True)
            atomic_json(output / 'report.json', report)
        report['comparisons'] = {name: compare_cases(report['cases']['cold_a'], report['cases'][name]) for name in CASES[1:]}
        report['complete'] = True
    except InterruptedError:
        report['error_type'] = 'InterruptedError'
        report['complete'] = False
        status = 75
    except BaseException as error:
        report['error_type'] = type(error).__name__
        report['complete'] = False
        raise
    finally:
        decoder = None
        gc.collect()
        if network is not None:
            network.clear_cache()
        video_path.unlink(missing_ok=True)
        report['snapshot_unchanged'] = snapshot_unchanged(context['snapshot'], full_run)
        if not report['snapshot_unchanged']:
            report['complete'] = False
            status = 1
        report['wall_seconds'] = time.monotonic() - started
        report['signals'] = stopped
        atomic_json(output / 'report.json', report)
        for number, handler in old_handlers.items():
            signal.signal(number, handler)
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--mode', choices=['diagnose'], default='diagnose')
    for name in ('baseline', 'source-cache', 'development-run', 'full-run'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('config', 'annotations', 'output', 'ram-root', 'index'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--max-new-questions', type=int, default=0)
    p.add_argument('--max-seconds', type=int, default=13800)
    args = p.parse_args()
    if args.max_seconds <= 0 or args.max_new_questions != 0:
        p.error('Diagnostic has a fixed single-video scope and positive session time')
    if args.preflight:
        data = validate_inputs(args.source_cache, args.baseline, args.development_run, args.full_run)
        print(json.dumps({'preflight': 'passed', 'target': TARGET, 'cases': list(CASES), 'snapshot': data['snapshot']}))
        return 0
    if any(getattr(args, name) is None for name in ('config', 'annotations', 'output', 'ram_root', 'index')):
        p.error('Need config, annotations, output, ram-root and index inside a GPU allocation')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
