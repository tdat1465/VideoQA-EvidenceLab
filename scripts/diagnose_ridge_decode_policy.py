"""Fixed decoder-policy comparisons on one video; no evaluation writes or answerer loads."""
from __future__ import annotations
import argparse
import base64
import gc
from itertools import combinations
import json
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import diagnose_ridge_event_drift as prior
from evidencelab.data import Question, file_hash
from evidencelab.store import atomic_json

TARGET, VIDEO = prior.TARGET, prior.VIDEO
REFERENCE_SHA256 = '3995571579c2aa717ea941ffbae512dd2eb1256e7214942b470bddee795d3957'
POLICIES = ('decord_index_t1', 'decord_seq_t1', 'decord_seq_t2', 'pyav_seq_t1')
CASES = tuple(policy + '_' + repeat for policy in POLICIES for repeat in ('a', 'b'))
PROBES = {11218, 11219, 11220, 11221, 11222}


def validate_inputs(source, baseline, development, full_run, reference_report):
    context = prior.validate_inputs(source, baseline, development, full_run)
    reference = prior.full.load(reference_report)
    if (file_hash(reference_report) != REFERENCE_SHA256 or not reference['complete']
            or not reference['snapshot_unchanged'] or reference['target'] != TARGET
            or reference['snapshot_before'] != context['snapshot']
            or reference['full_code_contract'] != prior.full.code_contract()):
        raise ValueError('Need the unchanged completed report from diagnostic job 80368')
    return {**context, 'reference': reference, 'reference_sha256': file_hash(reference_report)}


class LazyFrame:
    def __init__(self, native, to_array):
        self.native, self.to_array = native, to_array

    def asnumpy(self):
        return self.to_array(self.native)


class SequentialReader:
    """Decode every presentation-order frame; never seek or retain the whole video."""
    def __init__(self, iterator, to_array, fps, expected_count, close=lambda: None,
                 timestamp=lambda frame: None, stop=lambda: False):
        self.iterator, self.to_array = iter(iterator), to_array
        self.fps, self.expected_count = fps, expected_count
        self.close_resource, self.timestamp, self.stop = close, timestamp, stop
        self.position, self.native, self.ended = -1, None, False
        self.probes, self.candidate_pts = {}, []
        self.first_pts = None
        self.closed = False

    def _advance(self):
        if self.stop():
            raise InterruptedError('Stop during sequential decode')
        self.native = next(self.iterator)
        self.position += 1
        pts = self.timestamp(self.native)
        if self.position == 0:
            self.first_pts = pts
        if self.position in PROBES:
            from PIL import Image
            array = self.to_array(self.native)
            thumbnail = Image.fromarray(array).convert('RGB').resize((32, 18), Image.Resampling.BILINEAR)
            self.probes[str(self.position)] = {**prior.pixel_fingerprint(array), 'pts_seconds': pts,
                'thumbnail_size': [32, 18], 'thumbnail_rgb_base64': base64.b64encode(thumbnail.tobytes()).decode('ascii')}

    def __getitem__(self, index):
        if self.closed:
            raise ValueError('Reader is closed')
        index = int(index)
        if index < 0 or index < self.position:
            raise ValueError('Sequential reader rejects backward/negative access; create a fresh reader')
        while self.position < index:
            self._advance()
        self.candidate_pts.append({'frame_index': index, 'pts_seconds': self.timestamp(self.native)})
        return LazyFrame(self.native, self.to_array)

    def __len__(self):
        return self.expected_count

    def get_avg_fps(self):
        return self.fps

    def finish(self):
        if not self.ended:
            try:
                while True:
                    self._advance()
            except StopIteration:
                self.ended = True
        count = self.position + 1
        errors = [abs((r['pts_seconds'] - self.first_pts) - r['frame_index'] / self.fps)
                  for r in self.candidate_pts if r['pts_seconds'] is not None and self.first_pts is not None]
        return {'observed_decoded_frames': count, 'expected_frames': self.expected_count,
                'frame_count_matches': count == self.expected_count, 'candidate_pts': self.candidate_pts,
                'max_pts_error_vs_reference_seconds': max(errors) if errors else None,
                'probes': self.probes}

    def close(self):
        if not self.closed:
            self.closed = True
            self.native = None
            self.iterator = iter(())
            self.close_resource()


def open_reader(policy, path, fps, count, stop):
    if policy == 'pyav_seq_t1':
        import av
        container = av.open(str(path))
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        reader = SequentialReader(container.decode(video=0), lambda f: f.to_ndarray(format='rgb24'),
            fps, count, container.close,
            lambda f: None if f.pts is None or f.time_base is None else float(f.pts * f.time_base), stop)
        return reader, {'api': 'PyAV container.decode; no seek', 'thread_count': stream.codec_context.thread_count,
            'stream_average_rate': None if stream.average_rate is None else float(stream.average_rate),
            'library_versions': av.library_versions}
    import decord
    threads = 2 if policy == 'decord_seq_t2' else 1
    native = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=threads)
    metadata = {'api': 'decord next()' if '_seq_' in policy else 'decord index access',
                'thread_count': threads, 'reported_frames': len(native), 'reported_fps': float(native.get_avg_fps())}
    if len(native) != count or float(native.get_avg_fps()) != fps:
        raise ValueError('Decord metadata/grid differs from the pinned input')
    if '_seq_' not in policy:
        return native, metadata
    def frames():
        while True:
            try:
                frame = native.next()
            except StopIteration:
                return
            yield frame
    return SequentialReader(frames(), lambda f: f.asnumpy(), fps, count, stop=stop), metadata


def run(args):
    from evidencelab.focus_adapter import BlipITMScorer
    from evidencelab.hardware import execution_profile
    from evidencelab.longvideo import longvideo_runtime, validate_ram_root
    from evidencelab.longvideo_config import LongVideoConfig
    from evidencelab.longvideo_data import (archive_index, HTTPRangeSource, MultipartReader, fetch_video, TRANSFER_CHUNK_BYTES)
    from evidencelab.ridge_adapter import fetch_ridge, load_ridge
    import torch
    baseline, source, development, full_run, output, ram, reference_path = [p.resolve() for p in
        (args.baseline, args.source_cache, args.development_run, args.full_run, args.output, args.ram_root, args.reference_report)]
    for original in (baseline, source, development, full_run, reference_path.parent):
        if output == original or output in original.parents or original in output.parents:
            raise ValueError('Decoder-policy diagnostic must use a separate output directory')
    if (output / 'report.json').exists():
        raise FileExistsError('Use a new diagnostic session')
    context = validate_inputs(source, baseline, development, full_run, reference_path)
    reference = context['reference']
    validate_ram_root(ram)
    config = LongVideoConfig.load(args.config)
    execution, runtime = execution_profile(config), longvideo_runtime(config)
    if (config.contract() != context['data']['base']['config'] or runtime != context['data']['base']['runtime']
            or execution['resolved_dtype'] != context['data']['base']['resolved_dtype']
            or file_hash(args.annotations) != context['data']['base']['annotation_sha256']):
        raise ValueError('Pinned runtime/configuration/annotations changed')
    question = Question(TARGET, reference['target_input']['question'], tuple(reference['target_input']['choices']))
    grids = reference['target_input']['candidate_indices']
    fps = reference['target_input']['fps']
    count = context['caches'][TARGET]['key']['source_frames']
    weights = reference['target_input']['weights']
    author, _ = load_ridge(fetch_ridge(ram / 'ridge-upstream'))
    output.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'ridge-decode-policy-diagnostic-v1', 'target': TARGET, 'video': VIDEO,
        'evaluation_results': False, 'complete': False, 'planned_cases': list(CASES), 'cases': {},
        'models_loaded': ['BLIP ITM FP32'], 'answerer_loaded': False, 'new_answer_forwards': 0,
        'reference_report_sha256': context['reference_sha256'], 'snapshot_before': context['snapshot'],
        'script_sha256': file_hash(Path(__file__)), 'execution': execution, 'runtime': runtime,
        'limitations': ['One video only; repeatability here is not full-dataset validation.',
            'Sequential frame ordinals and PyAV PTS/count must be checked before considering decoder equivalence.',
            'Different decoder RGB conversion may differ even for the same picture; cross-library hashes alone do not establish incorrect decoding.',
            'Only BLIP is loaded; global allocator history differs from the old Molmo2 run.']}
    stopped, old_handlers, status = [], {}, 0
    for name in ('SIGINT', 'SIGTERM', 'SIGUSR1'):
        number = getattr(signal, name, None)
        if number is not None:
            old_handlers[number] = signal.signal(number, lambda signum, frame: stopped.append(signum))
    started = time.monotonic()
    stop = lambda: bool(stopped) or time.monotonic() - started >= args.max_seconds
    network, decoder, underlying = None, None, None
    video_path = ram / 'decode-policy-video.mp4'
    try:
        network = MultipartReader(HTTPRangeSource(stop=stop))
        members = archive_index(network, args.index, {VIDEO})
        network.clear_cache()
        network.block_size = TRANSFER_CHUNK_BYTES
        video_sha = fetch_video(network, members['videos/' + VIDEO], video_path, stop=stop)
        network.clear_cache()
        if video_sha != reference['target_input']['video_sha256']:
            raise ValueError('Video SHA changed')
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        scorer = BlipITMScorer(config)
        transform = scorer.transform
        def tracking_transform(image):
            value = transform(image)
            if decoder is not None and decoder.capture:
                decoder.transformed.append({'frame_index': decoder.last_index, **prior.pixel_fingerprint(value.numpy())})
            return value
        scorer.transform = tracking_transform
        report['gpu_uuid'] = str(getattr(torch.cuda.get_device_properties(0), 'uuid', 'unavailable'))
        for case in CASES:
            if stop():
                raise InterruptedError('Stop between decoding cases')
            policy = case.rsplit('_', 1)[0]
            underlying, metadata = open_reader(policy, video_path, fps, count, stop)
            decoder = prior.TrackedReader(underlying)
            decoder.capture = True
            begin, raw = time.perf_counter(), []
            scorer.reset()
            for offset in range(0, len(grids), config.blip_batch_size):
                if stop():
                    raise InterruptedError('Stop during decoding/scoring')
                raw.extend(scorer(decoder, question.question, grids[offset:offset + config.blip_batch_size]))
            decoder.capture = False
            if isinstance(underlying, SequentialReader):
                metadata.update(underlying.finish())
            if [r['frame_index'] for r in decoder.raw_pixels] != grids or [r['frame_index'] for r in decoder.transformed] != grids:
                raise ValueError('Fingerprint trace incomplete')
            positions, _ = prior.full.select_full(author, {**context['caches'][TARGET], 'question_scores': raw}, weights, 'sigma2')
            selected = [grids[i] for i in positions]
            expected = context['bases'][TARGET]['calls'][0]['frame_indices']
            result = {'question_scores': raw, 'frame_indices': selected, 'raw_pixels': decoder.raw_pixels,
                'transformed_pixels': decoder.transformed, 'decoder': metadata,
                'common_with_old_baseline': len(set(selected) & set(expected)), 'matches_old_baseline': selected == expected,
                'instrumented_seconds': time.perf_counter() - begin}
            report['cases'][case] = result
            print(f'{case}: baseline overlap {result["common_with_old_baseline"]}/64', flush=True)
            if isinstance(underlying, SequentialReader):
                underlying.close()
            decoder, underlying = None, None
            gc.collect()
            atomic_json(output / 'report.json', report)
        report['repeat_comparisons'] = {p: prior.compare_cases(report['cases'][p + '_a'], report['cases'][p + '_b']) for p in POLICIES}
        report['reference_comparisons'] = {case: {name: prior.compare_cases(value, reference['cases'][name])
            for name in ('cold_a', 'baseline_history')} for case, value in report['cases'].items()}
        report['cross_policy_comparisons'] = {a + '__' + b: prior.compare_cases(report['cases'][a + '_a'], report['cases'][b + '_a'])
            for a, b in combinations(POLICIES, 2)}
        report['complete'] = True
    except InterruptedError:
        report['error_type'], status = 'InterruptedError', 75
    except BaseException as error:
        report['error_type'] = type(error).__name__
        raise
    finally:
        if isinstance(underlying, SequentialReader):
            underlying.close()
        decoder, underlying = None, None
        gc.collect()
        if network is not None:
            network.clear_cache()
        video_path.unlink(missing_ok=True)
        report['snapshot_unchanged'] = prior.snapshot_unchanged(context['snapshot'], full_run)
        report['reference_unchanged'] = file_hash(reference_path) == context['reference_sha256']
        if not report['snapshot_unchanged'] or not report['reference_unchanged']:
            report['complete'], status = False, 1
        report['wall_seconds'] = time.monotonic() - started
        atomic_json(output / 'report.json', report)
        for number, handler in old_handlers.items():
            signal.signal(number, handler)
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--mode', choices=['decode'], default='decode')
    for name in ('baseline', 'source-cache', 'development-run', 'full-run', 'reference-report'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('config', 'annotations', 'output', 'ram-root', 'index'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--max-new-questions', type=int, default=0)
    p.add_argument('--max-seconds', type=int, default=13800)
    args = p.parse_args()
    if args.max_seconds <= 0 or args.max_new_questions != 0:
        p.error('Fixed diagnostic scope and positive time required')
    if args.preflight:
        data = validate_inputs(args.source_cache, args.baseline, args.development_run, args.full_run, args.reference_report)
        print(json.dumps({'preflight': 'passed', 'cases': list(CASES), 'reference_sha256': data['reference_sha256'], 'snapshot': data['snapshot']}))
        return 0
    if any(getattr(args, n) is None for n in ('config', 'annotations', 'output', 'ram_root', 'index')):
        p.error('Missing worker arguments')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
