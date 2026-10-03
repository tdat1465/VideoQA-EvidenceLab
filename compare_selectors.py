"""Paired CPU selection on exactly the same cached AKS scores and frame IDs.

Optionally builds independent lmms-eval tasks from upstream templates, without
overwriting the source annotation files. Selection statistics are not QA accuracy.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np

from frame_select import select_frames
from watershed_selector import WatershedConfig

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(Path(path).resolve()), 'sha256': digest.hexdigest()}


def make_task(directory, dataset, selector, annotations, video_root, budget):
    """Keep upstream prompt/grader, use a unique task and JSON annotation path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'datasets' / dataset
    basename = 'videomme.yaml' if dataset == 'videomme' else 'longvideobench_val_v.yaml'
    template = (source / basename).read_text(encoding='utf-8')
    task_name = f'aks_{selector}_{dataset}'
    split = 'test' if dataset == 'videomme' else 'validation'
    annotation_path = directory.parent / 'annotations.json'
    rows = [dict(row, frame_idx=frame_ids, frame_num=budget, use_topk=True)
            for row, frame_ids in annotations]
    if any(not row['frame_idx'] for row in rows):
        raise ValueError('lmms-eval tasks require nonempty frame sets; check K and all_depth')
    write_json(annotation_path, rows)
    template = re.sub(r'^dataset_path:.*$', 'dataset_path: json', template, flags=re.M)
    # HF JSON dataset, with the exact upstream split and grader; no custom
    # dataset builder or silent access to another selector's mutable annotations.
    dataset_kwargs = (f'dataset_kwargs:\n  data_files: {{{json.dumps(split)}: '
                      f'{json.dumps(str(annotation_path.resolve()))}}}\n'
                      f'  cache_dir: {json.dumps(Path(video_root).name)}\n')
    template = re.sub(r'dataset_kwargs:\n.*?(?=^task:)', lambda _: dataset_kwargs, template, flags=re.S | re.M)
    template = re.sub(r'^task:.*$', f'task: {task_name}', template, flags=re.M)
    (directory / basename).write_text(template, encoding='utf-8')
    utils = (source / 'utils.py').read_text(encoding='utf-8')
    utils = utils.replace("base_cache_dir = './datasets/'", f'base_cache_dir = {str(Path(video_root).resolve().parent)!r}')
    (directory / 'utils.py').write_text(utils, encoding='utf-8')
    return task_name


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--score_path', required=True)
    parser.add_argument('--frame_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--dataset_name', choices=('longvideobench', 'videomme'), default='videomme')
    parser.add_argument('--max_num_frames', type=int, default=64)
    parser.add_argument('--ratio', type=int, default=1)
    parser.add_argument('--t1', type=float, default=0.8)
    parser.add_argument('--t2', type=float, default=-100)
    parser.add_argument('--all_depth', type=int, default=5)
    parser.add_argument('--watershed_sigma', type=float, default=1.0)
    parser.add_argument('--watershed_min_prominence', type=float, default=0.05)
    parser.add_argument('--watershed_min_distance', type=int, default=3)
    parser.add_argument('--watershed_prominence_weight', type=float, default=0.25)
    parser.add_argument('--limit', type=int, help='first N queries, equally for both selectors')
    parser.add_argument('--annotation_path', help='same ordered labels used to extract these AKS scores')
    parser.add_argument('--video_root', help='benchmark root containing data/ or videos/; creates lmms tasks')
    return parser.parse_args(argv)


def run(args):
    scores, frames = read_json(args.score_path), read_json(args.frame_path)
    if not isinstance(scores, list) or not isinstance(frames, list) or len(scores) != len(frames):
        raise ValueError('score/frame JSON query counts must agree')
    if not scores:
        raise ValueError('comparison requires at least one query')
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be positive')
    if bool(args.annotation_path) != bool(args.video_root):
        raise ValueError('--annotation_path and --video_root must be supplied together')
    annotations = read_json(args.annotation_path) if args.annotation_path else None
    if annotations is not None and (not isinstance(annotations, list) or len(annotations) != len(scores)):
        raise ValueError('annotations must align with the full score list before --limit')
    count = min(len(scores), args.limit) if args.limit else len(scores)
    config = WatershedConfig(args.watershed_sigma, args.watershed_min_prominence,
                             args.watershed_min_distance, args.watershed_prominence_weight)
    output = Path(args.output_dir).resolve()
    # A previous run's checkpoints must never get mistaken for this experiment.
    if output.exists() and any(output.iterdir()):
        raise ValueError('output_dir must be empty or new; use a fresh experiment directory')
    outputs, diagnostics, runtimes = {}, {}, {}
    for selector in ('original', 'watershed'):
        selected, details = [], []
        start = time.perf_counter()
        for i in range(count):
            try:
                chosen, info = select_frames(scores[i], frames[i], args.max_num_frames, args.ratio,
                                              args.t1, args.t2, args.all_depth, selector, config)
            except ValueError as error:
                raise ValueError(f'query {i}: {error}') from error
            selected.append(chosen)
            details.append(info)
        runtimes[selector] = time.perf_counter() - start
        outputs[selector], diagnostics[selector] = selected, details
    if any(a['leaf_plan'] != b['leaf_plan'] or a['selected_count'] != b['selected_count']
           for a, b in zip(diagnostics['original'], diagnostics['watershed'])):
        raise RuntimeError('paired selectors changed the AKS allocation or actual frame count')

    summary = {}
    for selector in outputs:
        selected, details = outputs[selector], diagnostics[selector]
        normalized_scores, gaps, coverage = [], [], []
        for curve, frame_ids, info in zip(scores[:count], selected, details):
            sampled = np.asarray(curve, dtype=float)[np.arange(len(curve) // args.ratio) * args.ratio]
            if len(sampled) and frame_ids:
                scale = np.ptp(sampled)
                normalized = (sampled - sampled.min()) / scale if scale else np.zeros(len(sampled))
                positions = info.get('selected_positions', list(range(len(frame_ids))))
                normalized_scores.extend(normalized[positions].tolist())
                gaps.extend(np.diff(positions).tolist())
                coverage.append((max(positions) - min(positions)) / max(1, len(sampled) - 1))
        counts = Counter(len(indices) for indices in selected)
        summary[selector] = {
            'selection_seconds': runtimes[selector], 'query_count': count,
            'selected_count_histogram': {str(k): v for k, v in sorted(counts.items())},
            'queries_below_K': sum(len(ids) < args.max_num_frames for ids in selected),
            'mean_normalized_selected_relevance': float(np.mean(normalized_scores)) if normalized_scores else None,
            'mean_temporal_span_fraction': float(np.mean(coverage)) if coverage else None,
            'gap_units': 'candidate ticks after ratio; not seconds or decoded frame IDs',
            'adjacent_candidate_pairs': sum(gap == 1 for gap in gaps),
            'pairs_closer_than_requested_distance': sum(gap < config.min_distance for gap in gaps),
            'distance_relaxations': sum(info['distance_relaxations'] for info in details),
            'peak_selected_frames': sum(item['reason'] == 'peak' for info in details for item in info.get('selections', [])),
        }
    summary['changed_queries'] = sum(a != b for a, b in zip(outputs['original'], outputs['watershed']))
    summary['note'] = 'Selection diagnostics only. No model inference or QA accuracy measured.'

    task_names = {}
    for selector in outputs:
        directory = output / selector
        write_json(directory / 'selected_frames.json', outputs[selector])
        write_json(directory / 'diagnostics.json', diagnostics[selector])
        if annotations is not None:
            task_names[selector] = make_task(directory / 'lmms_task', args.dataset_name, selector,
                                             zip(annotations[:count], outputs[selector]), args.video_root, args.max_num_frames)
    try:
        git_head = subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}',
                                            '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        git_status = subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}',
                                              '-C', str(ROOT), 'status', '--porcelain'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head, git_status = None, None
    manifest = {'arguments': vars(args), 'watershed_config': asdict(config),
                'inputs': [fingerprint(args.score_path), fingerprint(args.frame_path)],
                'query_count': count, 'git_head': git_head, 'git_status': git_status,
                'code': [fingerprint(ROOT / name) for name in ('frame_select.py', 'watershed_selector.py', 'compare_selectors.py', 'evaluation/llava_vid.py')],
                'numpy_version': np.__version__, 'tasks': task_names}
    if args.annotation_path:
        manifest['inputs'].append(fingerprint(args.annotation_path))
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == '__main__':
    run(parse_args())
