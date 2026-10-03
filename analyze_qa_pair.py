"""Regrade paired predictions and report coverage, accuracy and regressions."""
import argparse
from collections import Counter, defaultdict
import csv
import math
from pathlib import Path
import random

from qa_protocol import SELECTORS, atomic_json, identity, load_pair, parse_answer, read


def summarize(records):
    n = len(records)
    accuracies = {s: sum(r[s + '_correct'] for r in records) / n if n else None for s in SELECTORS}
    return {'count': n, 'accuracy': accuracies,
            'delta_percentage_points': 100 * (accuracies['watershed'] - accuracies['original']) if n else None,
            'outcomes': dict(Counter(r['outcome'] for r in records)),
            'invalid_answers': {s: sum(r[s + '_answer'] is None for r in records) for s in SELECTORS}}


def bootstrap(records, repetitions=2000, seed=42):
    """Paired cluster bootstrap: sample videos, retain all their questions."""
    clusters = defaultdict(list)
    for r in records:
        clusters[r['video_id']].append(int(r['watershed_correct']) - int(r['original_correct']))
    if len(clusters) < 2:
        return None
    stats = [(sum(v), len(v)) for v in clusters.values()]
    rng = random.Random(seed)
    values = []
    for _ in range(repetitions):
        samples = rng.choices(stats, k=len(stats))
        values.append(100 * sum(v[0] for v in samples) / sum(v[1] for v in samples))
    values.sort()
    return [values[int(.025 * (repetitions - 1))], values[int(.975 * (repetitions - 1))]]


def analyze(experiment):
    root = Path(experiment)
    manifest, annotations = load_pair(root)
    qa = root / 'qa_pair'
    config = read(qa / 'config.json')
    from qa_protocol import digest
    if config['selection_manifest_sha256'] != digest(root / 'manifest.json') or any(
            config['annotation_hashes'][s] != digest(root / s / 'annotations.json') for s in SELECTORS):
        raise ValueError('QA inputs do not match this selection experiment')
    dataset = manifest['arguments']['dataset_name']
    records, seen = [], set()
    times = {s: defaultdict(float) for s in SELECTORS}
    max_memory = {s: 0 for s in SELECTORS}
    for path in sorted((qa / 'predictions').glob('*.json')):
        pair = read(path)
        index = pair['index']
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < manifest['query_count'] or index in seen:
            raise ValueError(f'Invalid or duplicate prediction index: {path}')
        seen.add(index)
        row = annotations['original'][index]
        qid, vid, gold, choices = identity(row, dataset)
        if (pair['question_id'], pair['video_id'], pair['gold']) != (qid, vid, gold):
            raise ValueError(f'Prediction identity/label mismatch: {path}')
        record = {'index': index, 'question_id': qid, 'video_id': vid, 'question': row['question'], 'gold': gold,
                  'task_type': row.get('task_type', row.get('question_category', 'unknown')),
                  'duration_group': row.get('duration', 'unknown') if dataset == 'videomme' else 'all'}
        for s in SELECTORS:
            result = pair[s]
            if result['status'] != 'ok' or result['frame_ids'] != annotations[s][index]['frame_idx']:
                raise ValueError(f'Invalid prediction/frame IDs: {path}, {s}')
            answer = parse_answer(result['response'], len(choices))
            record.update({s + '_answer': answer, s + '_correct': answer == gold,
                           s + '_response': result['response'], s + '_frame_count': len(result['frame_ids'])})
            for metric in ('decode_seconds', 'preprocessing_seconds', 'generation_seconds'):
                value = result[metric]
                if not math.isfinite(value) or value < 0:
                    raise ValueError('Invalid timing')
                times[s][metric] += value
            max_memory[s] = max(max_memory[s], result['peak_gpu_memory_bytes'])
        a, b = record['original_correct'], record['watershed_correct']
        record['outcome'] = 'both_correct' if a and b else 'regressed' if a else 'improved' if b else 'both_wrong'
        records.append(record)
    report = summarize(records)
    report.update(expected_questions=manifest['query_count'], completed_questions=len(records),
                  complete=len(records) == manifest['query_count'],
                  coverage=len(records) / manifest['query_count'],
                  full_benchmark_selection=manifest['arguments'].get('limit') is None,
                  bootstrap_delta_95ci_percentage_points=bootstrap(records),
                  timing_totals_seconds={s: dict(times[s]) for s in SELECTORS}, peak_gpu_memory_bytes=max_memory,
                  selection_summary=read(root / 'summary.json'), model=config['model'], model_revision=config['model_revision'],
                  notes=['Accuracy denominator is completed pairs only; missing pairs are reported as incomplete.',
                         'Invalid/ambiguous MCQ output counts as wrong; no random fallback.',
                         'Video-cluster bootstrap; CI on tiny smoke subsets is not a benchmark conclusion.',
                         'Inference timing includes first-pair warmup, excludes model load/download; selectors run in alternating order.'])
    groups = {}
    for key in ('duration_group', 'task_type'):
        grouped = defaultdict(list)
        for record in records:
            grouped[str(record[key])].append(record)
        groups[key] = {k: summarize(v) for k, v in grouped.items()}
    report['groups'] = groups
    out = qa / 'report'
    atomic_json(out / 'comparison.json', report)
    atomic_json(out / 'improved.json', [r for r in records if r['outcome'] == 'improved'])
    atomic_json(out / 'regressed.json', [r for r in records if r['outcome'] == 'regressed'])
    with (out / 'per_question.csv').open('w', encoding='utf-8', newline='') as stream:
        fields = list(records[0]) if records else ['question_id', 'outcome']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    fmt = lambda x: 'N/A' if x is None else f'{100*x:.2f}%'
    (out / 'comparison.md').write_text(
        f"# Paired QA report\n\nCompleted: {len(records)}/{manifest['query_count']}; "
        f"{'complete selection run' if report['complete'] else 'INCOMPLETE'}; "
        f"{'full input' if report['full_benchmark_selection'] else 'subset only'}.\n\n"
        f"Model: {config['model']} @ {config['model_revision']}\n\n"
        f"| Selector | Accuracy |\n|---|---:|\n| Original | {fmt(report['accuracy']['original'])} |\n"
        f"| Watershed | {fmt(report['accuracy']['watershed'])} |\n\n"
        f"Delta (percentage points): {report['delta_percentage_points']}\n\n"
        f"Improved: {report['outcomes'].get('improved', 0)}; regressed: {report['outcomes'].get('regressed', 0)}.\n\n"
        + '\n'.join('- ' + note for note in report['notes']) + '\n', encoding='utf-8')
    print((out / 'comparison.md').read_text(encoding='utf-8'))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    analyze(parser.parse_args().experiment)
