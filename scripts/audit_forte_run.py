"""Read-only audit of a partial or complete FORTE export (stdlib, no weights)."""
import argparse
import json
import math
from pathlib import Path


def audit(directory):
    contract = json.loads((directory / 'contract.json').read_text(encoding='utf-8'))
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in (directory / 'predictions.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if contract['config']['method'] != 'forte' or not rows:
        raise ValueError('Expected nonempty FORTE exports')
    if len(rows) != summary['completed_count'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Export count/ID mismatch')
    for row in rows:
        budget, scored = row['forte_scoring_budget'], row['forte_scored_indices']
        n, fps = row['forte_source_frames'], row['forte_source_fps']
        stride = max(1, math.floor(fps / contract['config']['forte_rho']))
        if budget != max(1, n // stride):
            raise ValueError(f'Reference budget mismatch: {row["id"]}')
        acquired = row['forte_initial_indices'] + [i for round_ in row['forte_rounds'] for i in round_['acquired_indices']]
        chosen = row['calls'][0]['frame_indices']
        if (len(scored) != budget or len(set(acquired)) != budget or sorted(acquired) != scored
                or len(row['forte_raw_scores']) != budget
                or row['scorer_unique_frames'] != budget or row['scorer_frame_evaluations'] != budget
                or chosen != sorted(set(chosen)) or len(chosen) != row['final_frames']):
            raise ValueError(f'Budget/frame accounting failed: {row["id"]}')
        expected_k = contract['config']['frames']
        if len(chosen) != expected_k and not row['forte_source_shorter_than_k']:
            raise ValueError(f'Incomplete final frame budget: {row["id"]}')
        if row['forte_supplemental_frames'] != len(set(chosen) - set(scored)):
            raise ValueError(f'Supplement accounting failed: {row["id"]}')
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in row['forte_raw_scores']):
            raise ValueError(f'Invalid measured scores: {row["id"]}')
        timestamps = row['calls'][0]['timestamps']
        if timestamps != sorted(set(timestamps)) or len(timestamps) != len(chosen):
            raise ValueError(f'Invalid chronological evidence: {row["id"]}')
        if (any(i < 0 or i >= n for i in chosen)
                or any(not math.isclose(t, i / fps, rel_tol=0., abs_tol=1e-9) for i, t in zip(chosen, timestamps))):
            raise ValueError(f'Original source timestamp mismatch: {row["id"]}')
    return {'audit': 'PASS', 'completed': len(rows), 'selected': contract['selected_count'],
            'complete': summary['complete'], 'source_sha256': contract['source_sha256'],
            'accuracy_completed': summary['accuracy_completed'],
            'mean_scorer_frame_evaluations': summary['mean_scorer_frame_evaluations'],
            'mean_final_frames': summary['mean_final_frames'],
            'latency_mean_seconds': summary['latency_mean_seconds'],
            'mean_selection_seconds': summary['mean_selection_seconds']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_dir), indent=2, allow_nan=False))
