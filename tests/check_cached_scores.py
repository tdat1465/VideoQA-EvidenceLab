"""Verify the baseline against a frozen upstream function on cached real scores.

Run from repository root:
python tests/check_cached_scores.py --score_root ../AKS/outscores
"""
import argparse
import json
from pathlib import Path
import sys
import time
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from frame_select import select_frames
from test_selection import golden


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--score_root', required=True)
    parser.add_argument('--budgets', nargs='+', type=int, default=[8, 16, 32, 64])
    parser.add_argument('--limit', type=int, default=0, help='0 = all queries')
    parser.add_argument('--report', help='save validation result as JSON')
    args = parser.parse_args()
    reports = []
    start = time.perf_counter()
    for path in sorted(Path(args.score_root).glob('*/*/scores.json')):
        scores = json.loads(path.read_text())
        frames = json.loads(path.with_name('frames.json').read_text())
        if len(scores) != len(frames):
            raise RuntimeError(f'misaligned data: {path}')
        if args.limit:
            scores, frames = scores[:args.limit], frames[:args.limit]
        for budget in args.budgets:
            for index, (curve, ids) in enumerate(zip(scores, frames)):
                result, _ = select_frames(curve, ids, budget)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    expected = golden(curve, ids, budget)
                if result != expected:
                    raise AssertionError(f'baseline differs: {path}, K={budget}, query={index}')
            record = {'dataset': path.parent.parent.name, 'scorer': path.parent.name,
                      'budget': budget, 'queries': len(scores), 'baseline_mismatches': 0}
            print(json.dumps(record), flush=True)
            reports.append(record)
    if not reports:
        raise ValueError('no */*/scores.json files found')
    report = {'runs': reports, 'elapsed_seconds': time.perf_counter() - start,
              'note': 'Frame-index regression only, not model QA evaluation.'}
    if args.report:
        output = Path(args.report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
