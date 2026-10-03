import ast
import heapq
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import warnings

import numpy as np

from compare_selectors import make_task, parse_args, run
from frame_select import parse_arguments, select_frames
from watershed_selector import WatershedConfig, peak_markers, select_leaves, smooth_curve, watershed_labels

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('upstream', ROOT / 'tests/fixtures/upstream_frame_select.py')
upstream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)


def golden(scores, ids, k, ratio=1, t1=0.8, t2=-100, depth=5):
    # Execute the frozen upstream judge and exact terminal selection expressions.
    nums = len(scores) // ratio
    values = [scores[i * ratio] for i in range(nums)]
    ids = [ids[i * ratio] for i in range(nums)]
    if nums < k:
        return ids
    normalized = (values - np.min(values)) / (np.max(values) - np.min(values))
    leaves, frames = upstream.meanstd(nums, [dict(score=normalized, depth=0)], k, [ids], t1, t2, depth)
    selected = []
    for leaf, frame_ids in zip(leaves, frames):
        quota = int(k / 2**leaf['depth'])
        selected.extend(frame_ids[i] for i in heapq.nlargest(quota, range(len(leaf['score'])), leaf['score'].__getitem__))
    return sorted(selected)


class BaselineTests(unittest.TestCase):
    def test_defaults_and_float_thresholds(self):
        args = parse_arguments([])
        self.assertEqual(args.selector, 'original')
        self.assertEqual(args.t1, 0.8)
        self.assertEqual(parse_arguments(['--t1', '0.35', '--t2', '-0.1']).t1, 0.35)

    def test_exact_upstream_on_nonflat_curves(self):
        rng = np.random.default_rng(42)
        for n, k, ratio, depth, t1 in ((128, 64, 1, 5, 0.8), (65, 9, 1, 3, 0.2),
                                      (81, 8, 2, 4, 0.35), (20, 64, 1, 5, 0.8),
                                      (100, 5, 1, 2, -0.1), (256, 16, 1, 4, 0.15)):
            for _ in range(10):
                scores = rng.random(n).tolist()
                frames = (np.arange(n) * 29 + 5).tolist()
                actual, _ = select_frames(scores, frames, k, ratio=ratio, all_depth=depth, t1=t1)
                self.assertEqual(actual, golden(scores, frames, k, ratio=ratio, depth=depth, t1=t1))

    def test_tied_values_match_upstream(self):
        scores = [0, 1, 1, 0, 2, 2, 0, 1] * 4
        self.assertEqual(select_frames(scores, list(range(32)), 4, all_depth=0)[0], golden(scores, list(range(32)), 4, depth=0))

    def test_flat_path_matches_upstream_without_nan_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            expected = golden([7.] * 64, list(range(64)), 32)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            actual, _ = select_frames([7.] * 64, list(range(64)), 32)
        self.assertEqual(actual, expected)

    def test_odd_budget_rounding_is_not_reallocated(self):
        scores = [0, 1] * 32
        baseline, a = select_frames(scores, list(range(64)), 7, all_depth=2)
        variant, b = select_frames(scores, list(range(64)), 7, all_depth=2, selector='watershed')
        self.assertEqual(len(baseline), 4)
        self.assertEqual(len(variant), 4)
        self.assertEqual(a['leaf_plan'], b['leaf_plan'])

    def test_zero_quota_leaves_are_preserved(self):
        for selector in ('original', 'watershed'):
            out, info = select_frames([0, 1] * 16, list(range(32)), 1, all_depth=5, selector=selector)
            self.assertEqual(out, [])
            self.assertEqual(info['allocated_capacity'], 0)

    def test_deep_short_leaves_do_not_split_empty_arrays(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            for selector in ('original', 'watershed'):
                out, _ = select_frames([1., 0.], [0, 30], 2, all_depth=30, selector=selector)
                self.assertEqual(out, [])  # preserve singleton stopping depth and zero quota

    def test_affine_score_transform_preserves_plan(self):
        scores = np.random.default_rng(17).random(128)
        first = select_frames(scores, list(range(128)), 16, selector='watershed')
        second = select_frames(3.2 * scores - 1.5, list(range(128)), 16, selector='watershed')
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1]['leaf_plan'], second[1]['leaf_plan'])


class WatershedTests(unittest.TestCase):
    def select(self, curve, k, **options):
        return select_frames(curve, list(range(len(curve))), k, all_depth=0, selector='watershed', watershed_config=WatershedConfig(sigma=0, **options))

    def test_distinct_moments_replace_neighboring_topk(self):
        curve = [0, .95, 1, .96, 0, 0, .8, .85, .8, 0]
        baseline, _ = select_frames(curve, list(range(10)), 2, all_depth=0)
        selected, _ = self.select(curve, 2, min_distance=3)
        self.assertEqual(baseline, [2, 3])
        self.assertEqual(selected, [2, 7])

    def test_excess_basins_rank_relevance_and_prominence(self):
        curve = [.9, 1., .9, 0, .95, 0, .6, 0]
        selected, info = self.select(curve, 1, min_distance=1, prominence_weight=2)
        self.assertEqual(selected, [4])  # endpoint-side floor makes peak 1 shallow
        self.assertEqual(len(info['leaves'][0]['basins']), 3)
        self.assertEqual(self.select(curve, 1, min_distance=1, prominence_weight=0)[0], [1])
        selected, _ = self.select([0, .9, .8, .91, .8, 0], 1, min_distance=1, prominence_weight=2)
        self.assertEqual(selected, [3])

    def test_utility_can_favor_a_prominent_peak_over_a_higher_shallow_one(self):
        # Endpoint is shallow due to a higher internal peak; a lower isolated
        # event outranks it after adding prominence to relevance.
        curve = [.95, .94, 1, 0, .90, 0]
        selected, _ = self.select(curve, 2, min_distance=1, prominence_weight=1)
        self.assertEqual(selected, [2, 4])

    def test_marker_flood_partitions_whole_curve(self):
        curve = np.array([0, 1, .5, 0, .3, .8, .2])
        markers = peak_markers(curve, 0)
        labels = watershed_labels(curve, markers)
        self.assertEqual([p[0] for p in markers], [1, 5])
        self.assertTrue(np.all(labels >= 0))
        self.assertEqual(labels[1], 0)
        self.assertEqual(labels[5], 1)
        self.assertTrue(all(len(np.flatnonzero(labels == i)) == np.ptp(np.flatnonzero(labels == i)) + 1 for i in range(2)))

    def test_representative_uses_unsmoothed_relevance(self):
        curve = [0, .8, 1., .8, 0, 0, .99, 0, 0]
        result, info = select_frames(curve, list(range(9)), 2, all_depth=0, selector='watershed',
                                     watershed_config=WatershedConfig(sigma=1, min_distance=1, min_prominence=0))
        for basin in info['leaves'][0]['basins']:
            lo, hi = basin['start'], basin['end']
            self.assertEqual(curve[basin['representative']], max(curve[lo:hi + 1]))
        self.assertEqual(len(result), 2)

    def test_flat_curve_refills_with_temporal_spread(self):
        selected, info = self.select([1.] * 20, 4, min_distance=3)
        self.assertEqual(len(selected), 4)
        self.assertEqual(info['leaves'][0]['basins'], [])
        self.assertTrue(all(b - a >= 3 for a, b in zip(selected, selected[1:])))
        self.assertEqual(info['distance_relaxations'], 0)

    def test_plateau_has_one_marker_and_stable_center(self):
        curve = np.array([0, 1, 1, 1, 0], dtype=float)
        self.assertEqual(peak_markers(curve, .05), [(2, 1.)])
        self.assertEqual(self.select(curve, 1)[0], [2])

    def test_flat_peak_at_an_endpoint(self):
        self.assertEqual(peak_markers(np.array([1, 1, .5, 0]), 0), [(0, 1.)])

    def test_monotonic_curves_and_endpoints(self):
        self.assertEqual(self.select([0, .2, .4, .6, 1], 1)[0], [4])
        self.assertEqual(self.select([1, .6, .4, .2, 0], 1)[0], [0])
        self.assertEqual(self.select([1, .5, 0, .4, .9], 2, min_distance=3)[0], [0, 4])

    def test_no_sufficiently_prominent_peak(self):
        selected, info = self.select([0, 1, 0, .9, 0, .8, 0], 3, min_prominence=2, min_distance=2)
        self.assertEqual(info['leaves'][0]['basins'], [])
        self.assertEqual(len(selected), 3)
        self.assertEqual(info['selections'][0]['position'], 1)

    def test_small_false_peak_is_filtered(self):
        peaks = peak_markers(np.array([0, .8, .79, .81, .4, .7, 0]), .05)
        self.assertEqual([i for i, _ in peaks], [3, 5])

    def test_smoothing_suppresses_small_oscillations(self):
        curve = np.array([0, .4, .6, .59, .61, .6, .5, 0])
        self.assertLess(len(peak_markers(smooth_curve(curve, 1), .01)), len(peak_markers(curve, .01)))

    def test_large_finite_sigma_is_bounded_by_signal_support(self):
        self.assertTrue(np.all(np.isfinite(smooth_curve(np.array([0, 1, 0]), 1e308))))

    def test_close_peaks_are_suppressed(self):
        selected, info = self.select([0, 1, 0, .9, 0, 0, 0, .8, 0], 2, min_distance=4)
        self.assertEqual(selected, [1, 7])
        self.assertEqual(info['distance_relaxations'], 0)

    def test_gap_is_enforced_across_leaf_boundaries(self):
        leaves = [dict(scores=np.array([0, 0, 1]), positions=[0, 1, 2], quota=1),
                  dict(scores=np.array([.9, 0, .5]), positions=[3, 4, 5], quota=1)]
        selected, info = select_leaves(leaves, WatershedConfig(sigma=0, min_distance=3))
        self.assertEqual(selected, [2, 5])
        self.assertEqual([s['leaf'] for s in info['selections']], [0, 1])

    def test_refill_preserves_accepted_peak(self):
        selected, info = self.select([0, 0, 1, .5, 0, 0, 0, 0], 4, min_distance=2)
        self.assertIn(2, selected)
        self.assertEqual(info['selections'][0]['reason'], 'peak')
        self.assertEqual(len(selected), 4)

    def test_impossible_distance_relaxes_and_still_fills(self):
        selected, info = self.select([0, 1, 0, .9], 4, min_distance=50)
        self.assertEqual(selected, [0, 1, 2, 3])
        self.assertEqual(info['distance_relaxations'], 3)
        self.assertEqual(len({i['position'] for i in info['selections']}), 4)

    def test_random_invariants_and_repeatability(self):
        rng = np.random.default_rng(91)
        for _ in range(150):
            n = int(rng.integers(1, 130))
            k = int(rng.integers(0, 80))
            values = rng.choice([0., .25, .5, .75, 1.], n)
            ids = (np.arange(n) * 17).tolist()
            depth = int(rng.integers(0, 8))
            ratio = int(rng.integers(1, 4))
            a, pa = select_frames(values, ids, k, ratio=ratio, all_depth=depth)
            b, pb = select_frames(values, ids, k, ratio=ratio, all_depth=depth, selector='watershed')
            c, pc = select_frames(values, ids, k, ratio=ratio, all_depth=depth, selector='watershed')
            self.assertEqual(pa['leaf_plan'], pb['leaf_plan'])
            self.assertEqual(len(a), len(b))
            self.assertLessEqual(len(b), k)
            self.assertEqual(b, sorted(set(b)))
            self.assertTrue(set(b).issubset(ids))
            self.assertEqual((b, pb), (c, pc))


class InputAndIntegrationTests(unittest.TestCase):
    def test_empty_singleton_zero_budget_and_short_video(self):
        for selector in ('original', 'watershed'):
            for scores, ids, k, expected in (([], [], 4, []), ([1], [100], 2, [100]),
                                            ([1], [100], 1, []), ([1], [100], 0, [])):
                self.assertEqual(select_frames(scores, ids, k, selector=selector)[0], expected)
            self.assertEqual(select_frames([1], [100], 1, all_depth=0, selector=selector)[0], [100])

    def test_ratio_matches_upstream_floor_sampling(self):
        self.assertEqual(select_frames([0, 1, 2, 3, 4], [0, 10, 20, 30, 40], 8, ratio=2)[0], [0, 20])
        self.assertEqual(select_frames([1], [0], 8, ratio=2)[0], [])

    def test_official_cache_integer_valued_float_ids_are_accepted(self):
        out, _ = select_frames([0, 1, 0], [0., 29., 58.], 1, all_depth=0)
        self.assertEqual(out, [29])
        self.assertIsInstance(out[0], int)

    def test_flat_negative_threshold_still_matches_upstream_decisions(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            expected = golden([1.] * 64, list(range(64)), 8, t1=-.1)
        self.assertEqual(select_frames([1.] * 64, list(range(64)), 8, t1=-.1)[0], expected)

    def test_nan_inf_misalignment_and_frame_ids_fail_explicitly(self):
        for scores, ids in (([np.nan], [0]), ([np.inf], [0]), ([1, 2], [0]),
                            ([1, 2], [0, 0]), ([1, 2], [2, 1]), ([1], [-1]),
                            ([1], [1.5]), ([[1]], [0])):
            with self.assertRaises(ValueError):
                select_frames(scores, ids)

    def test_invalid_parameters(self):
        for kw in ({'max_num_frames': -1}, {'ratio': 0}, {'all_depth': -1}, {'selector': 'bad'}, {'t1': np.nan}):
            with self.assertRaises(ValueError):
                select_frames([1], [0], **kw)
        for kw in ({'sigma': -1}, {'sigma': np.inf}, {'min_prominence': np.nan}, {'min_distance': 0}, {'prominence_weight': -1}):
            with self.assertRaises(ValueError):
                WatershedConfig(**kw)

    def test_cli_default_matches_actual_upstream_main(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scores = [[0, .5, 1, .4, 0, .9, .3, 0] * 8, [1, .5]]
            frames = [list(range(64)), [5, 25]]
            for filename, rows in (('scores.json', scores), ('frames.json', frames)):
                (root / filename).write_text(json.dumps(rows))
            original_dir = root / 'upstream'
            original_dir.mkdir()
            args = SimpleNamespace(score_path=str(root / 'scores.json'), frame_path=str(root / 'frames.json'),
                                    output_file=str(original_dir), max_num_frames=8, ratio=1, t1=.8,
                                    t2=-100, all_depth=2, dataset_name='videomme', extract_feature_model='blip')
            upstream.main(args)
            subprocess.run([sys.executable, str(ROOT / 'frame_select.py'), '--score_path', args.score_path,
                            '--frame_path', args.frame_path, '--max_num_frames', '8', '--all_depth', '2',
                            '--dataset_name', 'videomme', '--output_file', str(root / 'new')], check=True)
            self.assertEqual((original_dir / 'videomme/blip/selected_frames.json').read_bytes(),
                             (root / 'new/videomme/blip/selected_frames.json').read_bytes())

    def test_task_builder_preserves_upstream_prompt_and_separates_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for dataset in ('videomme', 'longvideobench'):
                name = make_task(root / dataset / 'lmms_task', dataset, 'watershed',
                                  [({'question': 'Q'}, [10, 30])], root / 'videos', 8)
                self.assertEqual(name, f'aks_watershed_{dataset}')
                config = next((root / dataset / 'lmms_task').glob('*.yaml')).read_text()
                self.assertIn('dataset_path: json', config)
                self.assertIn('temperature: 0', config)
                self.assertIn('generation_kwargs:', config)
                self.assertIn('process_results:', config)
                rows = json.loads((root / dataset / 'annotations.json').read_text())
                self.assertEqual(rows[0]['frame_idx'], [10, 30])
                self.assertTrue(rows[0]['use_topk'])

    def test_paired_runner_manifest_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'scores.json').write_text(json.dumps([[0, 1, .9, 0, .8, 0, 0, 0]]))
            (root / 'frames.json').write_text(json.dumps([list(range(8))]))
            args = parse_args(['--score_path', str(root / 'scores.json'), '--frame_path', str(root / 'frames.json'),
                               '--output_dir', str(root / 'out'), '--max_num_frames', '2', '--all_depth', '0'])
            summary = run(args)
            self.assertEqual(summary['original']['query_count'], 1)
            manifest = json.loads((root / 'out/manifest.json').read_text())
            self.assertEqual(len(manifest['inputs'][0]['sha256']), 64)
            self.assertEqual(len(manifest['code']), 4)
            with self.assertRaises(ValueError):
                run(args)

    def test_runner_rejects_unaligned_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, rows in (('scores', [[1]]), ('frames', [[0]]), ('annotations', [])):
                (root / f'{name}.json').write_text(json.dumps(rows))
            args = parse_args(['--score_path', str(root / 'scores.json'), '--frame_path', str(root / 'frames.json'),
                               '--output_dir', str(root / 'out'), '--annotation_path', str(root / 'annotations.json'),
                               '--video_root', str(root), '--limit', '1'])
            with self.assertRaises(ValueError):
                run(args)


class EvaluationLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the real method without importing CUDA, llava or lmms_eval.
        tree = ast.parse((ROOT / 'evaluation/llava_vid.py').read_text(encoding='utf-8'))
        model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'LlavaVid')
        method = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == 'load_video_index')
        class Reader:
            def __init__(self, *args, **kwargs):
                pass
            def __len__(self):
                return 120
            def get_avg_fps(self):
                return 29.97
            def get_batch(self, ids):
                return SimpleNamespace(asnumpy=lambda: np.asarray(ids))
        namespace = {'np': np, 'VideoReader': Reader, 'cpu': lambda _: None}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<loader>', 'exec'), namespace)
        cls.loader = staticmethod(namespace['load_video_index'])

    def test_strict_loader_consumes_exact_shortfall_ids_and_real_timestamps(self):
        frames, times, _ = self.loader(SimpleNamespace(strict_selected_frames=True), 'test.mp4', 8, 1, {'frame_idx': [0, 60]})
        self.assertEqual(frames.tolist(), [0, 60])
        self.assertEqual(times, '0.00s,2.00s')

    def test_default_loader_still_falls_back_to_uniform(self):
        frames, _, _ = self.loader(SimpleNamespace(strict_selected_frames=False), 'test.mp4', 8, 1, {'frame_idx': [0, 60]})
        self.assertEqual(len(frames), 8)
        self.assertEqual(frames.tolist(), np.linspace(0, 119, 8, dtype=int).tolist())

    def test_strict_loader_rejects_invalid_ids_and_forced_sampling(self):
        model = SimpleNamespace(strict_selected_frames=True)
        for ids in ([], [0] * 2, [-1], [120], [1.5], list(range(9)), [10, 0]):
            with self.assertRaises(ValueError):
                self.loader(model, 'test.mp4', 8, 1, {'frame_idx': ids})
        with self.assertRaises(ValueError):
            self.loader(model, 'test.mp4', 8, 1, {'frame_idx': [1]}, force_sample=True)


if __name__ == '__main__':
    unittest.main()
