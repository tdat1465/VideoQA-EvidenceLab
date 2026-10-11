import itertools
import importlib.util
import math
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json

import numpy as np

from evidencelab.data import Question
from evidencelab.forte_adapter import (acquire_batch, coverage, exact_keyframes, gp_predict,
                                      normalize, reference_grid, select_forte, supplement)
from evidencelab.longvideo_config import LongVideoConfig


class ExactOptimizerTests(unittest.TestCase):
    def test_matches_exhaustive_subsets_including_ties_and_boundaries(self):
        rng = np.random.default_rng(42)
        cases = 0
        for n in range(1, 10):
            for trial in range(12):
                t = np.sort(rng.choice(np.arange(50), size=n, replace=False)).astype(float)
                duration = max(1., float(t[-1]))
                scores = normalize(rng.uniform(size=n)) if trial % 3 else np.zeros(n)
                for k in range(1, min(5, n) + 1):
                    for beta in (0., .01, 1., 100.):
                        chosen, value = exact_keyframes(t, scores, k, duration, beta)
                        def objective(ids):
                            return float(scores[list(ids)].sum()) + beta * coverage(t[list(ids)].tolist(), duration, k)
                        optimum = max(objective(ids) for ids in itertools.combinations(range(n), k))
                        self.assertEqual(chosen, sorted(set(chosen)))
                        self.assertEqual(len(chosen), k)
                        self.assertAlmostEqual(objective(chosen), optimum, places=9)
                        self.assertAlmostEqual(value, optimum, places=9)
                        cases += 1
        self.assertEqual(cases, 1680)

    def test_matches_quadratic_dp_on_longer_irregular_sequences(self):
        rng = np.random.default_rng(18)
        for n in (17, 65, 137):
            times = np.cumsum(rng.uniform(.001, 8., n))
            duration = float(times[-1] + 5.)
            scores = rng.random(n)
            for k in (1, 8, min(64, n)):
                tau = duration / (k + 1)
                previous = scores + np.log1p(times / tau)
                for layer in range(1, k):
                    current = np.full(n, -np.inf)
                    for i in range(layer, n):
                        current[i] = scores[i] + np.max(previous[:i] + np.log1p((times[i] - times[:i]) / tau))
                    previous = current
                optimum = float(np.max(previous + np.log1p((duration - times) / tau)))
                _, value = exact_keyframes(times, scores, k, duration)
                self.assertAlmostEqual(value, optimum, places=10)

    def test_stop_and_invalid_order(self):
        with self.assertRaises(InterruptedError):
            exact_keyframes([0., 1., 2.], [0., 1., 0.], 2, 2., stop=lambda: True)
        with self.assertRaises(ValueError):
            exact_keyframes([1., 0.], [0., 1.], 1, 2.)


class GaussianProcessTests(unittest.TestCase):
    def test_full_band_matches_dense_posterior_and_clipping(self):
        t = np.array([0., 2., 7., 13., 20., 40.])
        y = np.array([.4, .1, .9, .3, 0., 1.])
        q = np.arange(.5, 45., .7)
        predictions, details = gp_predict(t, y, q)
        kernel = np.exp(-.5 * ((t[:, None] - t[None, :]) / 10.) ** 2)
        alpha = np.linalg.solve(kernel + np.eye(len(t)) * details['epsilon'], y - y.mean())
        expected = y.mean() + np.exp(-.5 * ((q[:, None] - t[None, :]) / 10.) ** 2) @ alpha
        np.testing.assert_allclose(predictions, np.clip(expected, y.min(), y.max()), atol=1e-9)
        self.assertEqual(details['half_width'], len(t) - 1)

    def test_banded_fit_and_local_prediction_match_independent_dense_reference(self):
        t = np.arange(24.) * 20.
        y = np.sin(t / 100.) * .5 + .5
        q = np.arange(-1., 500., 3.)
        pred, details = gp_predict(t, y, q, initial_width=2, max_width=4)
        self.assertEqual(details['half_width'], 2)
        full = np.exp(-.5 * ((t[:, None] - t[None, :]) / 10.) ** 2)
        full[np.abs(np.arange(len(t))[:, None] - np.arange(len(t))[None, :]) > 2] = 0.
        alpha = np.linalg.solve(full + np.eye(len(t)) * details['epsilon'], y - y.mean())
        expected = []
        for point in q:
            pos = next((i for i, ti in enumerate(t) if ti >= point), len(t) - 1)
            ids = list(range(max(0, pos - 2), min(len(t), pos + 3)))
            expected.append(y.mean() + sum(math.exp(-.5 * ((point - t[i]) / 10.) ** 2) * alpha[i] for i in ids))
        np.testing.assert_allclose(pred, np.clip(expected, y.min(), y.max()), atol=1e-12)

    def test_recovery_restarts_jitter_after_widening_and_raises_at_cap(self):
        import scipy.linalg
        original = scipy.linalg.cholesky_banded
        def recover(band, **kwargs):
            if len(band) < 5:
                raise np.linalg.LinAlgError('forced narrow-band failure')
            return original(band, **kwargs)
        with patch('scipy.linalg.cholesky_banded', side_effect=recover):
            _, details = gp_predict(np.arange(5.) * 20, np.arange(5.) / 4, [.5], initial_width=1, max_width=4)
        self.assertEqual([a['half_width'] for a in details['attempts']], [1] * 10 + [2] * 10 + [4])
        self.assertEqual(details['attempts'][10]['epsilon'], 1e-12)
        with patch('scipy.linalg.cholesky_banded', side_effect=np.linalg.LinAlgError('forced failure')):
            with self.assertRaisesRegex(RuntimeError, 'recovery cap'):
                gp_predict([0., 1.], [0., 1.], [.5])

    def test_constant_scores_and_controlled_stop(self):
        np.testing.assert_array_equal(normalize([.5, .5]), [0., 0.])
        pred, _ = gp_predict([0., 2.], [0., 0.], [.5, 1.])
        np.testing.assert_array_equal(pred, [0., 0.])
        with self.assertRaises(InterruptedError):
            gp_predict([0.], [0.], [1.], stop=lambda: True)


class AcquisitionTests(unittest.TestCase):
    def test_gap_heap_matches_naive_recomputed_greedy_after_each_addition(self):
        rng = np.random.default_rng(42)
        for n in (2, 9, 57):
            for trial in range(10):
                initial = sorted(rng.choice(n, max(1, n // 7), replace=False).tolist())
                mu = rng.random(n) if trial else np.zeros(n)
                budget = min(n, len(initial) + 5)
                count = budget - len(initial)
                expected, observed = [], set(initial)
                tau = (n - 1) / (budget + 1)
                for _ in range(count):
                    candidates = []
                    for i in range(n):
                        if i in observed:
                            continue
                        left = max([0] + [j for j in observed if j < i])
                        right = min([n - 1] + [j for j in observed if j > i])
                        gain = math.log1p((i - left) / tau) + math.log1p((right - i) / tau) - math.log1p((right - left) / tau)
                        candidates.append((-(mu[i] + 3.5 * gain), i))
                    selected = min(candidates)[1]
                    observed.add(selected)
                    expected.append(selected)
                self.assertEqual(acquire_batch(mu, initial, count, 30., budget), expected)

    def test_preserve_supplement_matches_naive_distance_and_earlier_ties(self):
        for n, selected, target in ((17, [3, 11], 8), (8, [3], 64), (20, [0, 19], 8)):
            expected = set(selected)
            while len(expected) < min(n, target):
                remaining = [i for i in range(n) if i not in expected]
                expected.add(max(remaining, key=lambda i: (min(abs(i - j) for j in expected), -i)))
            self.assertEqual(supplement(selected, n, target), sorted(expected))


class SelectorTests(unittest.TestCase):
    class Video:
        def __init__(self, n=30000, fps=30.): self.n, self.fps = n, fps
        def __len__(self): return self.n
        def get_avg_fps(self): return self.fps

    class Scorer:
        def reset(self): self.evaluated, self.calls, self.queries = [], 0, []
        def __call__(self, video, query, ids):
            if len(ids) > 4: raise AssertionError('unbounded decoder/scorer batch')
            self.evaluated.extend(ids)
            self.calls += 1
            self.queries.append(query)
            return [math.sin(i / 997.) * .4 + .5 for i in ids]

    def test_budget_rounds_output_timestamps_and_question_only(self):
        config = LongVideoConfig(method='forte')
        video, scorer = self.Video(6000), self.Scorer()
        chosen, details = select_forte(video, Question('q', 'Question only', ('SECRET', 'OPTION')), config, scorer)
        self.assertEqual(len(chosen), 64)
        self.assertEqual(chosen, sorted(set(chosen)))
        self.assertEqual(details['forte_scoring_budget'], 100)
        self.assertEqual(details['scorer_frame_evaluations'], 100)
        self.assertEqual(len(set(scorer.evaluated)), 100)
        self.assertEqual(set(scorer.queries), {'Question only'})
        sizes = [len(r['acquired_indices']) for r in details['forte_rounds']]
        self.assertEqual(sizes, [13, 13, 12, 12])
        self.assertEqual(details['forte_supplemental_frames'], 0)
        self.assertTrue(set(chosen).issubset(details['forte_scored_indices']))

    def test_short_video_supplements_without_spending_scoring_budget(self):
        for n, fps in ((1, 30.), (3, 30.), (300, 30.), (3, .25)):
            scorer = self.Scorer()
            chosen, details = select_forte(self.Video(n, fps), Question('q', 'Q', ('A', 'B')),
                                           LongVideoConfig(method='forte'), scorer)
            self.assertEqual(len(chosen), min(n, 64))
            self.assertEqual(details['scorer_frame_evaluations'], len(reference_grid(n, fps, .5)))
            self.assertTrue(set(details['forte_scored_indices']).issubset(chosen))

    def test_stops_before_scoring_and_rejects_bad_scorer(self):
        with self.assertRaises(InterruptedError):
            select_forte(self.Video(), Question('q', 'Q', ('A', 'B')), LongVideoConfig(method='forte'), self.Scorer(), stop=lambda: True)
        scorer = self.Scorer()
        with patch.object(type(scorer), '__call__', return_value=[float('nan')]):
            with self.assertRaises(ValueError):
                select_forte(self.Video(), Question('q', 'Q', ('A', 'B')), LongVideoConfig(method='forte'), scorer)

    def test_config_and_grid_edge_cases(self):
        self.assertEqual(reference_grid(300, 29.97, .5), [0, 75, 150, 224, 299])
        self.assertEqual(reference_grid(3, 30., .5), [1])
        for kwargs in ({'forte_rho': 0}, {'forte_rho': float('nan')}, {'forte_rounds': 0}, {'forte_beta_acq': -1}):
            with self.assertRaises(ValueError): LongVideoConfig(method='forte', **kwargs)


class RunnerIntegrationTests(unittest.TestCase):
    def test_real_runner_commits_forte_contract_and_source_indices_with_resume(self):
        from evidencelab.longvideo import run
        from evidencelab.longvideo_data import LongVideoSample
        from evidencelab.pipeline import Decision
        sample = LongVideoSample('q1', 'v.mp4', 'Public question', ('a', 'b'), 0, 'S2E', '1', 10.)
        scorer = SelectorTests.Scorer()
        class Video(SelectorTests.Video):
            def __init__(self): super().__init__(300, 30.)
            def __getitem__(self, i): return SimpleNamespace(asnumpy=lambda: np.zeros((8, 8, 3), np.uint8))
        class Backend:
            torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
            def reset_peak(self): pass
            def measurements(self): return {'peak_vram_allocated_gib': None}
            def answer(self, question, frames, times):
                if times != sorted(set(times)) or len(frames) != 64 or hasattr(question, 'answer'):
                    raise AssertionError('Invalid FORTE evidence/label leakage')
                return Decision([.9, .1], 20)
        def download(reader, entry, path, **kwargs):
            path.write_bytes(b'video')
            return 'a' * 64
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            annotations, index = root/'val.json', root/'index.json'
            annotations.write_text('[]')
            index.write_text('{}')
            stack.enter_context(patch.dict('sys.modules', {'decord': SimpleNamespace(
                bridge=SimpleNamespace(set_bridge=lambda _: None), cpu=lambda _: None,
                VideoReader=lambda *a, **k: Video())}))
            patches = {'parse_annotations': lambda _: [sample] * 1337, 'choose_subset': lambda *a: [sample],
                       'validate_ram_root': lambda _: None, 'longvideo_runtime': lambda _: {},
                       'execution_profile': lambda _: {'resolved_dtype': 'bfloat16', 'node': 'fake'},
                       'HTTPRangeSource': lambda **k: SimpleNamespace(bytes_read=0),
                       'archive_index': lambda *a: {'videos/v.mp4': [0, 5]},
                       'HFBackend': lambda _: Backend(), 'check_space': lambda *a: None,
                       'BlipITMScorer': lambda _: scorer, 'fetch_video': download}
            for name, value in patches.items(): stack.enter_context(patch('evidencelab.longvideo.' + name, value))
            config = LongVideoConfig(method='forte', limit=1)
            self.assertEqual(run(config, annotations, root/'ram', root/'out', index), 0)
            rows = [json.loads(line) for line in (root/'out/predictions.jsonl').read_text().splitlines()]
            contract = json.loads((root/'out/contract.json').read_text())
            self.assertEqual(rows[0]['scorer_unique_frames'], 5)
            self.assertEqual(rows[0]['forte_supplemental_frames'], 59)
            self.assertEqual(rows[0]['final_frames'], 64)
            self.assertEqual(contract['forte_paper'], 'https://arxiv.org/abs/2610.00573v1')
            self.assertEqual(set(scorer.queries), {'Public question'})
            self.assertFalse((root/'ram/active-video.mp4').exists())
            spec = importlib.util.spec_from_file_location('forte_audit', Path(__file__).resolve().parents[1] / 'scripts/audit_forte_run.py')
            audit_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(audit_module)
            self.assertEqual(audit_module.audit(root/'out')['audit'], 'PASS')
            rows[0]['scorer_frame_evaluations'] += 1
            (root/'out/predictions.jsonl').write_text(json.dumps(rows[0]) + '\n')
            with self.assertRaisesRegex(ValueError, 'accounting'):
                audit_module.audit(root/'out')
            self.assertEqual(run(config, annotations, root/'ram', root/'out', index, resume=True), 0)
            self.assertEqual(len((root/'out/predictions.jsonl').read_text().splitlines()), 1)

    def test_forte_and_uniform_suite_share_subset_but_have_distinct_directories(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/slurm/submit_longvideo_suite.py'
        spec = importlib.util.spec_from_file_location('forte_suite', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            for limit in (200, 0):
                plans = module.plans(Path(temp), 'molmo2', 64, ['forte', 'uniform'], 'forte-r05-v1', True, limit)
                self.assertEqual(len(plans), 2)
                self.assertNotEqual(plans[0][1], plans[1][1])
                configs = [LongVideoConfig.load(module.REPO / plan[0]) for plan in plans]
                self.assertEqual([(c.frames, c.seed, c.limit) for c in configs], [(64, 18, limit)] * 2)


if __name__ == '__main__':
    unittest.main()
