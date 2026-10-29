import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import io
from unittest.mock import patch
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'scripts'), str(REPO / 'src')]
import ridge_canonical_frames as pixels
import ridge_event_canonical as worker
from evidencelab.data import Question, digest
from evidencelab.pipeline import Decision
from evidencelab.store import Store
sys.path.insert(0, str(REPO / 'scripts/slurm'))
spec = importlib.util.spec_from_file_location('canonical_submit', REPO / 'scripts/slurm/submit_ridge_event_canonical.py')
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)


class CanonicalTests(unittest.TestCase):
    def factory(self, drift=False, count=9):
        calls = []
        def frames(path, expected, fps):
            calls.append([])
            for i in range(count):
                calls[-1].append(i)
                value = i + (20 if drift and len(calls) == 2 and i == 4 else 0)
                yield pixels.Frame(np.full((3, 5, 3), value, dtype=np.uint8))
        return frames, calls

    def build(self, root, factory, **kw):
        return pixels.build(Path('unused.mp4'), root, 9, 3., [0, 4, 8], 'a'*64,
                            lambda *a: None, factory=factory, **kw)

    def test_two_passes_reach_eof_and_png_round_trip_is_exact(self):
        factory, calls = self.factory()
        with tempfile.TemporaryDirectory() as temp:
            cache = self.build(Path(temp) / 'cache', factory)
            self.assertEqual(calls, [list(range(9)), list(range(9))])
            self.assertEqual(sorted(p.name for p in cache.root.iterdir()), ['0.png', '4.png', '8.png'])
            # Backward accesses are now cache reads; no video seek is possible.
            for i in (8, 0, 4, 0):
                np.testing.assert_array_equal(cache[i].asnumpy(), np.full((3, 5, 3), i, dtype=np.uint8))
            with self.assertRaises(KeyError): cache[2]

    def test_second_decode_drift_aborts_and_cleans_tmpfs_cache(self):
        factory, _ = self.factory(drift=True)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'cache'
            with self.assertRaisesRegex(ValueError, 'repeat pixels differ'):
                self.build(root, factory)
            self.assertFalse(root.exists())

    def test_eof_and_memory_guard_failures_prevent_inference(self):
        for observed in (8, 10):
            factory, _ = self.factory(count=observed)
            with tempfile.TemporaryDirectory() as temp:
                with self.assertRaises(ValueError):
                    self.build(Path(temp) / 'cache', factory)
        factory, _ = self.factory()
        with tempfile.TemporaryDirectory() as temp, patch.dict(pixels.POLICY, max_cache_bytes=1):
            root = Path(temp) / 'cache'
            with self.assertRaises(MemoryError): self.build(root, factory)
            self.assertFalse(root.exists())

    def test_cache_corruption_and_cancel_are_rejected(self):
        factory, _ = self.factory()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'cache'
            with self.assertRaises(InterruptedError): self.build(root, factory, stop=lambda: True)
            self.assertFalse(root.exists())
            cache = self.build(root, factory)
            (root / '4.png').write_bytes((root / '0.png').read_bytes())
            with self.assertRaisesRegex(ValueError, 'pixels changed'): cache[4]

    def test_uniform_preserves_native_grid_when_ridge_has_few_candidates(self):
        score = {'key': {'source_frames': 200, 'candidate_indices': [0, 30, 60, 90, 120, 150, 180]}}
        u, _ = worker.selected_frames(None, score, {}, 'uniform64')
        r, _ = worker.selected_frames(None, score, {}, 'sigma2')
        self.assertEqual(len(u), 64)
        self.assertEqual((u[0], u[-1]), (0, 199))
        self.assertEqual(r, score['key']['candidate_indices'])

    def test_full_gate_requires_every_pilot_variant(self):
        manifest = {'pilot_ids': ['q'+str(i) for i in range(23)]}
        rows = [{'id': q+'::'+v} for q in manifest['pilot_ids'] for v in worker.VARIANTS]
        worker.require_pilot(rows, manifest)
        with self.assertRaises(ValueError): worker.require_pilot(rows[:-1], manifest)

    def test_submit_routes_pilot_and_blocks_full_without_pilot(self):
        ids = ['q'+str(i) for i in range(23)]
        data = {'manifest': {'ids': ids, 'pilot_ids': ids}, 'base': {}}
        calls = []
        def check_output(argv, **kw):
            calls.append((argv, kw))
            if argv[0] == 'squeue': return ''
            if argv[0] == 'sinfo': return 'gpu01\ngpu02\ngpu03\ngpu04\n'
            if argv[0] == 'sbatch': return '90000\n'
            self.fail(str(argv))
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            repo = root / 'code/VideoQA-EvidenceLab-ridge-v1'
            repo.mkdir(parents=True)
            stack.enter_context(patch.multiple(submit, ROOT=root, REPO=repo, OUTPUT=root/'new-v2'))
            stack.enter_context(patch.object(submit.getpass, 'getuser', return_value='lnthanh03'))
            stack.enter_context(patch.object(worker, 'prepare', return_value=data))
            stack.enter_context(patch.object(worker, 'expected_contract', return_value={}))
            stack.enter_context(patch.object(submit.subprocess, 'run'))
            stack.enter_context(patch.object(submit.subprocess, 'check_output', side_effect=check_output))
            stack.enter_context(patch.dict(os.environ, HF_TOKEN='unit-test-fixture'))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            with patch.object(sys, 'argv', ['submit', 'full1337', '--submit']):
                with self.assertRaises(SystemExit): submit.main()
            self.assertEqual(calls, [])
            with patch.object(sys, 'argv', ['submit', 'pilot23', '--submit', '--session-minutes', '360']):
                self.assertEqual(submit.main(), 0)
            dispatch = [c for c in calls if c[0][0] == 'sbatch']
            self.assertEqual(len(dispatch), 1)
            argv, kw = dispatch[0]
            self.assertIn('--gres=gpu:1', argv)
            self.assertIn('--mem=90G', argv)
            self.assertIn('--exclude=gpu04', argv)
            self.assertEqual(kw['env']['EXPERIMENT_MODE'], 'pilot23')
            self.assertTrue(kw['env']['RUN_DIR'].endswith('new-v2'))
            self.assertIn('ridge_event_canonical_job.sh', argv[-1])

    def test_evaluate_resume_reuses_only_v2_rows_and_rejects_changed_pixels(self):
        sample = SimpleNamespace(id='q', video='x.mp4', question='Q?', choices=('a','b'), answer=0,
                                 question_type='test', duration_group='short')
        weights = {'weights': {}, 'record_sha256': 'w'}
        score = {'key': {'source_frames': 3, 'candidate_indices': [0,1,2], 'fps': 1.,
                         'pixel_manifest_sha256': 'p'}, 'sha256': 's'}
        baseline = {'calls': [{'frame_indices': [1]}], 'prediction': 1}
        calls = []
        def answer(indices, times):
            calls.append(indices)
            return Decision([.8,.2], 10), 1., {'execution': {'test': True}}
        with tempfile.TemporaryDirectory() as temp:
            store = Store(Path(temp), {}, False)
            try:
                worker.evaluate(sample, score, weights, None, store, Path(temp)/'curve.json', answer, baseline, lambda:False)
                self.assertEqual(len(calls), 1)
                self.assertEqual(len(store.results()), 6)
                worker.evaluate(sample, score, weights, None, store, Path(temp)/'curve.json', answer, baseline, lambda:False)
                self.assertEqual(len(calls), 1)
                score['key']['pixel_manifest_sha256'] = 'changed'
                with self.assertRaisesRegex(ValueError, 'Resume pixels'):
                    worker.evaluate(sample, score, weights, None, store, Path(temp)/'curve.json', answer, baseline, lambda:False)
            finally:
                store.close()


if __name__ == '__main__':
    unittest.main()
