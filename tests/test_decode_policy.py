from contextlib import ExitStack
import base64
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'scripts'), str(REPO / 'scripts/slurm'), str(REPO / 'src')]
import diagnose_ridge_decode_policy as policy
spec = importlib.util.spec_from_file_location('decode_submit', REPO / 'scripts/slurm/submit_ridge_decode_policy.py')
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)


class DecodePolicyTests(unittest.TestCase):
    def test_probe_thumbnail_does_not_replace_decoded_pixels(self):
        pixels = [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(3)]
        reader = policy.SequentialReader(iter(pixels), lambda x: x, 1., 3)
        with patch.object(policy, 'PROBES', {1}):
            actual = reader[2].asnumpy()
        np.testing.assert_array_equal(actual, pixels[2])
        self.assertEqual(reader.probes['1']['thumbnail_size'], [32, 18])
        self.assertEqual(len(base64.b64decode(reader.probes['1']['thumbnail_rgb_base64'])), 32 * 18 * 3)

    def test_sequential_skips_by_decoding_and_only_converts_requested_frames(self):
        advances, conversions = [], []
        def frames():
            for i in range(7):
                advances.append(i)
                yield i
        def array(i):
            conversions.append(i)
            return np.full((2, 3, 3), i, dtype=np.uint8)
        reader = policy.SequentialReader(frames(), array, 2., 7, timestamp=lambda i: 100. + i / 2)
        for i in (0, 3, 5):
            self.assertEqual(reader[i].asnumpy()[0, 0, 0], i)
        meta = reader.finish()
        self.assertEqual(advances, list(range(7)))
        self.assertEqual(conversions, [0, 3, 5])
        self.assertTrue(meta['frame_count_matches'])
        self.assertEqual(meta['observed_decoded_frames'], 7)
        self.assertEqual(meta['max_pts_error_vs_reference_seconds'], 0.)

    def test_backward_access_is_rejected_instead_of_seeking(self):
        reader = policy.SequentialReader(iter(range(8)), lambda x: x, 1., 8)
        reader[4]
        with self.assertRaises(ValueError): reader[2]
        with self.assertRaises(ValueError): reader[-1]

    def test_frame_count_mismatch_and_early_end_are_observable(self):
        reader = policy.SequentialReader(iter(range(3)), lambda x: x, 1., 5)
        with self.assertRaises(StopIteration): reader[4]
        meta = reader.finish()
        self.assertFalse(meta['frame_count_matches'])
        self.assertEqual(meta['observed_decoded_frames'], 3)

    def test_close_is_idempotent_and_stop_interrupts_decode(self):
        closes = []
        reader = policy.SequentialReader(iter(range(8)), lambda x: x, 1., 8, close=lambda: closes.append(True))
        reader.close();reader.close()
        self.assertEqual(len(closes), 1)
        with self.assertRaises(ValueError): reader[0]
        reader = policy.SequentialReader(iter(range(8)), lambda x: x, 1., 8, stop=lambda: True)
        with self.assertRaises(InterruptedError): reader[0]

    def test_submission_passes_reference_and_uses_separate_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);repo = root / 'code/VideoQA-EvidenceLab-ridge-v1';repo.mkdir(parents=True)
            calls, queue = [], ['']
            def output(command, **kwargs):
                calls.append(command)
                if command[0] == 'squeue': return queue[0]
                if command[0] == 'sinfo': return 'gpu01\ngpu02\ngpu03\ngpu04\n'
                if command[0] == 'sbatch':
                    self.assertEqual(kwargs['env']['EXPERIMENT_MODE'], 'decode')
                    self.assertEqual(kwargs['env']['RUN_DIR'], str(root / 'decode'))
                    self.assertTrue(kwargs['env']['REFERENCE_REPORT'].endswith('session-80368' + os.sep + 'report.json'))
                    return '99998\n'
                raise AssertionError(command)
            patches = [patch.object(submit, 'ROOT', root), patch.object(submit, 'REPO', repo),
                patch.object(submit, 'BASELINE', root / 'base'), patch.object(submit, 'SOURCE', root / 'source'),
                patch.object(submit, 'DEVELOPMENT', root / 'dev'), patch.object(submit, 'FULL_RUN', root / 'full'),
                patch.object(submit, 'OUTPUT', root / 'decode'),
                patch.object(submit.diagnostic, 'validate_inputs', return_value={'snapshot': {}, 'reference_sha256': 'ref'}),
                patch.object(submit.getpass, 'getuser', return_value='lnthanh03'), patch.object(submit.subprocess, 'run'),
                patch.object(submit.subprocess, 'check_output', side_effect=output), patch.dict(os.environ, {'HF_TOKEN': 'test-only-secret'}),
                patch.object(sys, 'argv', ['submit', '--submit'])]
            with ExitStack() as stack:
                for p in patches: stack.enter_context(p)
                self.assertEqual(submit.main(), 0)
                jobs = [c for c in calls if c[0] == 'sbatch'];self.assertEqual(len(jobs), 1)
                self.assertIn('--gres=gpu:1', jobs[0]);self.assertIn('--time=04:00:00', jobs[0])
                calls.clear();queue[0] = '99998|ridge-decpol\n'
                with self.assertRaises(SystemExit): submit.main()
                self.assertFalse(any(c[0] == 'sbatch' for c in calls))


if __name__ == '__main__':
    unittest.main()
