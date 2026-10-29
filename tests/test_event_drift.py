from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'scripts'), str(REPO / 'scripts/slurm'), str(REPO / 'src')]
import diagnose_ridge_event_drift as diagnostic
spec = importlib.util.spec_from_file_location('event_drift_submit', REPO / 'scripts/slurm/submit_ridge_event_drift.py')
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)


class DriftTests(unittest.TestCase):
    def test_tracking_preserves_decoded_array_and_read_order(self):
        class Reader:
            def __init__(self): self.accesses = []
            def __getitem__(self, i):
                self.accesses.append(i)
                return diagnostic.TrackedFrame(np.full((2, 3, 3), i, dtype=np.uint8))
        reader = Reader()
        tracked = diagnostic.TrackedReader(reader)
        self.assertEqual(tracked[2].asnumpy()[0, 0, 0], 2)
        self.assertEqual(tracked.raw_pixels, [])
        tracked.capture = True
        for i in (4, 1, 8):
            np.testing.assert_array_equal(tracked[i].asnumpy(), np.full((2, 3, 3), i, dtype=np.uint8))
        self.assertEqual(reader.accesses, [2, 4, 1, 8])
        self.assertEqual([x['frame_index'] for x in tracked.raw_pixels], [4, 1, 8])
        self.assertEqual(tracked.last_index, 8)

    def test_comparison_separates_pixel_changes_and_score_changes(self):
        a = {'question_scores': [.1, .2], 'frame_indices': [0, 1],
             'raw_pixels': [{'frame_index': 0, 'sha256': 'a'}, {'frame_index': 1, 'sha256': 'b'}],
             'transformed_pixels': [{'frame_index': 0, 'sha256': 'c'}, {'frame_index': 1, 'sha256': 'd'}]}
        b = json.loads(json.dumps(a))
        b['question_scores'][1] += 1e-7
        result = diagnostic.compare_cases(a, b)
        self.assertEqual(result['raw_pixels']['different_frames'], 0)
        self.assertEqual(result['transformed_pixels']['different_frames'], 0)
        self.assertEqual(result['scores']['different_samples'], 1)
        b['raw_pixels'][1]['sha256'] = 'changed'
        self.assertEqual(diagnostic.compare_cases(a, b)['raw_pixels']['frame_indices'], [1])

    def test_comparison_rejects_different_fingerprint_grid(self):
        a = {'question_scores': [.1], 'frame_indices': [0], 'raw_pixels': [{'frame_index': 0}], 'transformed_pixels': [{'frame_index': 0}]}
        b = {**a, 'raw_pixels': [{'frame_index': 1}]}
        with self.assertRaises(ValueError): diagnostic.compare_cases(a, b)

    def test_snapshot_guard_detects_evaluation_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'scores').mkdir()
            for name in ('contract.json', 'predictions.jsonl'):
                (root / name).write_text('{}')
            target = root / 'scores' / (diagnostic.digest(diagnostic.TARGET) + '.json')
            target.write_text('{}')
            snapshot = {'full_contract_sha256': diagnostic.file_hash(root / 'contract.json'),
                'full_predictions_sha256': diagnostic.file_hash(root / 'predictions.jsonl'),
                'target_score_file_sha256': diagnostic.file_hash(target)}
            self.assertTrue(diagnostic.snapshot_unchanged(snapshot, root))
            target.write_text('{"changed":true}')
            self.assertFalse(diagnostic.snapshot_unchanged(snapshot, root))

    def test_submission_is_separate_single_gpu_and_blocks_full_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / 'code/VideoQA-EvidenceLab-ridge-v1'
            repo.mkdir(parents=True)
            calls, queue = [], ['']
            def output(command, **kwargs):
                calls.append(command)
                if command[0] == 'squeue': return queue[0]
                if command[0] == 'sinfo': return 'gpu01\ngpu02\ngpu03\ngpu04\n'
                if command[0] == 'sbatch':
                    self.assertEqual(kwargs['env']['RUN_DIR'], str(root / 'diagnostic'))
                    self.assertEqual(kwargs['env']['FULL_RUN'], str(root / 'full'))
                    self.assertEqual(kwargs['env']['EXPERIMENT_MODE'], 'diagnose')
                    return '99999\n'
                raise AssertionError(command)
            patches = [patch.object(submit, 'ROOT', root), patch.object(submit, 'REPO', repo),
                patch.object(submit, 'BASELINE', root / 'baseline'), patch.object(submit, 'SOURCE', root / 'source'),
                patch.object(submit, 'DEVELOPMENT', root / 'dev'), patch.object(submit, 'FULL_RUN', root / 'full'),
                patch.object(submit, 'OUTPUT', root / 'diagnostic'),
                patch.object(submit.diagnostic, 'validate_inputs', return_value={'snapshot': {}}),
                patch.object(submit.getpass, 'getuser', return_value='lnthanh03'),
                patch.object(submit.subprocess, 'run'), patch.object(submit.subprocess, 'check_output', side_effect=output),
                patch.dict(os.environ, {'HF_TOKEN': 'test-only-secret'}),
                patch.object(sys, 'argv', ['submit', '--submit'])]
            with ExitStack() as stack:
                for p in patches: stack.enter_context(p)
                self.assertEqual(submit.main(), 0)
                jobs = [c for c in calls if c[0] == 'sbatch']
                self.assertEqual(len(jobs), 1)
                for flag in ('--gres=gpu:1', '--mem=90G', '--nodelist=gpu03', '--time=04:00:00'):
                    self.assertIn(flag, jobs[0])
                exclusion = next(x for x in jobs[0] if x.startswith('--exclude='))
                self.assertIn('gpu04', exclusion.removeprefix('--exclude=').split(','))
                queue[0] = '80364|ridge-evtfull\n'
                calls.clear()
                with self.assertRaises(SystemExit): submit.main()
                self.assertFalse(any(c[0] == 'sbatch' for c in calls))


if __name__ == '__main__':
    unittest.main()
