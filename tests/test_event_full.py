import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import ExitStack
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'scripts'), str(REPO / 'scripts/slurm'), str(REPO / 'src')]
import ridge_event_full as full
from evidencelab.data import Question, digest
from evidencelab.pipeline import Decision
from evidencelab.store import Store
spec = importlib.util.spec_from_file_location('event_full_submission', REPO / 'scripts/slurm/submit_ridge_event_full.py')
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)


class FullTests(unittest.TestCase):
    def sample(self):
        return SimpleNamespace(id='q', question='What happens?', choices=('one', 'two'), answer=0,
                               video='v.mp4', question_type='TEST', duration_group='short',
                               public_question=lambda: Question('q', 'What happens?', ('one', 'two')))

    def short_score(self, n):
        return full.make_score(self.sample(), {}, list(range(n)), 1., n, 'video-sha', {}, None, None, None, None, 4, lambda: False)

    def test_every_small_grid_uses_all_candidates_and_no_scorer(self):
        for n in range(1, 65):
            score = self.short_score(n)
            self.assertIsNone(score['question_scores'])
            self.assertIsNone(score['query_plan'])
            self.assertEqual(score['question_batches'], 0)
            for v in full.VARIANTS:
                positions, details = full.select_full(None, score, {}, v)
                self.assertEqual(positions, list(range(n)))
                self.assertTrue(details['fallback'])

    def test_short_answers_are_fresh_then_reused_only_in_this_run_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            score = self.short_score(8)
            calls = []
            def answer(indices, times):
                calls.append(indices)
                return Decision([.9, .1], 10), .5, {'execution': {'resolved_dtype': 'bfloat16'}}
            weights = {'weights': {}, 'record_sha256': 'weight-sha'}
            reference = {'calls': [{'frame_indices': list(range(8))}], 'prediction': 0}
            contract = {'test': 'transaction-resume', 'selected_count': 6685, 'manifest': {'development_videos': ['v.mp4']}}
            store = Store(output, contract, False)
            with self.assertRaises(InterruptedError):
                full.evaluate(self.sample(), score, weights, None, store, output / 'curve.json', answer, reference, {}, lambda: len(store.done()) >= 2)
            self.assertEqual(len(store.results()), 2)
            self.assertEqual(len(calls), 1)
            before = store.results()
            store.close()
            store = Store(output, contract, True)
            try:
                full.evaluate(self.sample(), score, weights, None, store, output / 'curve.json', answer, reference, {})
                rows = store.results()
                self.assertEqual(len(rows), 5)
                self.assertEqual(len(calls), 1)
                self.assertEqual(sum(r['answer_inference_performed'] for r in rows), 1)
                self.assertTrue(all(r['final_frames'] == 8 for r in rows))
                self.assertTrue(all(r in rows for r in before))
                (output / 'scores').mkdir()
                (output / 'scores/score.json').write_text(json.dumps(score))
                summary = full.export(output, contract, store)
                self.assertFalse(summary['complete'])
                self.assertEqual(summary['target_variant_answers'], 6685)
                self.assertEqual(summary['completed_questions'], 1)
                self.assertEqual(summary['subsets']['development_videos']['completed_questions'], 1)
                self.assertEqual(summary['subsets']['unseen_videos']['completed_questions'], 0)
                self.assertEqual(summary['actual_answer_forwards'], 1)
                self.assertEqual(summary['same_run_answer_reuses'], 4)
            finally:
                store.close()

    def test_control_prediction_mismatch_is_not_committed(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            store = Store(output, {'test': 'control-mismatch'}, False)
            answer = lambda indices, times: (Decision([.1, .9], 10), .5, {'execution': {}})
            try:
                with self.assertRaisesRegex(ValueError, 'baseline prediction mismatch'):
                    full.evaluate(self.sample(), self.short_score(8), {'weights': {}, 'record_sha256': 'w'}, None,
                                  store, output / 'curve.json', answer,
                                  {'calls': [{'frame_indices': list(range(8))}], 'prediction': 0}, {})
                self.assertEqual(store.results(), [])
            finally:
                store.close()

    def test_fresh_question_scoring_retains_65_candidates_without_event_queries(self):
        class Scorer:
            processor = SimpleNamespace(tokenizer=None)
            def reset(self): self.calls = 0
            def __call__(self, video, text, indices):
                self.calls += 1
                return [.5] * len(indices)
        score = full.make_score(self.sample(), {'development_event_inventory': {}}, list(range(65)), 1., 65,
                                'video-sha', {'inventory': {}}, None, None, None, Scorer(), 4, lambda: False)
        self.assertEqual(score['question_scores'], [.5] * 65)
        self.assertEqual(score['question_batches'], 17)
        self.assertEqual(score['question_origin'], 'fresh-blip')
        self.assertEqual(score['event_batches'], 0)

    def test_cache_checksum_and_empty_grid_are_rejected(self):
        score = self.short_score(8)
        score['question_batches'] = 1
        with self.assertRaises(ValueError): full.validate_score(score, score['key'])
        with self.assertRaises(ValueError): self.short_score(0)

    def test_progress_requires_all_five_variants_and_reads_journal_without_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'results.sqlite3'
            self.assertEqual(submit.progress(path, {'q'}), (0, 0))
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE results (value TEXT)')
                db.executemany('INSERT INTO results VALUES (?)', [(json.dumps({'id': 'q::' + v, 'question_id': 'q', 'variant': v}),) for v in full.VARIANTS[:2]])
            db.close()
            before = path.read_bytes()
            self.assertEqual(submit.progress(path, {'q'}), (2, 0))
            self.assertEqual(path.read_bytes(), before)

    def test_launcher_submits_one_gpu_and_blocks_active_or_completed_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / 'code/VideoQA-EvidenceLab-ridge-v1'
            repo.mkdir(parents=True)
            calls, queue = [], ['']
            data = {'inputs': {}, 'manifest': {'ids': [f'q{i}' for i in range(1337)]}, 'base': {}}
            def command(command, **kwargs):
                calls.append(command)
                if command[0] == 'squeue': return queue[0]
                if command[0] == 'sinfo': return 'gpu01\ngpu02\ngpu03\ngpu04\n'
                if command[0] == 'sbatch':
                    self.assertEqual(kwargs['env']['EXPERIMENT_MODE'], 'full1337')
                    self.assertEqual(kwargs['env']['EVENT_DEV_RUN'], str(root / 'dev200'))
                    self.assertEqual(kwargs['env']['RESUME'], '0')
                    return '12345\n'
                raise AssertionError(command)
            patches = [patch.object(submit, 'ROOT', root), patch.object(submit, 'REPO', repo),
                patch.object(submit, 'BASELINE', root / 'baseline'), patch.object(submit, 'SOURCE', root / 'source'),
                patch.object(submit, 'DEVELOPMENT', root / 'dev200'), patch.object(submit, 'OUTPUT', root / 'full1337'),
                patch.object(submit.full, 'validate_inputs', return_value=data),
                patch.object(submit.full, 'code_contract', return_value={}),
                patch.object(submit.full, 'source_fingerprint', return_value='frozen'),
                patch.object(submit.getpass, 'getuser', return_value='lnthanh03'),
                patch.object(submit.subprocess, 'run'), patch.object(submit.subprocess, 'check_output', side_effect=command),
                patch.dict(os.environ, {'HF_TOKEN': 'test-only-secret'}),
                patch.object(sys, 'argv', ['submit', 'full1337', '--node', 'any', '--submit'])]
            with ExitStack() as stack:
                for item in patches: stack.enter_context(item)
                self.assertEqual(submit.main(), 0)
                jobs = [c for c in calls if c[0] == 'sbatch']
                self.assertEqual(len(jobs), 1)
                for flag in ('--gres=gpu:1', '--mem=90G', '--cpus-per-task=8', '--exclude=gpu04', '--time=2-00:00:00'):
                    self.assertIn(flag, jobs[0])
                record = json.loads((repo / 'logs/ridge-event-full-12345.json').read_text())
                self.assertEqual(record['target_variant_answers'], 6685)
                calls.clear()
                queue[0] = '12345|ridge-evtfull\n'
                with self.assertRaises(SystemExit): submit.main()
                self.assertFalse(any(c[0] == 'sbatch' for c in calls))
                calls.clear()
                with patch.object(submit, 'progress', return_value=(6685, 1337)):
                    self.assertEqual(submit.main(), 0)
                self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
