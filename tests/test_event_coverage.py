import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'scripts'), str(REPO/'src')]
import ridge_event_coverage as events
import ridge_event_coverage_ablation as worker
from evidencelab.data import Question
from evidencelab.pipeline import Decision
from evidencelab.store import Store, atomic_json


class Tokenizer:
    def encode(self, text, **kwargs):
        assert kwargs == {'add_special_tokens': False, 'truncation': False}
        return list(range(1, len(text.split())+1))


def question():
    a, b, c = 'a woman holding a guitar', 'a man washing red clothes', 'two people walking outside'
    return Question('q', 'Which sequence of scenes is correct?', (
        f'First, {a}. Next, {b}. Finally, {c}.',
        f'First, {c}; then {a}; finally {b}.'))


def scored():
    plan = events.event_plan(question(), Tokenizer())
    x = np.arange(200)
    vectors = [(.1+.8*np.exp(-((x-p)/2)**2)).tolist() for p in (24, 94, 155)]
    return {'question_scores': [.2]*200, 'per_event_scores': vectors, 'query_plan': plan}


class EventTests(unittest.TestCase):
    def test_public_text_only_order_and_frequency_invariant(self):
        q = question()
        p = events.event_plan(q, Tokenizer())
        self.assertTrue(p['eligible'])
        self.assertEqual(len(p['queries']), 3)
        other = Question('different-id', q.question, (q.choices[1], q.choices[0], q.choices[0]))
        self.assertEqual(p, events.event_plan(other, Tokenizer()))
        self.assertTrue(all('first' not in x['text'] and 'finally' not in x['text'] for x in p['queries']))
        self.assertIn('a man washing red clothes', p['events'])
        # No answer/category/video/ID is available to the parser or selector.
        public = SimpleNamespace(question=q.question, choices=q.choices)
        self.assertEqual(events.event_plan(public, Tokenizer()), p)

    def test_parser_fails_closed_and_keeps_meaningful_descriptors(self):
        q = question()
        for prompt, choices in [('What does the man do next?', q.choices),
                                (q.question, (q.choices[0], 'an unrelated option')),
                                (q.question, ('First, a man goes outside. Then he returns.', q.choices[0]))]:
            self.assertFalse(events.event_descriptions(Question('q', prompt, choices))['eligible'])
        self.assertNotEqual(events.normalized_event('a ship in the desert'),
                            events.normalized_event('a ship in outer space'))
        self.assertIn('first aid', events.normalized_event('A man gives first aid.'))

    def test_long_event_chunks_keep_all_tokens(self):
        long = ' '.join('token'+str(i) for i in range(80))
        choices = (f'First, {long}. Finally, a man runs away.',
                   f'First, a man runs away. Finally, {long}.')
        plan = events.event_plan(Question('q', question().question, choices), Tokenizer())
        chunked = max(plan['queries'], key=lambda p: p['token_count'])
        self.assertEqual([len(c) for c in chunked['content_ids']], [33, 33, 14])
        self.assertEqual([v for c in chunked['content_ids'] for v in c], list(range(1, 81)))

    def test_no_forced_support_for_flat_weak_or_single_sample_peaks(self):
        score = scored()
        vectors = [[.8]*200, [.1]*100+[.9]+[.1]*99, [.1]*90+[.3]*5+[.1]*105]
        matrix, support = events.event_evidence(score['question_scores'], vectors, score['query_plan'])
        self.assertEqual(matrix.shape, (0, 200))
        self.assertFalse(any(x['accepted'] for x in support))
        vectors[0][0] = float('nan')
        with self.assertRaises(ValueError):
            events.event_evidence(score['question_scores'], vectors, score['query_plan'])

    def test_coverage_recovers_separated_events_with_fixed_budget_and_context(self):
        score = scored()
        def frozen(*args, **kwargs):
            return list(range(64)), {'curve': {'s_smooth': [.2]*200}}
        with patch.object(worker.frozen, 'select_variant', side_effect=frozen):
            time, td = events.select_variant(None, score, {}, 'time16_sigma1')
            chosen, cd = events.select_variant(None, score, {}, 'eventcover_sigma1')
            for positions in (time, chosen):
                self.assertEqual(positions, sorted(set(positions)))
                self.assertEqual(len(positions), 64)
                self.assertTrue(0 <= positions[0] < positions[-1] < 200)
            self.assertTrue(set(td['time_anchors']) <= set(chosen))
            self.assertTrue({94, 155} <= set(chosen))
            self.assertTrue({93, 95, 154, 156} <= set(chosen))
            self.assertLessEqual(cd['event_additions'], 16)
            self.assertGreaterEqual(cd['retained_sigma1_frames'], 32)
            self.assertTrue(all(a >= b for a, b in zip(cd['event_coverage_after'], cd['event_coverage_at_anchors'])))
            self.assertGreater(sum(cd['event_coverage_after']), sum(td['event_coverage_after']))
            self.assertEqual(score, scored())
            score['per_event_scores'] = [[.1]*200 for _ in score['per_event_scores']]
            for variant in events.VARIANTS[2:]:
                positions, info = events.select_variant(None, score, {}, variant)
                self.assertEqual(positions, list(range(64)))
                self.assertTrue(info['fallback'])

    def test_event_cache_rejects_query_source_and_checksum_changes(self):
        s = scored()
        key = {'candidate_indices': list(range(200))}
        rec = worker.event_score(key, s['question_scores'], s['per_event_scores'], s['query_plan'], 1., 150)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'score.json'
            atomic_json(path, rec)
            self.assertEqual(worker.read_event_score(path, key, s['question_scores'], s['query_plan']), rec)
            with self.assertRaises(ValueError):
                worker.read_event_score(path, key, [.3]*200, s['query_plan'])
            bad = {**rec, 'per_event_scores': [[.9]*200]*3}
            atomic_json(path, bad)
            with self.assertRaises(ValueError):
                worker.read_event_score(path, key, s['question_scores'], s['query_plan'])
        no_plan = events.event_plan(Question('q', 'What happened?', ('one', 'two')), None)
        self.assertEqual(worker.event_score(key, [.2]*200, [], no_plan, 0., 0)['scorer_batches'], 0)

    def test_event_mean_does_not_average_in_unsupported_distractor(self):
        score = scored()
        score['per_event_scores'][-1] = [.1]*200
        captured = []
        def frozen(author, values, weights, variant, k):
            captured.append(values)
            return list(range(64)), {'curve': {'s_smooth': values}}
        with patch.object(worker.frozen, 'select_variant', side_effect=frozen):
            _, info = events.select_variant(None, score, {}, 'eventmean_sigma1')
        expected = [.5*q+.5*(a+b)/2 for q, a, b in zip(score['question_scores'], *score['per_event_scores'][:2])]
        self.assertEqual(captured[-1], expected)
        self.assertEqual(info['accepted_events'], 2)

    def test_changed_control_answer_fails_before_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = Store(root, {}, False)
            sample = SimpleNamespace(id='q', choices=('one', 'two'))
            base = list(range(64))
            with patch.object(worker, 'select_variant', return_value=(base, {})):
                with self.assertRaisesRegex(ValueError, 'control mismatch'):
                    worker.evaluate_question(sample, list(range(200)), 1., {'weights': {}},
                        {**scored(), 'sha256': 's'}, None, store, root/'curves.json',
                        lambda *args: (Decision([.9, .1], 10), .1, {'execution': {}}),
                        reference={'calls': [{'frame_indices': base}], 'prediction': 1},
                        reference_sigma1={'frame_indices': base, 'prediction': 1})
            self.assertEqual(store.results(), [])
            store.close()

    def test_resume_and_same_run_identical_frame_reuse_do_not_copy_old_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'scores').mkdir()
            contract = {'selected_count': 5, 'pilot20_manifest': {'ids': ['q']}}
            sample = SimpleNamespace(id='q', question='demo', choices=('one', 'two'), answer=1,
                                     video='v', question_type='any', duration_group='600')
            base, one = list(range(64)), list(range(63))+[70]
            ref = {'calls': [{'frame_indices': base}], 'prediction': 1}
            ref1 = {'frame_indices': one, 'prediction': 1}
            score = {**scored(), 'sha256': 'score'}
            calls = []
            def select(author, score, weights, variant):
                return (base if variant == 'sigma2' else one), {}
            def answer(indices, times):
                calls.append(indices)
                return Decision([.1, .9], 100), .25, {'execution': {'node': 'mock'}}
            with patch.object(worker, 'select_variant', side_effect=select):
                store = Store(root, contract, False)
                with self.assertRaises(InterruptedError):
                    worker.evaluate_question(sample, list(range(200)), 1., {'weights': {}, 'record_sha256': 'w'},
                        score, None, store, root/'curve.json', answer, stop=lambda: len(calls) == 2,
                        reference=ref, reference_sigma1=ref1)
                self.assertEqual(len(store.results()), 2)
                store.close()
                store = Store(root, contract, True)
                worker.evaluate_question(sample, list(range(200)), 1., {'weights': {}, 'record_sha256': 'w'},
                    score, None, store, root/'curve.json', answer, reference=ref, reference_sigma1=ref1)
                summary = worker.export(root, contract, store)
                self.assertTrue(summary['complete'])
                self.assertEqual(summary['actual_answer_forwards'], 2)
                self.assertEqual(summary['same_run_answer_reuses'], 3)
                self.assertEqual(len(calls), 2)
                self.assertEqual(summary['new_qwen_calls'], 0)
                self.assertTrue(all(r['answer_decode_seconds'] == 0 for r in store.results() if r['reused_answer_from']))
                store.close()


if __name__ == '__main__':
    unittest.main()
