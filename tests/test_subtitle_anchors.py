"""Check the reusable selector against the frozen experimental source."""
import ast
from difflib import SequenceMatcher
from pathlib import Path
import random
import re
import unittest

from evidencelab import subtitle_anchors as portable

EXPERIMENT = Path(__file__).resolve().parents[1] / 'experiments' / 'subtitle_anchor_v1'


def original_functions(filename, names):
    tree = ast.parse((EXPERIMENT / filename).read_text(encoding='utf-8'))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {'re': re, 'SequenceMatcher': SequenceMatcher}
    exec(compile(ast.Module(body=functions, type_ignores=[]), filename, 'exec'), namespace)
    return namespace


class SubtitleAnchorParity(unittest.TestCase):
    def test_parser_matches_frozen_source(self):
        old = original_functions('prepare_full.py', {'seconds', 'tokens', 'windows', 'candidates'})
        rows = [
            {'start': '00:00:01,000', 'end': '00:00:02,000', 'line': 'First scene begins'},
            {'timestamp': [2, 3.2], 'text': 'then the train arrives'},
            {'start': 13, 'end': 14, 'line': 'first scene begins'},
            {'timestamp': [30, 31], 'text': 'first scene begins'},
        ]
        spans = list(portable.windows(rows))
        self.assertEqual(spans, list(old['windows'](rows)))
        for query in ('first scene begins', 'then the train arrives', 'too short', 'missing words here'):
            self.assertEqual(portable.candidates(query, spans), old['candidates'](query, spans))
        self.assertEqual(portable.queries('x', ['one', 'two'], 'TOS'), ['one', 'two'])
        self.assertEqual(portable.queries('It says "first scene begins".', [], 'T2E'), ['first scene begins'])
        self.assertEqual(portable.queries('x', ['one'], 'SSS'), [])

    def test_selection_matches_frozen_source_and_budget(self):
        old = original_functions('run_pilot.py', {'select'})['select']
        rng = random.Random(20261104)
        for _ in range(100):
            count = rng.randint(20, 2000)
            base = sorted(rng.sample(range(count), min(64, count)))
            record = {'offset': rng.choice([0, 1.5, None]), 'matches': [
                {'candidates': [(1, 1, rng.uniform(0, count / 25), rng.uniform(0, count / 25))
                                for _ in range(rng.randrange(4))]}
                for _ in range(rng.randrange(5))]}
            for limit in (1, 3):
                result = portable.select(base, count, 25, record, limit)
                self.assertEqual(result, old(base, count, 25, record, limit))
                self.assertEqual(len(result), len(base))
                self.assertEqual(len(result), len(set(result)))
        self.assertEqual(portable.select([1, 5, 9], 12, 1, {'offset': 0, 'matches': []}, 3), [1, 5, 9])


if __name__ == '__main__':
    unittest.main()
