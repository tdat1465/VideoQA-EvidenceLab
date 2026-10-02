import importlib.util
from contextlib import ExitStack
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evidencelab.aks_adapter import aks_candidates, select_aks
from evidencelab.data import Question
from evidencelab.longvideo_config import LongVideoConfig
from evidencelab.config import source_fingerprint
from evidencelab.store import Store
from evidencelab.longvideo import run
from evidencelab.longvideo_data import LongVideoSample
from evidencelab.pipeline import Decision


class AKSLongVideoTests(unittest.TestCase):
    def test_real_runner_wires_aks_scoring_and_commits_source_indices(self):
        import numpy as np
        sample = LongVideoSample('q1', 'v.mp4', 'Public question', ('a', 'b'), 0, 'S2E', '1', 10.)
        queries = []
        class Video:
            def __len__(self): return 300
            def get_avg_fps(self): return 30.
            def __getitem__(self, i): return SimpleNamespace(asnumpy=lambda: np.zeros((8, 8, 3), np.uint8))
        class Scorer:
            def reset(self): self.evaluated, self.calls = [], 0
            def __call__(self, video, query, ids):
                queries.append(query)
                self.evaluated.extend(ids)
                self.calls += 1
                return [i / 300 for i in ids]
        class Backend:
            torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
            def reset_peak(self): pass
            def measurements(self): return {'peak_vram_allocated_gib': None}
            def answer(self, question, frames, times):
                if times != [1., 3.] or hasattr(question, 'answer'): raise AssertionError('Invalid AKS evidence')
                return Decision([.9, .1], 20)
        def download(reader, entry, path, **kwargs):
            path.write_bytes(b'video')
            return 'a' * 64
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            annotations = root / 'val.json'
            annotations.write_text('[]')
            index = root / 'index.json'
            index.write_text('{}')
            stack.enter_context(patch.dict('sys.modules', {'decord': SimpleNamespace(
                bridge=SimpleNamespace(set_bridge=lambda _: None), cpu=lambda _: None,
                VideoReader=lambda *a, **k: Video())}))
            patches = {'parse_annotations': lambda _: [sample]*1337, 'choose_subset': lambda *a: [sample],
                       'validate_ram_root': lambda _: None, 'longvideo_runtime': lambda _: {},
                       'execution_profile': lambda _: {'resolved_dtype': 'bfloat16', 'node': 'fake'},
                       'HTTPRangeSource': lambda **k: SimpleNamespace(bytes_read=0),
                       'archive_index': lambda *a: {'videos/v.mp4': [0, 5]},
                       'HFBackend': lambda _: Backend(), 'check_space': lambda *a: None,
                       'BlipITMScorer': lambda _: Scorer(), 'fetch_video': download}
            for name, value in patches.items():
                stack.enter_context(patch('evidencelab.longvideo.' + name, value))
            stack.enter_context(patch('evidencelab.cli.fetch_aks'))
            stack.enter_context(patch('evidencelab.aks_adapter.aks', return_value=([1, 3], 'upstream')))
            self.assertEqual(run(LongVideoConfig(method='aks', frames=8, limit=1), annotations,
                                 root/'ram', root/'out', index), 0)
            summary = json.loads((root/'out/summary.json').read_text())
            contract = json.loads((root/'out/contract.json').read_text())
            self.assertEqual(summary['mean_vlm_calls'], 1)
            self.assertEqual(summary['completed_count'], 1)
            self.assertEqual(contract['aks_revision'], 'b0b8a58fedf1d05d78151e2969cecde60e83721d')
            self.assertTrue(all(q == 'Public question' for q in queries))
            self.assertFalse((root/'ram/active-video.mp4').exists())

    def test_candidates_follow_official_floor_fps_and_boundaries(self):
        self.assertEqual(aks_candidates(300, 29.97), [i * 29 for i in range(10)])
        self.assertEqual(aks_candidates(300, 30), [i * 30 for i in range(10)])
        self.assertEqual(aks_candidates(3, 30), [0])
        self.assertEqual(aks_candidates(3, .5), [0, 1, 2])
        for count, fps in ((0, 30), (1, float('nan')), (1, 0)):
            with self.assertRaises(ValueError):
                aks_candidates(count, fps)

    def test_bounded_scoring_maps_positions_to_source_frames_without_choices(self):
        class Video:
            def __len__(self): return 300
            def get_avg_fps(self): return 30.
        class Scorer:
            def reset(self): self.evaluated, self.calls = [], 0
            def __call__(self, video, query, ids):
                self.query = query
                self.evaluated.extend(ids)
                self.calls += 1
                if len(ids) > 4: raise AssertionError('unbounded frame batch')
                return [i / 300 for i in ids]
        scorer = Scorer()
        config = LongVideoConfig(method='aks', frames=8)
        question = Question('q', 'Question only', ('hidden choice', 'another'))
        with patch('evidencelab.aks_adapter.aks', return_value=([0, 2, 9], 'upstream')) as upstream:
            indices, details = select_aks(Video(), question, config, scorer, Path('not-read.py'))
        self.assertEqual(indices, [0, 60, 270])
        self.assertEqual(scorer.query, 'Question only')
        self.assertEqual(details['scorer_batches'], 3)
        self.assertEqual(details['scorer_frame_evaluations'], 10)
        self.assertEqual(len(upstream.call_args.args[0]), 10)
        with self.assertRaises(InterruptedError):
            select_aks(Video(), question, config, scorer, Path('not-read.py'), stop=lambda: True)


class SuitePlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'scripts/slurm/submit_longvideo_suite.py'
        spec = importlib.util.spec_from_file_location('suite_planning_test', path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_independent_directories_and_resume_skip_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = self.module.plans(root, 'molmo2', 64, ['aks', 'focus'], 'v1', False)
            self.assertEqual(len({r[1] for r in rows}), 2)
            first = rows[0][1]
            first.mkdir(parents=True)
            config = LongVideoConfig.load(self.module.REPO / rows[0][0])
            store = Store(first, {'config': config.contract(), 'source_sha256': source_fingerprint(),
                                  'selected_count': 1}, False)
            store.close()
            with self.assertRaisesRegex(ValueError, '--resume'):
                self.module.plans(root, 'molmo2', 64, ['aks', 'focus'], 'v1', False)
            rows = self.module.plans(root, 'molmo2', 64, ['aks', 'focus'], 'v1', True)
            self.assertEqual([r[2] for r in rows], [True, False])
            store = Store(first, {'config': config.contract(), 'source_sha256': source_fingerprint(),
                                  'selected_count': 1}, True)
            store.put({'id': 'q1'})
            store.close()
            self.assertEqual(len(self.module.plans(root, 'molmo2', 64, ['aks', 'focus'], 'v1', True)), 1)

    def test_full_validation_and_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = self.module.plans(root, 'llava_video', 64, ['uniform', 'aks', 'focus', 'lens'], 'full1', False, limit=0)
            self.assertTrue(all('_full.json' in row[0] and '-full-' in row[1].name for row in rows))
            for label in ('../bad', 'spaces not allowed', ''):
                with self.assertRaises(ValueError):
                    self.module.plans(root, 'molmo2', 64, ['aks'], label, False)
            with self.assertRaises(ValueError):
                self.module.plans(root, 'molmo2', 64, ['aks', 'aks'], 'v1', False)
