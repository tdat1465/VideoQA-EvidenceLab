import ast
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
import numpy as np

from analyze_qa_pair import analyze, bootstrap
from prepare_videomme import extract_required
from qa_protocol import atomic_json, decode_selected, digest, load_pair, parse_answer, prepare_inputs, prompt, video_path

ROOT = Path(__file__).resolve().parents[1]


class QATests(unittest.TestCase):
    def test_decode_and_processor_use_exact_shortfall_frames(self):
        frames = np.zeros((3, 28, 28, 3), dtype=np.uint8)
        class Decoder:
            def __len__(self): return 100
            def get_avg_fps(self): return 25
            def get_batch(self, ids):
                self.actual_ids = ids
                return SimpleNamespace(asnumpy=lambda: frames)
        decoder = Decoder()
        actual, fps = decode_selected(decoder, [0, 20, 99])
        self.assertEqual(decoder.actual_ids, [0, 20, 99])
        self.assertIs(actual, frames)
        self.assertEqual(fps, 25)
        with self.assertRaises(ValueError):
            decode_selected(decoder, [100])
        class Processor:
            image_processor = SimpleNamespace(temporal_patch_size=2)
            grid = [[2, 2, 2]]
            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return {'video_grid_thw': np.array(self.grid)}
        processor = Processor()
        _, grid = prepare_inputs(processor, frames, 'prompt')
        self.assertIs(processor.kwargs['videos'][0], frames)
        self.assertEqual(processor.kwargs['fps'], 1.0)
        self.assertEqual(grid[0][0], 2)  # Odd N pads within temporal patch.
        processor.grid = [[1, 2, 2]]
        with self.assertRaisesRegex(RuntimeError, 'temporal'):
            prepare_inputs(processor, frames, 'prompt')

    def fixture(self, root):
        manifest = {'arguments': {'dataset_name': 'videomme', 'max_num_frames': 2, 'limit': 2}, 'query_count': 2}
        atomic_json(root / 'manifest.json', manifest)
        rows = [{'question_id': str(i), 'videoID': 'video'+str(i), 'answer': 'A', 'options': ['A. yes', 'B. no'],
                 'question': 'Which?', 'duration': 'short', 'task_type': 'Counting', 'frame_idx': [0, 2]} for i in range(2)]
        for s in ('original', 'watershed'):
            atomic_json(root / s / 'annotations.json', rows)
        atomic_json(root / 'summary.json', {'note': 'test selection fixture'})
        atomic_json(root / 'qa_pair/config.json', {'selection_manifest_sha256': digest(root / 'manifest.json'),
                    'annotation_hashes': {s: digest(root / s / 'annotations.json') for s in ('original', 'watershed')},
                    'model': 'test-fixture', 'model_revision': 'fixture'})
        return rows

    def prediction(self, i, a, b):
        pair = {'index': i, 'question_id': str(i), 'video_id': 'video'+str(i), 'gold': 'A'}
        for s, response in (('original', a), ('watershed', b)):
            pair[s] = {'response': response, 'correct': False, 'frame_ids': [0, 2], 'status': 'ok',
                       'decode_seconds': .1, 'preprocessing_seconds': .2, 'generation_seconds': .3,
                       'peak_gpu_memory_bytes': 10}
        return pair

    def test_strict_parser(self):
        for text in ('A', '(A)', 'Answer: a', 'The best answer is: A.', '[A]'):
            self.assertEqual(parse_answer(text, 4), 'A')
        for text in ('A or B', 'A because it is blue', '', 'E', 'B and C', 'not A'):
            self.assertIsNone(parse_answer(text, 4))

    def test_report_regrades_and_marks_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            atomic_json(root / 'qa_pair/predictions/000000.json', self.prediction(0, 'B', 'A'))
            with contextlib.redirect_stdout(io.StringIO()):
                report = analyze(root)
            self.assertFalse(report['complete'])
            self.assertEqual(report['coverage'], .5)
            self.assertEqual(report['accuracy'], {'original': 0, 'watershed': 1})
            self.assertIsNone(report['bootstrap_delta_95ci_percentage_points'])
            atomic_json(root / 'qa_pair/predictions/000001.json', self.prediction(1, 'A', 'unclear'))
            with contextlib.redirect_stdout(io.StringIO()):
                report = analyze(root)
            self.assertTrue(report['complete'])
            self.assertFalse(report['full_benchmark_selection'])
            self.assertEqual(report['delta_percentage_points'], 0)
            self.assertEqual(report['outcomes'], {'improved': 1, 'regressed': 1})
            self.assertEqual(report['invalid_answers']['watershed'], 1)
            self.assertTrue((root / 'qa_pair/report/per_question.csv').exists())

    def test_report_rejects_duplicate_or_wrong_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            pair = self.prediction(0, 'A', 'A')
            pair['watershed']['frame_ids'] = [1, 2]
            atomic_json(root / 'qa_pair/predictions/000000.json', pair)
            with self.assertRaisesRegex(ValueError, 'frame IDs'):
                analyze(root)
            pair['watershed']['frame_ids'] = [0, 2]
            atomic_json(root / 'qa_pair/predictions/000000.json', pair)
            atomic_json(root / 'qa_pair/predictions/duplicate.json', pair)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                analyze(root)

    def test_pair_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.fixture(root)
            load_pair(root)
            rows[0]['answer'] = 'B'
            atomic_json(root / 'watershed/annotations.json', rows)
            with self.assertRaisesRegex(ValueError, 'differ'):
                load_pair(root)
            rows[0]['answer'] = 'A'
            rows[0]['frame_idx'] = [0, 0]
            atomic_json(root / 'watershed/annotations.json', rows)
            with self.assertRaisesRegex(ValueError, 'strict frame'):
                load_pair(root)

    def test_safe_video_paths_and_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data').mkdir()
            (root / 'data/example.MP4').write_bytes(b'test')
            self.assertEqual(video_path({'videoID': 'example'}, 'videomme', root).suffix, '.MP4')
            with self.assertRaises(ValueError):
                video_path({'video_path': '../../outside.mp4'}, 'longvideobench', root)
            with self.assertRaises(FileNotFoundError):
                video_path({'videoID': 'missing'}, 'videomme', root)

    def test_archive_extract_only_known_media_no_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'input.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('../../a.mp4', b'video-a')
                z.writestr('folder/b.mp4', b'video-b')
                z.writestr('malicious.py', b'code')
            wanted = {'a'}
            self.assertEqual(extract_required(archive, root / 'data', wanted), ['a.mp4'])
            self.assertFalse(wanted)
            self.assertEqual((root / 'data/a.mp4').read_bytes(), b'video-a')
            self.assertEqual(list((root / 'data').iterdir()), [root / 'data/a.mp4'])

    def test_prompt_matches_upstream_templates(self):
        for dataset, fname in (('videomme', 'videomme_doc_to_text'), ('longvideobench', 'longvideobench_doc_to_text')):
            tree = ast.parse((ROOT / 'datasets' / dataset / 'utils.py').read_text(encoding='utf-8'))
            fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == fname)
            namespace = {}
            exec(compile(ast.Module(body=[fn], type_ignores=[]), '<upstream>', 'exec'), namespace)
            post = "\nAnswer with the option's letter from the given choices directly." if dataset == 'videomme' else "Answer with the option's letter from the given choices directly.\n"
            row = {'question': 'test?', 'options': ['A. x', 'B. y'], 'candidates': ['x', 'y']}
            expected = namespace[fname](row, {'pre_prompt': '', 'post_prompt': post})
            self.assertEqual(prompt(row, dataset), expected)

    def test_cluster_bootstrap_reproducible(self):
        records = [{'video_id': 'a', 'original_correct': False, 'watershed_correct': True}] * 3
        records += [{'video_id': 'b', 'original_correct': True, 'watershed_correct': False}]
        self.assertEqual(bootstrap(records, 100, 42), bootstrap(records, 100, 42))


if __name__ == '__main__':
    unittest.main()
