import ast
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from evidencelab.coffeemate import (ROOT, VENDOR, convert_nextqa, materialize_runtime,
                                  grade_answers, prepare, recipe, verify_prepared, verify_vendor, write_json)


class CoffeeMateTests(unittest.TestCase):
    def test_three_seed_aggregate_checks_independence(self):
        spec = importlib.util.spec_from_file_location('coffee_script', ROOT / 'scripts/coffee_mate.py')
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = []
            for seed in (42, 43, 44):
                output = root / str(seed)
                outputs.append(output)
                write_json(output / 'eval_contract.json', {'expected_questions': 1, 'protocol': 'fixture',
                           'seed': seed, 'method': 'coma', 'profile': 'source', 'versions': {},
                           'runtime_hashes': {}, 'val_sha256': 'same-fixture'})
                write_json(output / 'predictions/000000.json', {'index': 0, 'type': 'CW', 'correct': True,
                           'response': 'B) y.' if seed == 43 else 'A) x.', 'ground_truth': 'A) x.', 'generation_seconds': .1})
            result = script.aggregate(outputs, root / 'aggregate.json')
            self.assertAlmostEqual(result['metrics']['overall']['mean_accuracy'], 2/3)
            contract = json.loads((outputs[2] / 'eval_contract.json').read_text())
            contract['seed'] = 42
            write_json(outputs[2] / 'eval_contract.json', contract)
            with self.assertRaisesRegex(ValueError, 'distinct'):
                script.aggregate(outputs, root / 'invalid.json')

    def test_legacy_grader_empty_answer_audit(self):
        self.assertEqual(grade_answers('', 'A) x.'), (True, False, None))
        self.assertEqual(grade_answers('B) y.', 'A) x.'), (False, False, 'B'))
        self.assertEqual(grade_answers('A) x.', 'A) x.'), (True, True, 'A'))
    def test_full_preparation_contract_detects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, videos = self.fixture(root)
            val = root / 'val.csv'
            fields = ['video', 'qid', 'question', 'answer', 'type'] + [f'a{i}' for i in range(5)]
            (videos / '12.mp4').write_bytes(b'validation-video-fixture')
            with val.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerow(dict(video=12, qid=12, question='val?', answer=0, type='CW',
                                     **{f'a{i}': f'Option {i}' for i in range(5)}))
            umt, stage3 = root / 'umt.pth', root / 'stage3.pth'
            umt.write_bytes(b'fake-init-test-only')
            stage3.write_bytes(b'fake-init-test-only')
            vicuna = root / 'vicuna'
            write_json(vicuna / 'config.json', {'hidden_size': 4096, 'num_hidden_layers': 32})
            output = prepare(train, val, videos, umt, vicuna, stage3, root / 'prepared')
            config, contract = verify_prepared(output)
            self.assertEqual(contract['train_count'], 11)
            self.assertEqual(contract['val_count'], 1)
            self.assertEqual(config['model']['method'], 'coma')
            (videos / '12.mp4').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'video changed'):
                verify_prepared(output)

    def test_vendor_pinned(self):
        self.assertEqual(verify_vendor()['commit'], 'beaadb3925e2c6ded40e52b95190858cc89933df')

    def test_runtime_patches_compile_and_preserve_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'runtime'
            hashes = materialize_runtime(root)
            for path in root.rglob('*.py'):
                ast.parse(path.read_text(encoding='utf-8'))
            provenance = verify_vendor()['files']
            changed = {k for k in hashes if hashes[k] != provenance[k]}
            self.assertEqual(changed, {'dataset/__init__.py', 'dataset/it_dataset.py', 'dataset/video_utils.py', 'tasks/train_coffee_mate.py'})
            tree = ast.parse((root / 'dataset/video_utils.py').read_text())
            self.assertFalse(any(isinstance(n, ast.Import) and any(a.name == 'av' for a in n.names) for n in tree.body))
            self.assertEqual((root / 'models/coffee_mate.py').read_bytes(), (VENDOR / 'models/coffee_mate.py').read_bytes())
            self.assertIn('qsn_id = qsn_id.to(device', (root / 'tasks/train_coffee_mate.py').read_text())
            self.assertNotIn('return self.__getitem__(index)', (root / 'dataset/it_dataset.py').read_text())

    def test_recipe_profiles_resolve_and_propagate_method(self):
        for profile in ('source', 'paper'):
            for world in (1, 2):
                cfg = recipe('train.json', 'val.json', '/videos', 'umt.pth', '/vicuna', 'stage3.pth', '/out',
                             profile=profile, method='m2l-sd', world_size=world)
                self.assertNotIn('${', json.dumps(cfg))
                self.assertEqual(cfg['model']['method'], 'm2l-sd')
                self.assertEqual(cfg['inputs']['video_input']['num_frames_test'], 8)
                self.assertEqual(cfg['inputs']['video_input']['sample_type_test'], 'middle')
                self.assertEqual(cfg['scheduler']['epochs'], 3)
                self.assertEqual(cfg['optimizer']['weight_decay'], .02 if profile == 'source' else .2)
                if profile == 'paper':
                    self.assertEqual(cfg['batch_size'] * cfg['accumulation_steps'] * world, 4)

    def fixture(self, root, count=11):
        videos = root / 'videos'
        videos.mkdir()
        fields = ['video', 'qid', 'question', 'answer', 'type'] + [f'a{i}' for i in range(5)]
        path = root / 'train.csv'
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for i in range(count):
                (videos / f'{i}.mp4').write_bytes(b'fixture')
                writer.writerow(dict(video=i, qid=i, question=f'Question {i}?', answer=i%5, type='CW',
                                     **{f'a{k}': f'Option {k}' for k in range(5)}))
        return path, videos

    def test_nextqa_conversion_format_and_no_answer_in_question(self):
        with tempfile.TemporaryDirectory() as directory:
            path, videos = self.fixture(Path(directory))
            rows = convert_nextqa(path, videos, [f'instruction {i}' for i in range(5)], training=True)
            self.assertEqual(len(rows), 11)
            self.assertEqual(rows[0]['QA'][0]['a'], 'Answer: (A) Option 0.')
            self.assertNotIn('Answer:', rows[0]['QA'][0]['q'])
            self.assertEqual(rows[6]['QA'][0]['i'], 'instruction 1')
            val = convert_nextqa(path, videos, ['val'] * 5)
            self.assertTrue(all(r['QA'][0]['i'] == 'val' for r in val))

    def test_negative_pool_and_missing_video_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path, videos = self.fixture(Path(directory), 4)
            with self.assertRaisesRegex(ValueError, 'negatives'):
                convert_nextqa(path, videos, ['i'] * 5, training=True)
            (videos / '0.mp4').unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'Missing video'):
                convert_nextqa(path, videos, ['i'] * 5)

    def test_upstream_prompt_does_not_put_gold_in_generation(self):
        tree = ast.parse((VENDOR / 'dataset/it_dataset.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ITImgTrainDataset')
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'process_qa')
        import random
        from types import SimpleNamespace
        namespace = {'random': random}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<upstream>', 'exec'), namespace)
        obj = SimpleNamespace(random_shuffle=False, end_signal=' ', system='', begin_signal='###',
                              role=('Human', 'Assistant'), start_token='<Video>', end_token='</Video>')
        _, instruction, question, generation, answer = namespace['process_qa'](
            obj, [{'i': 'watch', 'q': 'Question? Options: (A) x (B) y', 'a': 'Answer: (B) y.'}])
        self.assertEqual(answer, 'B) y.')
        self.assertNotIn('Answer: (B)', instruction)
        self.assertNotIn('Answer: (B)', generation)
        self.assertTrue(generation.endswith('Best option: ('))

    def test_report_partial_coverage(self):
        spec = importlib.util.spec_from_file_location('coffee_script', ROOT / 'scripts/coffee_mate.py')
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / 'eval_contract.json', {'expected_questions': 2, 'protocol': 'fixture'})
            write_json(root / 'predictions/000000.json', {'index': 0, 'type': 'CW', 'correct': False,
                       'response': 'A) x.', 'ground_truth': 'A) x.', 'generation_seconds': .1})
            result = script.report(root)
            self.assertFalse(result['complete'])
            self.assertEqual(result['accuracy']['C']['accuracy'], 1)


if __name__ == '__main__': unittest.main()
