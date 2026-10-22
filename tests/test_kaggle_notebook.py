"""Regression checks for a newer Kaggle kernel and a separate pinned worker."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]

class KaggleNotebookTests(unittest.TestCase):
    def check_missing_vocab(self, *, sidecar=False, corrupt_download=False):
        notebook = json.loads((REPO / 'notebooks/RIDGE_Kaggle_T4x2.ipynb').read_text(encoding='utf-8'))
        cell = next(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
        key = 'code/src/evidencelab/_vendor/lens/API_CLIP/clip_prs/utils/vocab/bpe_simple_vocab_16e6.txt.gz'
        payload = (REPO / key.removeprefix('code/')).read_bytes()
        expected = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / 'input'
            bundle = inputs / 'dataset/bundle'
            bundle.mkdir(parents=True)
            manifest = {'files':{key:expected}}
            if sidecar:
                manifest['transport_base64'] = {key:'vocab.base64.txt'}
                (bundle / 'vocab.base64.txt').write_bytes(base64.b64encode(payload))
            (bundle / 'bundle-manifest.json').write_text(json.dumps(manifest))
            for original, replacement in [('/kaggle/input',inputs), ('/kaggle/working',root/'working'),
                                          ('/kaggle/temp',root/'temp')]:
                cell = cell.replace(original, replacement.as_posix())
            cell = cell.replace("subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], check=True)", 'pass')
            namespace = {}
            data = b'corrupt' if corrupt_download else payload
            with patch('urllib.request.urlopen', return_value=io.BytesIO(data)) as download:
                if corrupt_download:
                    with self.assertRaisesRegex(AssertionError, 'Restored asset hash mismatch'):
                        exec(compile(cell, 'input-cell', 'exec'), namespace)
                    self.assertFalse((root/'working/ridge-kaggle-v1/ridge-kaggle-bundle'/key).exists())
                else:
                    exec(compile(cell, 'input-cell', 'exec'), namespace)
                    self.assertEqual((namespace['BUNDLE']/key).read_bytes(), payload)
                if sidecar:
                    download.assert_not_called()
                else:
                    download.assert_called_once()
                    self.assertIn('/a1d071733d7111c9c014f024669f959182114e33/', download.call_args.args[0])

    def test_nested_gzip_restored_offline_from_sidecar(self):
        self.check_missing_vocab(sidecar=True)

    def test_old_dataset_restores_missing_vocab_from_pinned_source(self):
        self.check_missing_vocab()

    def test_corrupt_download_is_rejected_before_write(self):
        self.check_missing_vocab(corrupt_download=True)

    def test_new_kernel_bootstraps_pinned_worker_and_reuses_it(self):
        notebook = json.loads((REPO / 'notebooks/RIDGE_Kaggle_T4x2.ipynb').read_text(encoding='utf-8'))
        cells = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
        for i, cell in enumerate(cells):
            compile(cell, f'cell-{i}', 'exec')
        self.assertNotIn('assert (3, 10)', cells[0])
        commands = []
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            def run(command, *, env, check):
                self.assertTrue(check)
                self.assertNotIn('VIRTUAL_ENV', env)
                self.assertNotIn('PYTHONHOME', env)
                commands.append(command)
                if '--target' in command:
                    uv = Path(command[command.index('--target') + 1]) / 'bin/uv'
                    uv.parent.mkdir(parents=True)
                    uv.touch()
                if 'venv' in command:
                    python = Path(command[-1]) / 'bin/python'
                    python.parent.mkdir(parents=True)
                    python.touch()
                return SimpleNamespace(returncode=0)
            namespace = dict(Path=Path, TEMP=temporary, CODE=REPO,
                os=SimpleNamespace(environ={**os.environ, 'VIRTUAL_ENV':'/kernel', 'PYTHONHOME':'/kernel'}),
                sys=SimpleNamespace(executable='/kernel/python3.13', version_info=(3,13,0)),
                subprocess=SimpleNamespace(run=run))
            exec(compile(cells[1], 'environment-cell', 'exec'), namespace)
            bootstrap = commands[0]
            self.assertEqual(bootstrap[0], '/kernel/python3.13')
            self.assertIn('--target', bootstrap)
            self.assertIn('uv==0.8.22', bootstrap)
            creation = next(c for c in commands if 'venv' in c)
            self.assertEqual(creation[creation.index('--python') + 1], '3.12.11')
            self.assertIn('--managed-python', creation)
            self.assertEqual(namespace['PY'], temporary / 'venv-py312-v2/bin/python')
            self.assertTrue(all(c[0] != '/kernel/python3.13' for c in commands[1:]))
            before = len(commands)
            exec(compile(cells[1], 'environment-cell-repeat', 'exec'), namespace)
            self.assertEqual(len(commands), before + 1)  # Version check only; no reinstall.

if __name__ == '__main__':
    unittest.main()
