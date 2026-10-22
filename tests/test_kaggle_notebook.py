"""Regression checks for a newer Kaggle kernel and a separate pinned worker."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]

class KaggleNotebookTests(unittest.TestCase):
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
