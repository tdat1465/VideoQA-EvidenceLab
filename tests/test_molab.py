import importlib.util
import json
import os
import sqlite3
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from evidencelab.longvideo import run, validate_ram_root
from evidencelab.longvideo_config import LongVideoConfig
from evidencelab.longvideo_data import cgroup_memory, check_space
from evidencelab.notebook_backup import backup_run, restore_run
from evidencelab.store import read_run
from test_longvideo import annotation


class MolabTests(unittest.TestCase):
    def test_decord_exception_only_accepts_exact_known_metadata_issue(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/check_molab_dependencies.py'
        spec = importlib.util.spec_from_file_location('molab_dependencies_test', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = subprocess.CompletedProcess([], 1, 'decord 0.6.0 is not supported on this platform\n', '')
        args = ('0.6.0', 'Tag: cp36-cp36m-manylinux2010_x86_64\n', 'Linux', 'x86_64')
        self.assertTrue(module.known_decord_tag_mismatch(result, *args))
        result.stdout += 'torch has incompatible numpy\n'
        self.assertFalse(module.known_decord_tag_mismatch(result, *args))
        result.stdout = 'decord 0.6.0 is not supported on this platform\n'
        self.assertFalse(module.known_decord_tag_mismatch(result, '0.6.0', 'Tag: different', 'Linux', 'x86_64'))
        self.assertFalse(module.known_decord_tag_mismatch(result, *args[:-1], 'aarch64'))

    def test_namespace_cgroup_limit_160_gib_without_slurm(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"EVIDENCELAB_EXECUTION": "molab"}):
            root = Path(temp)
            mount = root / "sys"
            memory = mount / "memory"
            memory.mkdir(parents=True)
            proc = root / "proc"
            proc.write_text("6:memory:/container-id\n")
            (memory / "memory.limit_in_bytes").write_text(str(160 * 1024**3))
            (memory / "memory.usage_in_bytes").write_text(str(20 * 1024**3))
            result = cgroup_memory(proc, mount)
            self.assertEqual(result["limit_bytes"], 160*1024**3)
            self.assertEqual(result["current_bytes"], 20*1024**3)

    def test_notebook_guard_caps_software_budget_not_reported_container_limit(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"EVIDENCELAB_EXECUTION": "molab"}):
            with patch('evidencelab.longvideo_data.shutil.disk_usage', return_value=type('Usage', (), {'free': 500*1024**3})()):
                with patch('evidencelab.longvideo_data.cgroup_memory', return_value={
                    'limit_bytes':160*1024**3, 'current_bytes':85*1024**3}):
                    with self.assertRaises(MemoryError):
                        check_space(Path(temp), 0)
                with patch('evidencelab.longvideo_data.cgroup_memory', return_value={}):
                    with self.assertRaises(RuntimeError):
                        check_space(Path(temp), 0)

    def test_notebook_does_not_disable_cluster_or_filesystem_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(RuntimeError):
                    validate_ram_root(root)
            with patch.dict(os.environ, {'EVIDENCELAB_EXECUTION':'molab'}, clear=True), \
                    patch('evidencelab.longvideo.subprocess.check_output', return_value='overlay\n'):
                with self.assertRaises(RuntimeError):
                    validate_ram_root(root)
            with patch.dict(os.environ, {'EVIDENCELAB_EXECUTION':'molab', 'SLURM_JOB_ID':'123'}, clear=True):
                with self.assertRaises(RuntimeError):
                    validate_ram_root(root)

    def test_backup_online_database_excludes_secrets_and_resume_matches(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            annotations = root / 'val.json'
            annotations.write_text(json.dumps([annotation(i) for i in range(4)]))
            config = LongVideoConfig(backend='mock', method='uniform', limit=4)
            output = root / 'run'
            self.assertEqual(run(config, annotations, root/'ram', output, root/'index', max_new_samples=2), 75)
            (output / '.env').write_text('TOKEN=never include')
            (output / 'video.mp4').write_bytes(b'not in backup')
            # An open writer connection is allowed; backup snapshots committed data only.
            db = sqlite3.connect(output/'results.sqlite3')
            db.execute("INSERT INTO results VALUES (?, ?)", ('uncommitted', '{}'))
            archive = root / 'backup.zip'
            try:
                self.assertEqual(backup_run(output, archive), 2)
            finally:
                db.close()
            with zipfile.ZipFile(archive) as z:
                self.assertNotIn('.env', z.namelist())
                self.assertNotIn('video.mp4', z.namelist())
                self.assertEqual(json.loads(z.read('summary.json'))['completed_count'], 2)
            restored = root / 'restored'
            self.assertEqual(restore_run(archive, restored), 2)
            self.assertEqual(run(config, annotations, root/'ram', restored, root/'index', resume=True), 0)
            self.assertEqual(len(read_run(restored)[1]), 4)
            with self.assertRaises(FileExistsError):
                restore_run(archive, restored)

    def test_restore_rejects_paths_before_writing_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root/'bad.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('results.sqlite3', b'fake')
                z.writestr('../.env', b'bad')
            with self.assertRaises(ValueError):
                restore_run(archive, root/'out')
            self.assertFalse((root/'out').exists())

    def test_driver_environment_isolates_caches_and_does_not_spoof_slurm(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/molab.py'
        spec = importlib.util.spec_from_file_location('molab_driver_test', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True):
            root = Path(temp)
            runtime = root / 'runtime'
            env = module.environment(root, runtime)
            self.assertEqual(env['EVIDENCELAB_EXECUTION'], 'molab')
            self.assertNotIn('SLURM_JOB_ID', env)
            self.assertEqual(Path(env['UV_PYTHON_INSTALL_DIR']), runtime/'python')
            for name in ('HF_HOME', 'TMPDIR', 'HF_MODULES_CACHE', 'TORCH_HOME', 'CUDA_CACHE_PATH'):
                self.assertTrue(Path(env[name]).is_relative_to(root))


if __name__ == '__main__':
    unittest.main()
