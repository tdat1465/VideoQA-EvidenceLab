import importlib.util
import subprocess
import unittest
from pathlib import Path


class DependencyCheckTests(unittest.TestCase):
    def test_only_exact_decord_metadata_warning_is_eligible_for_smoke(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/check_longvideo_dependencies.py'
        spec = importlib.util.spec_from_file_location('dependency_check_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = subprocess.CompletedProcess([], 1, 'decord 0.6.0 is not supported on this platform\n', '')
        args = ('0.6.0', 'Tag: cp36-cp36m-manylinux2010_x86_64\n', 'Linux', 'x86_64')
        self.assertTrue(module.known_decord_tag_mismatch(result, *args))
        result.stdout += 'another package is broken\n'
        self.assertFalse(module.known_decord_tag_mismatch(result, *args))
        result.stdout = 'decord 0.6.0 is not supported on this platform\n'
        self.assertFalse(module.known_decord_tag_mismatch(result, '0.6.0', 'Tag: different', 'Linux', 'x86_64'))
        self.assertFalse(module.known_decord_tag_mismatch(result, *args[:-1], 'aarch64'))
