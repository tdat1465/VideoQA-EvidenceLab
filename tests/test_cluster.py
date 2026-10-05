import importlib.util
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from evidencelab.config import Config
from evidencelab.data import Sample, digest
from evidencelab.hardware import choose_dtype
from evidencelab.runner import run, runtime_versions
from evidencelab.store import read_run

spec = importlib.util.spec_from_file_location("submit", Path(__file__).resolve().parents[1] / "scripts/slurm/submit.py")
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)


class SubmissionTests(unittest.TestCase):
    def test_memory_override_preserves_resource_limits(self):
        for memory in (64, 80, 90):
            args = submit.submission_args({"gpu01", "gpu04"}, memory_gib=memory)
            self.assertIn(f"--mem={memory}G", args)
            self.assertEqual(sum(x.startswith("--mem=") for x in args), 1)
            self.assertIn("--gres=gpu:1", args)
            self.assertIn("--time=2-00:00:00", args)
            self.assertIn("--exclude=gpu04", args)
        for memory in (0, 50, 63, 91, 100):
            with self.assertRaises(ValueError):
                submit.submission_args({"gpu01"}, memory_gib=memory)

    def test_workflow_memory_defaults_and_explicit_override_reach_sbatch(self):
        from types import SimpleNamespace
        for workflow, override, expected in (("standard", [], 90), ("longvideo", [], 64),
                                             ("longvideo", ["--mem-gib", "80"], 80)):
            with self.subTest(workflow=workflow, override=override), \
                 patch("sys.argv", ["submit.py", "--workflow", workflow, "--test-only"] + override), \
                 patch.object(submit.subprocess, "run", side_effect=[
                     SimpleNamespace(stdout="gpu01\ngpu04\n"), SimpleNamespace(returncode=0)]) as run:
                self.assertEqual(submit.main(), 0)
                cmd = run.call_args_list[1].args[0]
                self.assertIn(f"--mem={expected}G", cmd)
                self.assertIn("--test-only", cmd)

    def test_pool_excludes_gpu04_and_unrelated_partition_nodes(self):
        args = submit.submission_args({"gpu01", "gpu02", "gpu03", "gpu04", "gpu05", "cpu01"})
        self.assertIn("--exclude=cpu01,gpu04,gpu05", args)
        self.assertIn("--nodes=1", args)
        self.assertIn("--mem=90G", args)
        self.assertFalse(any(x.startswith("--nodelist") for x in args))

    def test_optional_single_node_and_submission_flags(self):
        args = submit.submission_args({"gpu01", "gpu02", "gpu03", "gpu04"}, node="gpu02",
                                      account="school", test_only=True, parsable=True)
        self.assertIn("--nodelist=gpu02", args)
        self.assertIn("--exclude=gpu01,gpu03,gpu04", args)
        self.assertIn("--test-only", args)
        self.assertIn("--account=school", args)

    def test_forbidden_or_absent_pool_rejected(self):
        with self.assertRaises(ValueError):
            submit.submission_args({"gpu04"}, node="gpu04")
        with self.assertRaises(ValueError):
            submit.submission_args({"gpu04", "cpu01"})
        with self.assertRaises(ValueError):
            submit.submission_args({"gpu01"}, node="gpu02")


class PrecisionTests(unittest.TestCase):
    def test_auto_precision_and_explicit_requests(self):
        self.assertEqual(choose_dtype("auto", True), "bfloat16")
        self.assertEqual(choose_dtype("auto", False), "float16")
        self.assertEqual(choose_dtype("float16", True), "float16")
        with self.assertRaises(RuntimeError):
            choose_dtype("bfloat16", False)
        with self.assertRaises(ValueError):
            Config(dtype="float32")

    def test_python_patch_difference_allowed_minor_difference_preserved(self):
        with patch("evidencelab.runner.platform.python_version_tuple", return_value=("3", "10", "12")):
            a = runtime_versions(mock=True)
        with patch("evidencelab.runner.platform.python_version_tuple", return_value=("3", "10", "14")):
            b = runtime_versions(mock=True)
        with patch("evidencelab.runner.platform.python_version_tuple", return_value=("3", "12", "2")):
            c = runtime_versions(mock=True)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_resume_on_other_node_but_not_other_precision(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [Sample(f"q{i}", "nextqa", "val", "v.mp4", digest("v"), "Why?",
                           ("a", "b"), 0, "CW") for i in range(3)]
            manifest = root / "manifest.jsonl"
            manifest.write_text("".join(json.dumps(asdict(s))+"\n" for s in rows))
            cfg = Config(backend="mock", limit=0)
            with patch("evidencelab.runner.execution_profile", return_value={"node": "gpu01", "resolved_dtype": "float16"}):
                self.assertEqual(run(cfg, manifest, root, root / "run", max_new_samples=1), 75)
            with patch("evidencelab.runner.execution_profile", return_value={"node": "gpu02", "resolved_dtype": "bfloat16"}):
                with self.assertRaisesRegex(ValueError, "contract mismatch"):
                    run(cfg, manifest, root, root / "run", resume=True)
            with patch("evidencelab.runner.execution_profile", return_value={"node": "gpu02", "resolved_dtype": "float16"}):
                self.assertEqual(run(cfg, manifest, root, root / "run", resume=True), 0)
            contract, results = read_run(root / "run")
            self.assertEqual(contract["resolved_dtype"], "float16")
            self.assertEqual({r["execution"]["node"] for r in results}, {"gpu01", "gpu02"})


if __name__ == "__main__":
    unittest.main()
