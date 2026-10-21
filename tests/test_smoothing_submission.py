import importlib.util
from pathlib import Path
import sys
import unittest

SLURM = Path(__file__).resolve().parents[1] / "scripts/slurm"
sys.path.insert(0, str(SLURM))
spec = importlib.util.spec_from_file_location("smoothing_submission", SLURM/"submit_ridge_smoothing.py")
submission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submission)


class SmoothingSubmissionTests(unittest.TestCase):
    def test_any_keeps_all_three_eligible_and_requests_one_gpu(self):
        args, eligible = submission.placement_args({"gpu01","gpu02","gpu03","gpu04","cpu01"})
        self.assertEqual(eligible,["gpu01","gpu02","gpu03"])
        self.assertIn("--nodes=1",args)
        self.assertIn("--gres=gpu:1",args)
        self.assertIn("--exclude=cpu01,gpu04",args)
        self.assertFalse(any(v.startswith("--nodelist=") for v in args))

    def test_each_explicit_node_is_supported(self):
        nodes={"gpu01","gpu02","gpu03","gpu04"}
        for node in ("gpu01","gpu02","gpu03"):
            args,eligible=submission.placement_args(nodes,node)
            self.assertEqual(eligible,[node])
            self.assertIn("--nodelist="+node,args)
            excluded=next(v.split("=",1)[1] for v in args if v.startswith("--exclude="))
            self.assertEqual(set(excluded.split(",")),nodes-{node})

    def test_unsupported_or_absent_nodes_are_rejected(self):
        for mode,nodes in (("gpu04",{"gpu04"}),("gpu02",{"gpu01"}),("any",{"gpu04"})):
            with self.assertRaises(ValueError): submission.placement_args(nodes,mode)


if __name__ == "__main__":
    unittest.main()
