import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from evidencelab.data import Question
from evidencelab.longvideo_config import LongVideoConfig
from evidencelab.ridge_adapter import load_ridge, select_ridge, selector_args, validate_weights

UPSTREAM = Path(os.environ.get("RIDGE_UPSTREAM_DIR", Path(__file__).resolve().parents[2] / "upstream-ridge"))
WEIGHTS = dict(zip(("peak_similarity", "slope_abs", "rising_slope", "falling_slope",
                   "boundary_change", "context_density"), (8, 6, 5, 7, 4, 3)))


class RidgeSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.author, cls.weight_author = load_ridge(UPSTREAM)

    def test_adapter_preserves_author_selection_on_multiple_curves_and_grids(self):
        rng = np.random.default_rng(42)
        class Scorer:
            def reset(self): self.evaluated, self.calls = [], 0
            def __call__(self, video, query, ids):
                self.evaluated.extend(ids)
                self.calls += 1
                self.query = query
                if len(ids) > 4: raise AssertionError("Unbounded scoring")
                return [video.scores[i // video.step] for i in ids]
        cases = 0
        for fps in (30., 29.97):
            step = int(fps)
            for n in (1, 3, 5, 32, 100, 300):
                for k in (8, 64):
                    for adaptive in (False, True):
                        config = LongVideoConfig(method="ridge", frames=k, ridge_adaptive_sigma=adaptive)
                        values = rng.random(n).tolist() if n % 2 else [0.5] * n
                        class Video:
                            scores = values
                            def __len__(self): return n * step
                            def get_avg_fps(self): return fps
                        video, scorer = Video(), Scorer()
                        video.step = step
                        expected = self.author.gradselect(values, k, WEIGHTS, selector_args(config))
                        indices, details = select_ridge(video, Question("q", "Public question", ("secret option", "b")),
                            config, scorer, self.author, {"weights": WEIGHTS, "record_sha256": "a"*64,
                                "generation_seconds": 2., "input_tokens": 120, "output_tokens": 40})
                        self.assertEqual(indices, [int(i) * step for i in expected])
                        self.assertEqual(len(indices), min(k, n))
                        self.assertEqual(scorer.query, "Public question")
                        self.assertEqual(details["scorer_frame_evaluations"], n)
                        self.assertFalse(details["ridge_fallback"])
                        self.assertEqual(details["auxiliary_input_tokens"], 120)
                        cases += 1
        self.assertEqual(cases, 48)

    def test_invalid_scores_and_interrupt_do_not_fall_back(self):
        class Video:
            def __len__(self): return 300
            def get_avg_fps(self): return 30.
        class Scorer:
            def reset(self): self.evaluated, self.calls = [], 0
            def __call__(self, *args): return [float("nan")]*4
        config = LongVideoConfig(method="ridge")
        record = {"weights": WEIGHTS}
        with self.assertRaisesRegex(ValueError, "BLIP scores"):
            select_ridge(Video(), Question("q", "q", ("a", "b")), config, Scorer(), self.author, record)
        with self.assertRaises(InterruptedError):
            select_ridge(Video(), Question("q", "q", ("a", "b")), config, Scorer(), self.author, record, stop=lambda: True)

    def test_source_checksum_rejects_modified_algorithm_or_prompt(self):
        for file in ("frame_select.py", "weight.py"):
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp)
                for name in ("frame_select.py", "weight.py"):
                    (target / name).write_bytes((UPSTREAM / name).read_bytes())
                with (target / file).open("ab") as f:
                    f.write(b"\n# changed\n")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    load_ridge(target)

    def test_weights_reject_missing_invalid_and_default_like_values(self):
        for wrong in ({}, {**WEIGHTS, "answer": 1}, {**WEIGHTS, "peak_similarity": True},
                      {**WEIGHTS, "peak_similarity": 11}, {**WEIGHTS, "peak_similarity": 2.5}):
            with self.assertRaises(ValueError): validate_weights(wrong)

    def test_config_pins_and_suite_create_separate_runs(self):
        for kwargs in ({"ridge_sigma": float("nan")}, {"ridge_adaptive_sigma": 1},
                       {"ridge_weight_revision": "main"}, {"ridge_weight_model_id": "other/model"}):
            with self.assertRaises(ValueError): LongVideoConfig(method="ridge", **kwargs)
        path = Path(__file__).resolve().parents[1] / "scripts/slurm/submit_longvideo_suite.py"
        spec = importlib.util.spec_from_file_location("ridge_suite_test", path)
        suite = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(suite)
        with tempfile.TemporaryDirectory() as tmp:
            plans = suite.plans(Path(tmp), "molmo2", 64, ["ridge", "uniform"], "ridge-v1", False)
            self.assertEqual(len({item[1] for item in plans}), 2)
            self.assertIn("ridge64-200-ridge-v1", plans[0][1].name)
            full = suite.plans(Path(tmp), "molmo2", 64, ["ridge", "uniform"], "ridge-v1", False, limit=0)
            self.assertTrue(all("-full-" in item[1].name for item in full))


if __name__ == "__main__":
    unittest.main()
