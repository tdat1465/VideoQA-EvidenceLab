from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from evidencelab.data import Question, digest
from evidencelab.longvideo import run
from evidencelab.longvideo_config import LongVideoConfig
from evidencelab.longvideo_data import LongVideoSample
from evidencelab.pipeline import Decision
from evidencelab.ridge_adapter import load_ridge
from evidencelab.ridge_weights import RidgeWeightCache, prepare_weights, weight_contract

UPSTREAM = Path(os.environ.get("RIDGE_UPSTREAM_DIR", Path(__file__).resolve().parents[2] / "upstream-ridge"))
WEIGHTS = dict(zip(("peak_similarity", "slope_abs", "rising_slope", "falling_slope",
                   "boundary_change", "context_density"), (8, 6, 5, 7, 4, 3)))


class RidgeWeightTests(unittest.TestCase):
    def setUp(self):
        self.questions = [Question(f"q{i}", f"Question {i}", ("a", "b")) for i in range(3)]

    def test_partial_weight_generation_is_durable_and_resume_skips_finished(self):
        calls, closed = [], []
        class Generator:
            def __call__(self, q):
                self_test.assertFalse(hasattr(q, "answer"))
                calls.append(q.id)
                if q.id == "q1" and len(calls) == 2: raise InterruptedError("lost session")
                return WEIGHTS, 100, 40, 1.5, "a"*64
            def close(self): closed.append(True)
        self_test = self
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"weights.json"
            cache = RidgeWeightCache(path, {"revision": "a"*40}, self.questions)
            with self.assertRaises(InterruptedError): prepare_weights(cache, self.questions, Generator)
            restored = RidgeWeightCache(path, {"revision": "a"*40}, self.questions)
            self.assertEqual(set(restored.records), {"q0"})
            result = prepare_weights(restored, self.questions, Generator)
            self.assertEqual(result["new_weights"], 2)
            self.assertEqual(calls, ["q0", "q1", "q1", "q2"])
            self.assertEqual(len(closed), 2)
            self.assertEqual(prepare_weights(restored, self.questions, lambda: self.fail("Unnecessary load"))["new_weights"], 0)

    def test_cache_rejects_config_question_checksum_and_completed_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"weights.json"
            cache = RidgeWeightCache(path, {"revision": "a"*40}, self.questions)
            record = cache.add(self.questions[0], WEIGHTS, 100, 40, 1., "a"*64)
            cache.check_completed([{"id": "q0", "ridge_weight_record_sha256": record["record_sha256"]}])
            with self.assertRaisesRegex(ValueError, "completed results"):
                cache.check_completed([{"id": "q0", "ridge_weight_record_sha256": "b"*64}])
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                RidgeWeightCache(path, {"revision": "b"*40}, self.questions)
            changed = [Question("q0", "Different question", ("a", "b")), *self.questions[1:]]
            with self.assertRaisesRegex(ValueError, "question/ID"):
                RidgeWeightCache(path, {"revision": "a"*40}, changed)
            raw = json.loads(path.read_text())
            raw["records"]["q0"]["weights"]["peak_similarity"] = 1
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "checksum"):
                RidgeWeightCache(path, {"revision": "a"*40}, self.questions)

    def test_invalid_generator_weights_fail_without_saving_and_release_model(self):
        closed = []
        class Generator:
            def __call__(self, q): return {}, 100, 40, 1., "a"*64
            def close(self): closed.append(True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"weights.json"
            cache = RidgeWeightCache(path, {}, self.questions)
            with self.assertRaises(ValueError): prepare_weights(cache, self.questions, Generator)
            self.assertFalse(path.exists())
            self.assertEqual(closed, [True])

    def test_stop_before_loading_and_prompt_is_part_of_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = RidgeWeightCache(Path(tmp)/"weights.json", {}, self.questions)
            with self.assertRaises(InterruptedError):
                prepare_weights(cache, self.questions, lambda: self.fail("Model loaded after stop"), stop=lambda: True)
        _, author = load_ridge(UPSTREAM)
        base = {"resolved_dtype": "bfloat16", "source_sha256": "a"*64, "runtime": {},
                "annotation_sha256": "b"*64, "selected_ids_hash": "c"*64}
        contract = weight_contract(LongVideoConfig(method="ridge"), base, author)
        with patch.object(author, "SYSTEM_PROMPT", "changed"):
            self.assertNotEqual(contract, weight_contract(LongVideoConfig(method="ridge"), base, author))
        self.assertEqual(contract["query"], "question-only-no-labels-v1")


class RidgeRunnerTests(unittest.TestCase):
    def test_real_runner_selection_and_weights_resume_with_fake_gpu_backend(self):
        order = []
        samples = [LongVideoSample(f"q{i}", "v.mp4", f"Public question {i}", ("a", "b"), i%2,
                                  "S2E", "60", 10.) for i in range(3)]
        author, weight_author = load_ridge(UPSTREAM)
        class Video:
            def __len__(self): return 300
            def get_avg_fps(self): return 30.
            def __getitem__(self, index):
                return SimpleNamespace(asnumpy=lambda: np.full((8, 8, 3), index % 255, np.uint8))
        class Generator:
            def __init__(self, *args): order.append("qwen-load")
            def __call__(self, q):
                if hasattr(q, "answer"): raise AssertionError("Label leakage")
                return WEIGHTS, 100, 40, 1., digest(q.question)
            def close(self): order.append("qwen-close")
        class Scorer:
            def reset(self): self.evaluated, self.calls = [], 0
            def __call__(self, video, query, ids):
                if not query.startswith("Public question"): raise AssertionError("Wrong scorer query")
                self.evaluated.extend(ids)
                self.calls += 1
                return [i / 300 for i in ids]
        class Backend:
            torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
            def __init__(self, config): order.append("molmo-load")
            def reset_peak(self): pass
            def measurements(self): return {"peak_vram_allocated_gib": None}
            def answer(self, q, frames, times):
                if hasattr(q, "answer") or len(frames) != 8: raise AssertionError("Bad answer evidence")
                if times != sorted(set(times)): raise AssertionError("Source timeline mapping failed")
                return Decision([.8, .2], 20)
        def download(reader, entry, path, **kwargs):
            path.write_bytes(b"fake video")
            return "a"*64
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            annotations, index = root/"val.json", root/"index.json"
            annotations.write_text("[]")
            index.write_text("{}")
            stack.enter_context(patch.dict("sys.modules", {"decord": SimpleNamespace(
                bridge=SimpleNamespace(set_bridge=lambda _: None), cpu=lambda _: None,
                VideoReader=lambda *a, **kw: Video())}))
            patches = {"parse_annotations": lambda _: [samples[0]]*1337,
                "choose_subset": lambda *a: samples, "validate_ram_root": lambda _: None,
                "longvideo_runtime": lambda _: {}, "execution_profile": lambda _: {"resolved_dtype": "bfloat16", "node": "fake"},
                "HTTPRangeSource": lambda **kw: SimpleNamespace(bytes_read=0),
                "archive_index": lambda *a: {"videos/v.mp4": [0, 5]},
                "HFBackend": Backend, "BlipITMScorer": lambda _: Scorer(), "check_space": lambda *a: None,
                "fetch_video": download, "fetch_ridge": lambda _: UPSTREAM,
                "QwenWeightGenerator": Generator}
            for name, value in patches.items():
                stack.enter_context(patch("evidencelab.longvideo." + name, value))
            config = LongVideoConfig(method="ridge", frames=8, limit=3)
            args = (config, annotations, root/"ram", root/"run", index)
            self.assertEqual(run(*args, max_new_samples=1), 75)
            self.assertEqual(run(*args, resume=True, max_new_samples=1), 75)
            self.assertEqual(run(*args, resume=True), 0)
            summary = json.loads((root/"run/summary.json").read_text())
            results = [json.loads(row) for row in (root/"run/predictions.jsonl").read_text().splitlines()]
            contract = json.loads((root/"run/contract.json").read_text())
            self.assertEqual(summary["completed_count"], 3)
            self.assertTrue(summary["complete"])
            self.assertEqual([r["id"] for r in results], ["q0", "q1", "q2"])
            self.assertTrue(all(not r["ridge_fallback"] for r in results))
            self.assertTrue(all(r["input_tokens"] == 120 for r in results))
            self.assertTrue(all(r["scorer_frame_evaluations"] == 10 for r in results))
            self.assertIn("ridge_revision", contract)
            self.assertFalse((root/"ram/active-video.mp4").exists())
            for i in range(0, len(order), 3):
                self.assertEqual(order[i:i+3], ["qwen-load", "qwen-close", "molmo-load"])


if __name__ == "__main__":
    unittest.main()
