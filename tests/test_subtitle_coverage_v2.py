"""Coverage invariants and interruption safety for the new experiment."""
import importlib.util
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "experiments/subtitle_coverage_v2"
sys.path.insert(0, str(DIRECTORY))
from selection import protected_select, subtitle_targets
from evidencelab.pipeline import Decision
from evidencelab.longvideo_data import LongVideoSample
from evidencelab.store import Store, run_lock
from evidencelab.subtitle_anchors import select as old_select

spec = importlib.util.spec_from_file_location("subtitle_coverage_worker", DIRECTORY / "run.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class ProtectedSelectionTests(unittest.TestCase):
    def test_tail_regression_preserves_anchors_and_budget(self):
        base = list(range(0, 640, 10))
        record = {"offset": 0, "matches": [{"candidates": [(1, 1, 19, 21)]}]}
        old = old_select(base, 640, 1, record, 3)
        self.assertNotIn(630, old)
        targets = subtitle_targets(640, 1, record)
        selected, details = protected_select(base, targets, 640, list(range(640)), dict.fromkeys(base, .5))
        self.assertIn(630, selected)
        self.assertIn(0, selected)
        self.assertTrue(set(details["mandatory"]) <= set(selected))
        self.assertTrue(set(targets) <= set(selected))
        self.assertEqual(len(selected), 64)
        self.assertGreaterEqual(details["occupied_bins_after"], details["occupied_bins_before"])

    def test_local_replacement_removes_nearest_available_frame(self):
        base = list(range(0, 640, 10))
        selected, details = protected_select(base, [11], 640, list(range(640)), dict.fromkeys(base, .5))
        self.assertEqual(details["swaps"][0]["removed"], 10)
        self.assertEqual(details["swaps"][0]["reason"], "nearby-same-bin")
        self.assertIn(11, selected)

    def test_fallback_removes_low_score_in_redundant_bin(self):
        # The first bin contains only the protected first frame, so frame 1
        # must displace a redundant frame elsewhere rather than that endpoint.
        base = [0] + list(range(40, 630, 10)) + [639]
        values = dict.fromkeys(base, .8)
        values[110] = .01
        selected, details = protected_select(base, [1], 640, list(range(640)), values)
        self.assertEqual(details["swaps"][0]["reason"], "low-relevance-covered-bin")
        self.assertEqual(details["swaps"][0]["removed"], 110)
        self.assertIn(0, selected)
        self.assertIn(639, selected)

    def test_short_fully_protected_video_skips_unsafe_insertions(self):
        base = [0, 30, 60]
        selected, details = protected_select(base, [1, 2], 90, [0, 30, 60], dict.fromkeys(base, .5))
        self.assertEqual(selected, base)
        self.assertEqual(details["skipped_targets"], [1, 2])

    def test_proposals_match_old_selection_and_random_coverage_invariants(self):
        rng = random.Random(20261105)
        for _ in range(200):
            count = rng.randint(10, 4000)
            fps = rng.choice([.5, 1., 23.976, 30.])
            base = sorted(rng.sample(range(count), min(64, count)))
            records = [{"candidates": [(1, 1, t, t + .1) for t in
                        sorted(rng.uniform(0, count / fps) for _ in range(3))]} for _ in range(4)]
            record = {"offset": .2, "matches": records}
            targets = subtitle_targets(count, fps, record)
            self.assertEqual(old_select(base, count, fps, record, 3),
                             sorted((targets + [i for i in base if i not in targets])[:len(base)]))
            values = {i: rng.random() for i in base}
            selected, details = protected_select(base, targets, count, list(range(count)), values)
            self.assertEqual(len(selected), len(base))
            self.assertEqual(selected, sorted(set(selected)))
            self.assertTrue(set(details["mandatory"]) <= set(selected))
            bins = min(16, len(base))
            self.assertTrue({i * bins // count for i in base} <= {i * bins // count for i in selected})
            self.assertEqual(selected, protected_select(base, targets, count, list(range(count)), values)[0])

    def test_invalid_scores_and_no_match(self):
        base = [0, 10, 20]
        self.assertEqual(protected_select(base, [], 30, base, dict.fromkeys(base, 0))[0], base)
        for scores in ({}, {0: .5, 10: .5, 20: float("nan")}):
            with self.assertRaises(ValueError):
                protected_select(base, [1], 30, base, scores)


class ResumeTests(unittest.TestCase):
    def test_worker_entrypoint_pilot_then_resume_keeps_video_manifest_stable(self):
        import numpy as np
        import frames_full
        from evidencelab import backends, focus_adapter, hardware, longvideo, longvideo_data

        samples = [LongVideoSample(f"q{i}", "v.mp4", f"Question {i}", ("a", "b"), 1,
                                   "TAA", "600", 640.) for i in range(2)]
        base = list(range(0, 640, 10))
        plan = {s.id: {"id": s.id, "kind": "TAA", "offset": 0,
                      "matches": [{"candidates": [(1, 1, 19 + i * 80, 21 + i * 80)]}]}
                for i, s in enumerate(samples)}
        frozen = {a: {} for a in worker.ARMS[:-1]}
        for sample in samples:
            frozen["ridge_sigma2"][sample.id] = {"frames": base}
            frozen["eventcover_sigma2"][sample.id] = {"frames": base}
            frozen["multi_anchor_v1"][sample.id] = {"frames": old_select(base, 640, 1, plan[sample.id], 3)}
        calls = []

        class Backend:
            def __init__(self, config):
                pass

            def answer(self, question, images, timestamps):
                calls.append((question.id, timestamps))
                return Decision([.1, .9], 10)

        class Scorer:
            def __init__(self, config):
                pass

            def reset(self):
                pass

            def __call__(self, video, query, indices):
                return [.5] * len(indices)

        class Cache:
            def __init__(self, indices):
                self.manifest = {"indices": sorted(indices), "source_frames": 640, "fps": 1.}

            def __getitem__(self, index):
                return SimpleNamespace(asnumpy=lambda: np.zeros((2, 2, 3), dtype=np.uint8))

        class Native:
            def __init__(self, *args, **kwargs):
                pass

            def __len__(self):
                return 640

            def get_avg_fps(self):
                return 1.

        reader = SimpleNamespace(clear_cache=lambda: None)
        decord = SimpleNamespace(VideoReader=Native, cpu=lambda x: None)
        torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))

        def build_index(reader, path, needed):
            with run_lock(path.parent):
                return {"videos/v.mp4": {}}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            args = SimpleNamespace(preflight=False, output=path / "run", ram=path / "ram",
                                   dtype="auto", max_new_questions=1, session_minutes=10)
            with (patch.object(worker, "prepare", return_value=(samples, frozen, plan, {})),
                  patch.dict(sys.modules, {"decord": decord, "torch": torch}),
                  patch.object(worker.subprocess, "check_output", return_value="tmpfs"),
                  patch.object(hardware, "execution_profile", return_value={"resolved_dtype": "float32"}),
                  patch.object(longvideo, "longvideo_runtime", return_value={}),
                  patch.object(backends, "HFBackend", Backend),
                  patch.object(focus_adapter, "BlipITMScorer", Scorer),
                  patch.object(longvideo_data, "HTTPRangeSource", return_value=None),
                  patch.object(longvideo_data, "MultipartReader", return_value=reader),
                  patch.object(longvideo_data, "archive_index", side_effect=build_index),
                  patch.object(longvideo_data, "fetch_video", return_value="video-hash"),
                  patch.object(longvideo_data, "check_space"),
                  patch.object(frames_full, "build", side_effect=lambda p, r, c, f, indices, *a: Cache(indices))):
                self.assertEqual(worker.run(args), 75)
                first_calls = len(calls)
                self.assertGreater(first_calls, 0)
                manifest = next((args.output / "pixels").glob("*.json")).read_bytes()
                args.max_new_questions = 0
                self.assertEqual(worker.run(args), 75)
                self.assertEqual(next((args.output / "pixels").glob("*.json")).read_bytes(), manifest)
                self.assertEqual(len({qid for qid, _ in calls}), 2)
                self.assertEqual(sum(qid == calls[0][0] for qid, _ in calls), first_calls)
                final_calls = len(calls)
                self.assertEqual(worker.run(args), 75)
                self.assertEqual(len(calls), final_calls)

    def test_interruption_reuses_only_same_journal_and_rejects_changed_frames(self):
        sample = SimpleNamespace(id="q", video="v.mp4", question_type="TAA", answer=1, choices=("a", "b"))
        sets = dict(zip(worker.ARMS, ([0, 2], [0, 2], [0, 3], [0, 4])))
        calls = []

        def answer(frames):
            calls.append(frames)
            return Decision([.1, .9], 10)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            store = Store(path, {"model": "frozen"}, False)
            with self.assertRaises(InterruptedError):
                worker.commit_arms(store, sample, sets, "pixels", {}, answer, lambda: len(store.done()) == 2)
            store.close()
            store = Store(path, {"model": "frozen"}, True)
            worker.commit_arms(store, sample, sets, "pixels", {}, answer, lambda: False)
            self.assertEqual(len(calls), 3)
            self.assertEqual(len(store.results()), 4)
            self.assertTrue(all(r["correct"] for r in store.results()))
            with self.assertRaises(ValueError):
                worker.commit_arms(store, sample, sets, "different-pixels", {}, answer, lambda: False)
            changed = {**sets, "protected_v2": [0, 5]}
            with self.assertRaises(ValueError):
                worker.commit_arms(store, sample, changed, "pixels", {}, answer, lambda: False)
            report = worker.export(path, store)
            self.assertFalse(report["complete"])
            self.assertEqual(report["fresh_answer_forwards"], 3)
            self.assertEqual(report["protected_v2_vs"]["multi_anchor_v1"]["paired_n"], 1)
            store.close()
            with self.assertRaises(ValueError):
                Store(path, {"model": "changed"}, True)

    def test_nonfinite_model_output_is_not_committed(self):
        sample = SimpleNamespace(id="q", video="v", question_type="TAA", answer=1, choices=("a", "b"))
        sets = dict.fromkeys(worker.ARMS, [0, 2])
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory), {}, False)
            with self.assertRaises(ValueError):
                worker.commit_arms(store, sample, sets, "pixels", {},
                                   lambda frames: Decision([float("nan"), .9], 1), lambda: False)
            self.assertEqual(store.results(), [])
            store.close()


if __name__ == "__main__":
    unittest.main()
