import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

from evidencelab.backends import MockBackend, MockScorer, molmo_metadata
from evidencelab.config import Config
from evidencelab.data import Question, Sample, choose_subset, digest, file_hash, load_manifest, prepare
from evidencelab.metrics import compare
from evidencelab.pipeline import Decision, infer
from evidencelab.runner import run
from evidencelab.selectors import evidence_frames, expand_uniform, uniform
from evidencelab.store import Store, read_run, run_lock


def sample(i=0):
    return Sample(f"q{i}", "nextqa", "val", f"v{i//2}.mp4", digest(i//2),
                  "What happens?", ("a", "b", "c", "d", "e"), i % 5, "TN")


class DataTests(unittest.TestCase):
    def test_star_clip_choices_and_no_graph_leakage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "abc.mp4").write_bytes(b"fixture")
            row = {"question_id": "Interaction_T1_1", "video_id": "abc", "start": 1.2, "end": 3.4,
                   "question": "Which?", "answer": "yes", "situations": {"secret": "ground truth"},
                   "choices": [{"choice_id": 1, "choice": "no"}, {"choice_id": 0, "choice": "yes"}]}
            annotations, out = root / "star.json", root / "manifest.jsonl"
            annotations.write_text(json.dumps([row]))
            self.assertEqual(prepare("star", annotations, root, "val", out), 1)
            s = load_manifest(out)[0]
            self.assertEqual((s.start, s.end, s.answer, s.choices), (1.2, 3.4, 0, ("yes", "no")))
            self.assertNotIn("situations", out.read_text())
            self.assertFalse(hasattr(s.public_question(), "answer"))

    def test_nextqa_csv_mapping_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "mapped.mp4").write_bytes(b"fixture")
            row = {"video": "123", "qid": "1", "question": "Why?", "answer": "2", "type": "CW",
                   **{f"a{i}": str(i) for i in range(5)}}
            annotations = root / "val.csv"
            with annotations.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            mapping = root / "map.json"
            mapping.write_text(json.dumps({"123": "dir/mapped"}))
            out = root / "manifest.jsonl"
            prepare("nextqa", annotations, root, "val", out, mapping)
            self.assertEqual(load_manifest(out)[0].answer, 2)
            out.write_text(out.read_text() * 2)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_manifest(out)

    def test_path_validation_and_subset_order(self):
        for path in ("../escape.mp4", "/absolute.mp4", "C:\\secret.mp4"):
            with self.assertRaises(ValueError):
                replace(sample(), video=path)
        rows = [sample(i) for i in range(40)]
        self.assertEqual(choose_subset(rows, 10, 18), choose_subset(rows[::-1], 10, 18))
        self.assertEqual(len(choose_subset(rows, 0, 18)), 40)


class SelectorTests(unittest.TestCase):
    def test_uniform_and_expansion_budgets(self):
        for n in range(1, 70):
            for k in (1, 8, 16, 32):
                ids = uniform(n, k)
                self.assertEqual(len(ids), min(n, k))
                self.assertEqual(ids, sorted(set(ids)))
                expanded = expand_uniform(n, ids, max(k, 32))
                self.assertTrue(set(ids) <= set(expanded))
                self.assertEqual(len(expanded), min(n, max(k, 32)))

    def test_evidence_keeps_context_budget_and_order(self):
        times = [i * .25 for i in range(64)]
        initial = uniform(64, 8)
        a = [0.] * 64
        b = [0.] * 64
        a[20], b[44] = 1., 1.
        ids, anchors = evidence_frames(times, initial, 16, [0.] * 64, [a, b], (0, 1), 1., .25)
        self.assertEqual(len(ids), 16)
        self.assertTrue(set(initial) <= set(ids))
        self.assertEqual(anchors[:2], [20, 44])
        self.assertTrue({16, 20, 24, 40, 44, 48} <= set(ids))
        self.assertEqual(ids, sorted(set(ids)))

    def test_gating_and_compute_control(self):
        class Fixed:
            def __init__(self, scores):
                self.scores = scores
            def answer(self, q, frames, times):
                assert not hasattr(q, "answer")
                return Decision(self.scores, len(frames) * 10)
        q = Question("q", "What?", ("A", "B"))
        cfg = Config(backend="mock", method="evidence")
        frames, times = list(range(64)), [i*.25 for i in range(64)]
        confident = infer(q, frames, times, cfg, Fixed([.9, .1]), MockScorer())
        self.assertEqual(len(confident["calls"]), 1)
        uncertain = infer(q, frames, times, cfg, Fixed([.51, .49]), MockScorer())
        control = infer(q, frames, times, replace(cfg, method="uniform_refine"), Fixed([.51, .49]))
        self.assertEqual(uncertain["frame_presentations"], 24)
        self.assertEqual(control["frame_presentations"], 24)
        self.assertEqual(len(uncertain["calls"]), 2)
        self.assertEqual(control["selector_queries"], 0)

    def test_invalid_backend_probabilities_fail(self):
        class Invalid:
            def answer(self, *args):
                return Decision([float("nan"), .4], 1)
        with self.assertRaises(ValueError):
            infer(Question("q", "Q", ("a", "b")), [0], [0.], Config(backend="mock"), Invalid())

    def test_virtual_timestamp_metadata_preserves_irregular_times(self):
        times = [11.12, 11.58, 15.4]
        meta = molmo_metadata(times)
        self.assertEqual([x / meta["fps"] for x in meta["frames_indices"]], times)


class ResumeTests(unittest.TestCase):
    def test_cli_exit_code_for_partial_and_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, cfg = root / "manifest.jsonl", root / "config.json"
            manifest.write_text("".join(json.dumps(asdict(sample(i)))+"\n" for i in range(3)))
            cfg.write_text(json.dumps(asdict(Config(backend="mock", limit=0))))
            cmd = [sys.executable, "-m", "evidencelab", "run", "--config", str(cfg),
                   "--manifest", str(manifest), "--video-root", str(root), "--output", str(root / "run")]
            partial = subprocess.run(cmd + ["--max-new-samples", "1"], capture_output=True, text=True)
            self.assertEqual(partial.returncode, 75, partial.stderr)
            complete = subprocess.run(cmd + ["--resume"], capture_output=True, text=True)
            self.assertEqual(complete.returncode, 0, complete.stderr)

    def test_resume_matches_uninterrupted_and_contract_rejects_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "data.jsonl"
            manifest.write_text("".join(json.dumps(asdict(sample(i)))+"\n" for i in range(12)))
            config = Config(backend="mock", method="evidence", margin_threshold=1, limit=0)
            left, right = root / "whole", root / "resume"
            self.assertEqual(run(config, manifest, root, left), 0)
            self.assertEqual(run(config, manifest, root, right, max_new_samples=3), 75)
            self.assertEqual(len(read_run(right)[1]), 3)
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                run(replace(config, margin_threshold=.5), manifest, root, right, resume=True)
            self.assertEqual(run(config, manifest, root, right, resume=True), 0)
            lrows, rrows = read_run(left)[1], read_run(right)[1]
            for rows in (lrows, rrows):
                for row in rows:
                    row.pop("elapsed_seconds")
            self.assertEqual(lrows, rrows)
            self.assertEqual(compare(left, right, 100)["delta_accuracy_right_minus_left"], 0)
            with self.assertRaises(FileExistsError):
                run(config, manifest, root, right)

    def test_duplicate_results_transaction_and_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with run_lock(root):
                with self.assertRaises((OSError, BlockingIOError)):
                    with run_lock(root):
                        pass
                store = Store(root, {"test": 1}, False)
                store.put({"id": "q", "value": 7})
                with self.assertRaises(Exception):
                    store.put({"id": "q", "value": 9})
                self.assertEqual(store.results(), [{"id": "q", "value": 7}])
                store.close()


class ArchiveTests(unittest.TestCase):
    def test_archive_traversal_rejected(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "stage_data.py"
        spec = importlib.util.spec_from_file_location("stage_data", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with zipfile.ZipFile(root / "bad.zip", "w") as z:
                z.writestr("../outside.mp4", b"escape")
            with self.assertRaises(ValueError):
                module.extract(root / "bad.zip", root / "out")
            self.assertFalse((root / "outside.mp4").exists())


if __name__ == "__main__":
    unittest.main()
