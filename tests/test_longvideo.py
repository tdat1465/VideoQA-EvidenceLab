import io
import json
import math
import sys
import tarfile
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from evidencelab.data import Question, digest
from evidencelab.focus_adapter import blip_caption, load_focus, question_seed, select_focus
from evidencelab.longvideo import run
from evidencelab.longvideo_config import LongVideoConfig
from evidencelab.longvideo_data import (HTTPRangeSource, MultipartReader, archive_index, cgroup_memory,
                                      check_space, fetch_video, parse_annotations)
from evidencelab.metrics import compare
from evidencelab.store import read_run


def annotation(i=0):
    return {"id": f"q{i}", "video_path": f"v{i//2}.mp4", "question": "What happens?",
            "candidates": ["First", "Second", "Third"], "correct_choice": i % 3,
            "question_category": "S2E", "duration_group": 2, "duration": 75.0,
            "subtitle_path": "do-not-use.json", "referred_timestamp": [25, 30]}


class MemorySource:
    def __init__(self, chunks):
        self.chunks, self.requests = chunks, []

    def read(self, name, offset, length, total):
        self.requests.append((name, offset, length))
        assert total == len(self.chunks[name])
        return self.chunks[name][offset:offset+length]


class LongVideoDataTests(unittest.TestCase):
    def test_cgroup_v1_memory_includes_job_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proc, mount = root / "proc", root / "cgroup"
            job = mount / "memory/slurm/job123"
            job.mkdir(parents=True)
            proc.write_text("5:cpu,cpuacct:/slurm/job123\n6:memory:/slurm/job123\n")
            (job / "memory.usage_in_bytes").write_text(str(50 * 1024**3))
            (job / "memory.limit_in_bytes").write_text(str(90 * 1024**3))
            (job / "memory.max_usage_in_bytes").write_text(str(60 * 1024**3))
            with patch.dict("os.environ", {"SLURM_MEM_PER_NODE": str(90*1024)}):
                self.assertEqual(cgroup_memory(proc, mount),
                                 {"current_bytes": 50*1024**3, "limit_bytes": 90*1024**3,
                                  "peak_bytes": 60*1024**3})

    def test_stream_does_not_prefetch_next_video_and_clears_cache(self):
        source = MemorySource({"part": b"before" + b"CURRENT" + b"do not download next video"})
        reader = MultipartReader(source, [("part", len(source.chunks["part"]))], block_size=1024)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "active.mp4"
            fetch_video(reader, [6, 7], path, guard=False)
            self.assertEqual(path.read_bytes(), b"CURRENT")
            self.assertEqual(source.requests, [("part", 6, 7)])
            self.assertEqual(reader.cached_data, b"")
            with self.assertRaises(FileExistsError):
                fetch_video(reader, [6, 7], path, guard=False)

    def test_ram_pressure_during_transfer_aborts_and_removes_partial(self):
        source = MemorySource({"part": b"a" * 24})
        reader = MultipartReader(source, [("part", 24)], block_size=8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "active.mp4"
            # Start with room; after the first block another allocation in this
            # job consumes headroom. Do not read or retain the remaining blocks.
            with patch("evidencelab.longvideo_data.TRANSFER_CHUNK_BYTES", 8), \
                 patch("evidencelab.longvideo_data.check_space",
                       side_effect=[{}, {}, MemoryError("RAM pressure")]) as guard:
                with self.assertRaisesRegex(MemoryError, "RAM pressure"):
                    fetch_video(reader, [0, 24], path)
            self.assertEqual([call.args[1] for call in guard.call_args_list], [24, 24, 16])
            self.assertEqual(len(source.requests), 1)
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix(".partial").exists())
            self.assertEqual(reader.cached_data, b"")

    def test_memory_guard_reserves_ram_for_model_and_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("evidencelab.longvideo_data.shutil.disk_usage",
                       return_value=types.SimpleNamespace(free=500*1024**3)), \
                 patch("evidencelab.longvideo_data.cgroup_memory",
                       return_value={"current_bytes": 83*1024**3, "limit_bytes": 90*1024**3}):
                check_space(Path(tmp), 1024**3)
                with self.assertRaisesRegex(MemoryError, "cgroup"):
                    check_space(Path(tmp), 2*1024**3)
                with self.assertRaisesRegex(MemoryError, "8 GiB"):
                    check_space(Path(tmp), 9*1024**3)

    def test_cgroup_reports_job_limit_never_unlimited_node_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proc, mount = root / "proc", root / "cgroup"
            leaf = mount / "job/step"
            leaf.mkdir(parents=True)
            proc.write_text("0::/job/step\n")
            for path in (mount, mount / "job", leaf):
                (path / "memory.current").write_text("100")
                (path / "memory.max").write_text("max")
                (path / "memory.peak").write_text("200")
            with patch.dict("os.environ", {"SLURM_MEM_PER_NODE": str(90*1024)}):
                self.assertEqual(cgroup_memory(proc, mount), {})
                (mount / "memory.max").write_text(str(1024**4))
                self.assertEqual(cgroup_memory(proc, mount), {})
                (mount / "job/memory.max").write_text(str(90*1024**3))
                self.assertEqual(cgroup_memory(proc, mount),
                                 {"current_bytes": 100, "limit_bytes": 90*1024**3, "peak_bytes": 200})

    def test_annotation_validation_and_public_question(self):
        sample = parse_annotations([annotation()])[0]
        self.assertEqual(sample.choices, ("First", "Second", "Third"))
        self.assertEqual(sample.answer, 0)
        public = sample.public_question()
        self.assertEqual(vars(public), {"id": sample.id, "question": sample.question, "choices": sample.choices})
        for override in ({"correct_choice": -1}, {"correct_choice": "@"}, {"video_path": "../escape"},
                         {"duration": float("nan")}, {"candidates": ["one"]}):
            with self.assertRaises(ValueError):
                parse_annotations([{**annotation(), **override}])
        with self.assertRaises(ValueError):
            parse_annotations([annotation(), annotation()])

    def test_multipart_tar_pax_index_and_exact_file_hash(self):
        payload = b"video bytes" * 513
        long_name = "a" * 140 + ".mp4"
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as tar:
            info = tarfile.TarInfo("videos/" + long_name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        raw = stream.getvalue()
        # Force both PAX headers and video data to cross multipart boundaries.
        parts = [(f"part{i}", raw[i*777:(i+1)*777]) for i in range(math.ceil(len(raw)/777))]
        source = MemorySource(dict(parts))
        reader = MultipartReader(source, [(n, len(b)) for n, b in parts], block_size=257)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "index/index.json"
            members = archive_index(reader, path, {long_name})
            video_path = root / "current.mp4"
            import hashlib
            self.assertEqual(fetch_video(reader, members["videos/"+long_name], video_path, guard=False),
                             hashlib.sha256(payload).hexdigest())
            self.assertEqual(video_path.read_bytes(), payload)
            n = len(source.requests)
            self.assertEqual(archive_index(reader, path, {long_name}), members)
            self.assertEqual(len(source.requests), n)  # Persistent index does not reread archive.
            with self.assertRaises(ValueError):
                reader.read()
            self.assertTrue(all(size <= 257 for _, _, size in source.requests))
            value = json.loads(path.read_text())
            value["members"]["videos/"+long_name][0] += 512
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "Corrupt"):
                archive_index(reader, path, {long_name})

    def test_range_rejects_ignored_range_wrong_offset_and_truncation(self):
        class Response:
            def __init__(self, code, content_range, body):
                self.status_code = code
                self.headers = {"Content-Range": content_range}
                self.raw = io.BytesIO(body)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        class Session:
            def __init__(self, response):
                self.response = response
            def get(self, *args, **kwargs):
                self.kwargs = kwargs
                return self.response
        for code, content_range, body in [(200, "", b"large"), (206, "bytes 1-3/10", b"abc"),
                                           (206, "bytes 2-4/10", b"ab"), (401, "", b"")]:
            source = HTTPRangeSource(session=Session(Response(code, content_range, body)))
            with patch("evidencelab.longvideo_data.time.sleep"), self.assertRaises(Exception):
                source.read("x", 2, 3, 10)
        session = Session(Response(206, "bytes 2-4/10", b"abc"))
        source = HTTPRangeSource(session=session)
        self.assertEqual(source.read("x", 2, 3, 10), b"abc")
        self.assertTrue(session.kwargs["stream"])
        self.assertEqual(source.bytes_read, 3)

    def test_cancel_removes_partial_download(self):
        source = MemorySource({"p": b"12345"})
        reader = MultipartReader(source, [("p", 5)])
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "video.mp4"
            with self.assertRaises(InterruptedError):
                fetch_video(reader, [0, 5], target, stop=lambda: True, guard=False)
            self.assertFalse(target.exists())
            self.assertFalse(target.with_suffix(".partial").exists())


class FocusTests(unittest.TestCase):
    def test_real_pinned_algorithm_deterministic_full_video_and_no_labels(self):
        import numpy  # Preload outside patch.dict, so restoring sys.modules keeps dependencies.
        import scipy.interpolate
        # decord is only a type annotation in the unmodified selector. Synthetic
        # tests need no decoder; production never replaces this import.
        stub = types.ModuleType("decord")
        stub.VideoReader = object
        with patch.dict(sys.modules, {"decord": stub}):
            focus = load_focus(Path(__file__).parents[1] / "src/evidencelab/_vendor/focus.py")
        class Video:
            def __len__(self):
                return 18000
            def get_avg_fps(self):
                return 30.
        class Scorer:
            def reset(self):
                self.evaluated, self.calls = [], 0
            def __call__(self, video, query, ids):
                self.evaluated.extend(ids)
                self.calls += 1
                return [.5 + .49 * math.sin(i / 100) for i in ids]
        question = Question("q123", "What happens?", ("a", "b"))
        config = LongVideoConfig()
        a = select_focus(Video(), question, config, Scorer(), focus)
        b = select_focus(Video(), question, config, Scorer(), focus)
        self.assertEqual(a, b)
        self.assertEqual(len(a[0]), 64)
        self.assertGreater(max(a[0]), 64)  # Indices are in the original full video.
        self.assertGreater(a[1]["scorer_frame_evaluations"], 64)
        json.dumps(a, allow_nan=False)
        self.assertNotEqual(question_seed(42, "q123"), question_seed(42, "q456"))
        self.assertEqual(blip_caption('HELLO!   (World).'), 'hello world')

    def test_every_shipped_config_valid(self):
        configs = list((Path(__file__).parents[1] / "configs").glob("lvb_*.json"))
        self.assertEqual(len(configs), 24)
        for path in configs:
            self.assertIn(LongVideoConfig.load(path).frames, (8, 64))


class LongVideoResumeTests(unittest.TestCase):
    def test_pilot_releases_active_video_before_export(self):
        from evidencelab.backends import MockBackend
        from evidencelab.metrics import export
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotation_path = root / "val.json"
            annotation_path.write_text(json.dumps([annotation(i) for i in range(3)]))
            config = LongVideoConfig(backend="mock", method="uniform", limit=0)
            active = root / "ram/active-video.mp4"
            original = MockBackend.answer
            def answer(self, *args):
                active.write_bytes(b"temporary video")
                return original(self, *args)
            def report(*args):
                self.assertFalse(active.exists())
                return export(*args)
            with patch.object(MockBackend, "answer", answer), patch("evidencelab.longvideo.export", report):
                self.assertEqual(run(config, annotation_path, root / "ram", root / "run",
                                     root / "index.json", max_new_samples=1), 75)

    def test_resume_pairing_and_reject_model_config_or_data_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotation_path = root / "val.json"
            annotation_path.write_text(json.dumps([annotation(i) for i in range(6)]))
            config = LongVideoConfig(backend="mock", method="uniform", limit=0)
            args = (annotation_path, root / "ram")
            index = root / "metadata/index.json"
            self.assertEqual(run(config, *args, root / "whole", index), 0)
            self.assertEqual(run(config, *args, root / "resume", index, max_new_samples=2), 75)
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                run(replace(config, frames=32), *args, root / "resume", index, resume=True)
            self.assertEqual(run(config, *args, root / "resume", index, resume=True), 0)
            self.assertEqual(compare(root / "whole", root / "resume", 100)["delta_accuracy_right_minus_left"], 0)
            contract, rows = read_run(root / "whole")
            self.assertEqual(contract["selected_count"], 6)
            self.assertEqual(len(rows), 6)
            self.assertFalse((root / "ram/active-video.mp4").exists())
            annotation_path.write_text(json.dumps([annotation(i) for i in range(5)]))
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                run(config, *args, root / "resume", index, resume=True)


if __name__ == "__main__":
    unittest.main()
