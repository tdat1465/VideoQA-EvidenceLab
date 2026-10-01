"""LongVideoBench validation runner: one RAM-resident video, durable per-question journal."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import signal
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from .backends import HFBackend, MockBackend
from .config import source_fingerprint
from .data import choose_subset, digest, file_hash
from .focus_adapter import BlipITMScorer, load_focus, select_focus
from .hardware import execution_profile
from .longvideo_config import FOCUS_REVISION, FOCUS_SHA256, LLAVA_REVISION, LongVideoConfig
from .longvideo_data import (PARTS, SOURCE, HTTPRangeSource, MultipartReader, archive_index,
                             TRANSFER_CHUNK_BYTES, cgroup_memory, check_space,
                             fetch_annotations, fetch_video, parse_annotations)
from .metrics import export
from .runner import runtime_versions
from .selectors import uniform
from .store import Store, atomic_json, run_lock


def longvideo_runtime(config):
    result = runtime_versions(config.backend == "mock")
    if config.backend != "mock":
        for name in ("decord", "scipy", "huggingface-hub", "tokenizers", "requests", "safetensors",
                     "opencv-python-headless", "ftfy", "regex"):
            result[name] = importlib.metadata.version(name)
    if config.backend == "llava_video":
        import llava
        root = Path(llava.__file__).resolve().parent
        # Every imported upstream source file participates in the resume contract.
        result["llava_source_sha256"] = digest([(p.relative_to(root).as_posix(), file_hash(p))
                                                for p in sorted(root.rglob("*.py"))])
        result["llava_revision"] = LLAVA_REVISION
    return result


def validate_ram_root(path):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run real LongVideoBench jobs inside a Slurm allocation")
    path.mkdir(parents=True, exist_ok=True)
    filesystem = subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "--target", str(path)], text=True).strip()
    if filesystem != "tmpfs":
        raise RuntimeError("Video/weights/cache root must be tmpfs; persistent disk fallback is forbidden")


def make_trace(decision, indices, times, config, details):
    scores = decision.scores
    if (any(not math.isfinite(x) or not 0 <= x <= 1 for x in scores) or abs(sum(scores)-1) > 1e-4):
        raise ValueError("Invalid option probabilities")
    call = {**asdict(decision), "frame_indices": indices, "timestamps": times}
    return {"method": config.method, "prediction": decision.prediction, "scores": scores,
            "calls": [call], "input_tokens": decision.input_tokens + details.get("auxiliary_input_tokens", 0),
            "answer_input_tokens": decision.input_tokens, "frame_presentations": len(indices),
            "final_frames": len(indices), "refined": False, "initial_margin": decision.margin,
            "frame_index_space": "source-video", **details}


def run(config, annotations, ram_root, output, index_path, resume=False,
        max_seconds=165600, max_new_samples=0):
    if max_seconds <= 0 or max_new_samples < 0:
        raise ValueError("Invalid session limits")
    samples = parse_annotations(json.loads(annotations.read_text(encoding="utf-8-sig")))
    selected = choose_subset(samples, config.limit, config.seed)
    mock = config.backend == "mock"
    if not mock:
        validate_ram_root(ram_root)
        if len(samples) != 1337:
            raise ValueError("Pinned LongVideoBench validation must have 1337 questions")
    execution = execution_profile(config)
    contract = {"schema": 3, "config": config.contract(), "resolved_dtype": execution["resolved_dtype"],
                "manifest_sha256": digest({"annotation_sha256": file_hash(annotations), "source": SOURCE}),
                "selected_ids_hash": digest(sorted(s.id for s in selected)), "selected_count": len(selected),
                "source_sha256": source_fingerprint(), "runtime": longvideo_runtime(config),
                "dataset_source": SOURCE, "annotation_sha256": file_hash(annotations),
                "focus_revision": FOCUS_REVISION, "focus_sha256": FOCUS_SHA256,
                "timing_scope": "video decode + selection (including LENS allocation/BLIP/CLIP/SSIM) + answer; excludes model loading, "
                                "archive indexing and video transfer/hash (recorded separately)"}
    old_handlers, stopped = {}, []
    def stop(signum, frame):
        stopped.append(signum)
    for name in ("SIGINT", "SIGTERM", "SIGUSR1"):
        if hasattr(signal, name):
            number = getattr(signal, name)
            old_handlers[number] = signal.signal(number, stop)
    started, processed = time.monotonic(), 0
    video, current_video, video_sha = None, None, None
    reader, frames = None, None
    video_path = ram_root / "active-video.mp4"
    ram_root.mkdir(parents=True, exist_ok=True)
    try:
        with run_lock(output):
            store = Store(output, contract, resume)
            try:
                done = store.done()
                pending = sorted((s for s in selected if s.id not in done), key=lambda s: (s.video, s.id))
                past_hashes = {r["video"]: r["video_sha256"] for r in store.results()}
                if pending:
                    atomic_json(output / "selected-ids.json", sorted(s.id for s in selected))
                    atomic_json(output / "config.latest.json", config.contract())
                    if not mock:
                        import decord
                        decord.bridge.set_bridge("native")
                        source = HTTPRangeSource(stop=lambda: bool(stopped))
                        reader = MultipartReader(source)
                        print("Preparing tar index (first allocation only; metadata is persistent)", flush=True)
                        members = archive_index(reader, index_path, {s.video for s in selected})
                        reader.clear_cache()
                        reader.block_size = TRANSFER_CHUNK_BYTES
                        store.event({"event": "index_ready", "index_sha256": file_hash(index_path),
                                     "transferred_bytes": source.bytes_read})
                    if stopped:
                        raise InterruptedError("Stop requested during setup")
                    if mock:
                        backend = MockBackend()
                    elif config.backend == "llava_video":
                        from .llava_backend import LlavaVideoBackend
                        backend = LlavaVideoBackend(config)
                    else:
                        backend = HFBackend(config)
                    scorer = BlipITMScorer(config) if config.method == "focus" and not mock else None
                    focus = load_focus(Path(__file__).parent / "_vendor/focus.py") if scorer else None
                    lens = None
                    if config.method == "lens" and not mock:
                        from .lens_adapter import LensSelector
                        lens = LensSelector(config, ram_root, stop=lambda: bool(stopped))
                    store.event({"event": "session_start", "execution": execution, "pending": len(pending)})
                    for sample_index, sample in enumerate(pending):
                        if stopped or time.monotonic() - started >= max_seconds:
                            break
                        if max_new_samples and processed >= max_new_samples:
                            break
                        download_seconds, download_bytes = 0., 0
                        new_video = current_video != sample.video
                        if new_video:
                            video = None  # Release decord's handle before replacing the RAM file.
                            video_path.unlink(missing_ok=True)
                            t = time.perf_counter()
                            if mock:
                                video_sha = digest(sample.video)
                            else:
                                entry = members["videos/" + sample.video]
                                network_before = source.bytes_read
                                video_sha = fetch_video(reader, entry, video_path, stop=lambda: bool(stopped))
                                download_bytes = source.bytes_read-network_before
                            download_seconds = time.perf_counter()-t
                            if sample.video in past_hashes and past_hashes[sample.video] != video_sha:
                                raise ValueError("Video bytes changed since previous session")
                            current_video = sample.video
                        backend.reset_peak()
                        if not mock:
                            check_space(ram_root, 0)
                        begin = time.perf_counter()
                        details = {"scorer_frame_evaluations": 0, "scorer_unique_frames": 0, "scorer_batches": 0}
                        if mock:
                            if config.method != "uniform":
                                raise ValueError("Mock runner covers uniform; test FOCUS directly with a synthetic scorer")
                            indices = uniform(2400, config.frames)
                            times = [i/30 for i in indices]
                            frames, duration = indices, 80.
                        else:
                            from PIL import Image
                            if video is None:
                                video = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=2)
                            fps = float(video.get_avg_fps())
                            if len(video) == 0 or not math.isfinite(fps) or fps <= 0:
                                raise ValueError("Invalid video frame count/FPS")
                            duration = len(video)/fps
                            select_start = time.perf_counter()
                            if config.method == "focus":
                                indices, details = select_focus(video, sample.public_question(), config, scorer, focus)
                            elif config.method == "lens":
                                indices, frames, details = lens.select(video, sample.public_question(), backend)
                            else:
                                indices = uniform(len(video), config.frames)
                            # Account for asynchronous CUDA scoring before reporting selector time.
                            backend.torch.cuda.synchronize()
                            details["selection_seconds"] = time.perf_counter()-select_start
                            times = [i/fps for i in indices]
                            if config.method != "lens":
                                frames = [Image.fromarray(video[i].asnumpy()).convert("RGB") for i in indices]
                        backend.video_duration = duration
                        if not mock:
                            check_space(ram_root, 0)
                        decision = backend.answer(sample.public_question(), frames, times)
                        if len(decision.scores) != len(sample.choices):
                            raise ValueError("Answerer returned wrong number of choices")
                        trace = make_trace(decision, indices, times, config, details)
                        memory = backend.measurements()
                        elapsed = time.perf_counter()-begin
                        result = {**trace, **memory, "id": sample.id, "video": sample.video,
                                  "video_sha256": video_sha, "dataset": "longvideobench", "split": "val",
                                  "question_type": sample.question_type, "duration_group": sample.duration_group,
                                  "source_duration_seconds": duration, "annotation_duration_seconds": sample.duration,
                                  "answer": sample.answer, "correct": decision.prediction == sample.answer,
                                  "elapsed_seconds": elapsed, "video_transfer_hash_seconds": download_seconds,
                                  "video_download_bytes": download_bytes, "execution": execution,
                                  "job_cgroup_memory": cgroup_memory()}
                        store.put(result)
                        processed += 1
                        frames = None
                        print(f"Committed {len(done)+processed}/{len(selected)}: {sample.id} "
                              f"({config.method}, {len(indices)} frames, {elapsed:.2f}s)", flush=True)
                        if sample_index + 1 == len(pending) or pending[sample_index + 1].video != sample.video:
                            # All selected questions for this source video are committed.
                            # Close decoder before unlink, so tmpfs pages can be released.
                            video = None
                            gc.collect()
                            video_path.unlink(missing_ok=True)
                            if reader is not None:
                                reader.clear_cache()
                            current_video = None
                            store.event({"event": "video_evicted", "video": sample.video,
                                         "job_cgroup_memory": cgroup_memory()})
                            print(f"Released RAM video: {sample.video}", flush=True)
                    store.event({"event": "session_end", "new_results": processed, "signals": stopped,
                                 "wall_seconds": time.monotonic()-started})
            except InterruptedError:
                store.event({"event": "controlled_stop", "signals": stopped})
            except Exception as error:
                store.event({"event": "error", "type": type(error).__name__, "message": str(error)})
                raise
            finally:
                # Also release an unfinished video on pilot stop, cancellation or error,
                # before exporting persistent reports.
                frames, video = None, None
                gc.collect()
                if reader is not None:
                    reader.clear_cache()
                video_path.unlink(missing_ok=True)
                try:
                    summary = export(output, contract, store.results())
                finally:
                    store.close()
        return 0 if summary["complete"] else 75
    finally:
        video = None
        video_path.unlink(missing_ok=True)
        for number, handler in old_handlers.items():
            signal.signal(number, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "doctor", "run"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ram-root", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=165600)
    parser.add_argument("--max-new-samples", type=int, default=0)
    args = parser.parse_args()
    config = LongVideoConfig.load(args.config)
    if args.command == "doctor":
        print(json.dumps({"runtime": longvideo_runtime(config), **execution_profile(config)}, indent=2))
        return 0
    if args.ram_root is None:
        parser.error("--ram-root is required")
    if args.command == "prepare":
        validate_ram_root(args.ram_root)
        path = fetch_annotations(args.ram_root)
        rows = parse_annotations(json.loads(path.read_text(encoding="utf-8")))
        if len(rows) != 1337:
            raise ValueError("Expected exactly 1337 validation questions")
        target = args.ram_root / "lvb_val.json"
        target.write_bytes(path.read_bytes())
        name, size = PARTS[0]
        HTTPRangeSource().read(name, 0, 512, size)
        print(f"Verified validation annotation: {len(rows)} questions, sha256={file_hash(target)}", flush=True)
        print("Verified HTTP Range support on the first official archive part", flush=True)
        return 0
    if args.annotations is None or args.output is None or args.index is None:
        parser.error("run requires --annotations, --output and --index")
    return run(config, args.annotations, args.ram_root, args.output, args.index, args.resume,
               args.max_seconds, args.max_new_samples)


if __name__ == "__main__":
    raise SystemExit(main())
