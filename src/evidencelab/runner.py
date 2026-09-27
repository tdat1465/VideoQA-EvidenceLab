from __future__ import annotations

import importlib.metadata
import platform
import signal
import time
from pathlib import Path

from .backends import ClipScorer, HFBackend, MockBackend, MockScorer
from .config import Config, source_fingerprint
from .data import choose_subset, digest, file_hash, load_manifest
from .metrics import export
from .pipeline import infer
from .selectors import AKS_SHA256
from .store import Store, run_lock
from .video import VideoReader


def runtime_versions(mock=False):
    result = {"python": platform.python_version()}
    if not mock:
        for name in ("torch", "torchvision", "transformers", "accelerate", "numpy", "Pillow", "av"):
            result[name] = importlib.metadata.version(name)
    return result


def run(config: Config, manifest: Path, video_root: Path, output: Path, resume=False,
        aks_file: Path | None = None, max_seconds=165600, max_new_samples=0):
    if max_seconds <= 0 or max_new_samples < 0:
        raise ValueError("Invalid session limits")
    selected = choose_subset(load_manifest(manifest), config.limit, config.seed)
    if config.method == "aks" and (aks_file is None or file_hash(aks_file) != AKS_SHA256):
        raise ValueError("Fetch the pinned AKS implementation and pass --aks-file")
    contract = {"schema": 1, "config": config.contract(), "manifest_sha256": file_hash(manifest),
                "selected_ids_hash": digest(sorted(s.id for s in selected)), "selected_count": len(selected),
                "source_sha256": source_fingerprint(), "runtime": runtime_versions(config.backend == "mock"),
                "aks_sha256": AKS_SHA256 if config.method == "aks" else None}
    stop = []
    old_handlers = {}

    def request_stop(signum, frame):
        stop.append(signum)  # Finish and commit this question; stop before the next.

    for name in ("SIGTERM", "SIGINT", "SIGUSR1"):
        if hasattr(signal, name):
            number = getattr(signal, name)
            old_handlers[number] = signal.signal(number, request_stop)
    started, processed = time.monotonic(), 0
    try:
        with run_lock(output):
            store = Store(output, contract, resume)
            try:
                done = store.done()
                pending = sorted((s for s in selected if s.id not in done), key=lambda s: (s.video, s.start, s.id))
                if pending:
                    mock = config.backend == "mock"
                    backend = MockBackend() if mock else HFBackend(config)
                    # CPU CLIP avoids competing with the answerer for VRAM. Its load is excluded from timing.
                    needs_scorer = config.method in {"evidence", "aks", "clip_topk"}
                    scorer = (MockScorer() if mock else ClipScorer(config)) if needs_scorer else None
                    reader = None if mock else VideoReader(video_root, config.candidate_frames)
                    store.event({"event": "session_start", "pending": len(pending), "resume": resume})
                    for sample in pending:
                        if stop or time.monotonic() - started >= max_seconds:
                            break
                        if max_new_samples and processed >= max_new_samples:
                            break
                        backend.reset_peak()
                        begin = time.perf_counter()
                        if mock:
                            frames = list(range(config.candidate_frames))
                            times = [sample.start + i * .25 for i in frames]
                        else:
                            frames, times = reader.read(sample)
                        trace = infer(sample.public_question(), frames, times, config, backend, scorer, aks_file)
                        memory = backend.measurements()  # Synchronize CUDA before stopping timer.
                        result = {**trace, **memory, "elapsed_seconds": time.perf_counter() - begin,
                                  "id": sample.id, "video": sample.video, "dataset": sample.dataset,
                                  "split": sample.split, "question_type": sample.question_type,
                                  "answer": sample.answer,
                                  "correct": trace["prediction"] == sample.answer if sample.answer is not None else None}
                        store.put(result)
                        processed += 1
                        print(f"Committed {len(done)+processed}/{len(selected)}: {sample.id}", flush=True)
                    store.event({"event": "session_end", "new_results": processed, "signals": stop,
                                 "wall_seconds": time.monotonic() - started})
            except Exception as error:
                store.event({"event": "error", "type": type(error).__name__, "message": str(error)})
                raise
            finally:
                try:
                    summary = export(output, contract, store.results())
                finally:
                    store.close()
        return 0 if summary["complete"] else 75
    finally:
        for number, handler in old_handlers.items():
            signal.signal(number, handler)
