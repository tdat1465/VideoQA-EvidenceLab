"""Bounded HTTP Range reader for the official, *uncompressed* multipart tar.

No concatenated tar, full shard, or complete dataset is written to disk.
Small tar indexes may persist. Video bytes only go under the job's RAM root.
"""
from __future__ import annotations

import bisect
import errno
import hashlib
import io
import json
import math
import os
import shutil
import tarfile
import time
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path

from .data import Question, digest, relative_video
from .longvideo_config import DATASET_REVISION
from .store import atomic_json, run_lock

DATASET_ID = "longvideobench/LongVideoBench"
PARTS = [(f"videos.tar.part.{suffix}", 5242880000) for suffix in
         ["a" + chr(i) for i in range(ord("a"), ord("z") + 1)] + ["ba", "bb", "bc", "bd"]]
PARTS.append(("videos.tar.part.be", 4277780480))
SOURCE = {"repo_id": DATASET_ID, "revision": DATASET_REVISION, "parts": PARTS}
TRANSFER_CHUNK_BYTES = 4 * 1024**2


@dataclass(frozen=True)
class LongVideoSample:
    id: str
    video: str
    question: str
    choices: tuple[str, ...]
    answer: int
    question_type: str
    duration_group: str
    duration: float

    def public_question(self):
        return Question(self.id, self.question, self.choices)


def parse_annotations(rows):
    result, seen = [], set()
    for row in rows:
        qid = "longvideobench:val:" + str(row["id"])
        if qid in seen:
            raise ValueError(f"Duplicate question: {qid}")
        seen.add(qid)
        choices = tuple(row["candidates"])
        answer = row["correct_choice"]
        if not 2 <= len(choices) <= 5 or any(not isinstance(s, str) or not s.strip() for s in choices):
            raise ValueError("Invalid LongVideoBench candidates")
        if type(answer) is not int or not 0 <= answer < len(choices):
            raise ValueError("Expected labelled validation; test_wo_gt is not supported")
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError("Empty question")
        duration = float(row["duration"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Invalid video duration")
        result.append(LongVideoSample(qid, relative_video(row["video_path"]), row["question"], choices,
                                     answer, str(row["question_category"]), str(row["duration_group"]), duration))
    if not result:
        raise ValueError("Empty validation annotations")
    return result


def fetch_annotations(ram_root):
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(DATASET_ID, "lvb_val.json", repo_type="dataset",
                               revision=DATASET_REVISION, cache_dir=str(ram_root / "annotation-cache"))
    except Exception as error:
        # Do not echo authentication headers or signed URLs in exception strings.
        raise RuntimeError("Cannot read LongVideoBench. Accept access conditions on Hugging Face, "
                           "then export HF_TOKEN with read permission before sbatch. "
                           f"Failure type: {type(error).__name__}") from None
    return Path(path)


class HTTPRangeSource:
    def __init__(self, token=None, stop=None, session=None, base_url=None):
        import requests
        self.session = session or requests.Session()
        self.token = token if token is not None else os.environ.get("HF_TOKEN")
        self.stop = stop or (lambda: False)
        self.base_url = base_url or f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}"
        self.bytes_read = 0

    def read(self, part, start, length, total):
        if length <= 0:
            return b""
        last = start + length - 1
        # Query differentiates CDN cache keys for different Range requests.
        url = f"{self.base_url}/{part}?download=true&range_start={start}&range_end={last}"
        headers = {"Range": f"bytes={start}-{last}", "Accept-Encoding": "identity"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        for attempt in range(4):
            if self.stop():
                raise InterruptedError("Stop requested during dataset access")
            try:
                # requests strips credentials on cross-host redirects to the signed CDN URL.
                with self.session.get(url, headers=headers, stream=True, timeout=(20, 60)) as response:
                    if response.status_code in {401, 403}:
                        raise PermissionError("Dataset access denied; check accepted conditions and HF_TOKEN")
                    if response.status_code != 206:
                        raise RuntimeError(f"Expected HTTP 206, got {response.status_code}; "
                                           "refusing full archive download")
                    if response.headers.get("Content-Range") != f"bytes {start}-{last}/{total}":
                        raise ValueError("Server returned the wrong Content-Range")
                    if response.headers.get("Content-Encoding", "identity") != "identity":
                        raise ValueError("Compressed HTTP response is not byte-addressable")
                    data = response.raw.read(length + 1)
                    if len(data) != length:
                        raise OSError("Truncated or oversized HTTP range")
                    self.bytes_read += len(data)
                    return data
            except (PermissionError, ValueError):
                raise
            except Exception as error:
                if attempt == 3:
                    raise RuntimeError(f"Range read failed ({type(error).__name__}); no whole-file fallback") from None
                time.sleep(2 ** attempt)


class MultipartReader(io.RawIOBase):
    """Seekable virtual concatenation; at most one small block stays cached."""
    def __init__(self, source, parts=PARTS, block_size=65536):
        self.source, self.parts, self.block_size = source, parts, block_size
        self.starts = [0]
        for _, size in parts:
            self.starts.append(self.starts[-1] + size)
        self.position = 0
        self.cached_start, self.cached_data = -1, b""
        self.read_end = self.starts[-1]

    def clear_cache(self):
        self.cached_start, self.cached_data = -1, b""

    @contextmanager
    def window(self, offset, size):
        """Bound requests to one member, including the last partial HTTP block."""
        if offset < 0 or size < 0 or offset + size > self.starts[-1]:
            raise ValueError("Invalid member range")
        old_end = self.read_end
        self.clear_cache()
        self.read_end = offset + size
        self.seek(offset)
        try:
            yield
        finally:
            self.clear_cache()
            self.read_end = old_end

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        target = offset + (self.position if whence == 1 else self.starts[-1] if whence == 2 else 0)
        if whence not in (0, 1, 2) or target < 0:
            raise ValueError("Invalid seek")
        self.position = target
        return target

    def read(self, size=-1):
        if size < 0:
            raise ValueError("Unbounded archive reads are forbidden")
        output = bytearray()
        remaining = min(size, max(0, self.read_end - self.position))
        while remaining:
            if not self.cached_start <= self.position < self.cached_start + len(self.cached_data):
                part_id = bisect.bisect_right(self.starts, self.position) - 1
                offset = self.position - self.starts[part_id]
                name, part_size = self.parts[part_id]
                self.cached_start = self.position
                length = min(self.block_size, part_size-offset, self.read_end-self.position)
                self.cached_data = self.source.read(name, offset, length, part_size)
            offset = self.position - self.cached_start
            chunk = self.cached_data[offset:offset+remaining]
            if not chunk:
                raise EOFError("Empty archive block")
            output.extend(chunk)
            remaining -= len(chunk)
            self.position += len(chunk)
        return bytes(output)


@contextmanager
def index_lock(directory, stop):
    waiting = False
    while True:
        lock = run_lock(directory)
        try:
            lock.__enter__()
            break
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            if stop():
                raise InterruptedError("Stopped while waiting for the shared archive index")
            if not waiting:
                print("Another allocation is building the archive index; waiting", flush=True)
                waiting = True
            time.sleep(2)
    try:
        yield
    finally:
        lock.__exit__(None, None, None)


def archive_index(reader, path, needed):
    """Index headers only; tarfile handles PAX and GNU long names across parts."""
    source_hash = digest(SOURCE)
    with index_lock(path.parent, getattr(reader.source, "stop", lambda: False)):
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["source_hash"] != source_hash:
                raise ValueError("Archive index belongs to a different source")
            members = saved["members"]
            if saved.get("members_hash") != digest(members):
                raise ValueError("Corrupt archive index")
        else:
            members = {}
            with tarfile.open(fileobj=reader, mode="r:") as archive:
                for item in archive:
                    name = relative_video(item.name)
                    if item.isdir():
                        continue
                    if not item.isfile():
                        raise ValueError(f"Unsupported archive member: {name}")
                    if name in members:
                        raise ValueError(f"Duplicate archive member: {name}")
                    members[name] = [item.offset_data, item.size]
                    if len(members) % 100 == 0:
                        print(f"Indexed {len(members)} tar members (headers only)", flush=True)
            atomic_json(path, {"source_hash": source_hash, "members": members, "members_hash": digest(members)})
        for name, pair in members.items():
            relative_video(name)
            if (not isinstance(pair, list) or len(pair) != 2 or any(type(x) is not int for x in pair)
                    or pair[0] < 0 or pair[1] < 0 or sum(pair) > reader.starts[-1]):
                raise ValueError("Invalid archive offset/size")
        missing = {"videos/" + v for v in needed} - members.keys()
        if missing:
            raise FileNotFoundError(f"Missing selected video in official archive: {sorted(missing)[:3]}")
        return members


def cgroup_memory(proc_path=Path("/proc/self/cgroup"), mount=Path("/sys/fs/cgroup")):
    """Read job cgroup v1/v2, never report host total as job allocation."""
    try:
        notebook = os.environ.get("EVIDENCELAB_EXECUTION") == "molab"
        allocation = (2**60 if notebook else int(os.environ["SLURM_MEM_PER_NODE"]) * 1024**2)
        for line in proc_path.read_text().splitlines():
            _, controllers, group_path = line.split(":", 2)
            version2 = line.startswith("0::")
            if version2 or "memory" in controllers.split(","):
                suffix = group_path.lstrip("/")
                if ".." in Path(suffix).parts:
                    continue
                base = mount if version2 else mount / "memory"
                names = (("memory.current", "memory.max", "memory.peak") if version2 else
                         ("memory.usage_in_bytes", "memory.limit_in_bytes", "memory.max_usage_in_bytes"))
                leaf = base / suffix
                candidates = [leaf] + [p for p in leaf.parents if p == base or base in p.parents]
                for root in candidates:
                    if (root / names[0]).exists():
                        raw = (root / names[1]).read_text().strip()
                        # A leaf may inherit its job's limit from an ancestor.
                        # Never call an unlimited/whole-node cgroup a job measurement.
                        if raw == "max" or not 0 < int(raw) <= allocation:
                            continue
                        return {"current_bytes": int((root / names[0]).read_text()),
                                "limit_bytes": int(raw),
                                "peak_bytes": int((root / names[2]).read_text())
                                if (root / names[2]).exists() else None}
    except (OSError, ValueError, KeyError):
        pass
    return {}


def check_space(root, incoming, max_video_bytes=8 * 1024**3, reserve=6 * 1024**3):
    if incoming < 0:
        raise ValueError("Negative incoming byte count")
    if incoming > max_video_bytes:
        raise MemoryError("Single video exceeds 8 GiB guard; no fallback to persistent disk")
    if shutil.disk_usage(root).free < incoming + reserve:
        raise MemoryError("Insufficient free tmpfs space")
    memory = cgroup_memory()
    if os.environ.get("EVIDENCELAB_EXECUTION") == "molab":
        if not memory:
            raise RuntimeError("MoLab requires readable finite container cgroup memory accounting")
        # A userspace budget, not a claim that we changed the container's limit.
        effective = min(memory["limit_bytes"], 90 * 1024**3)
        if effective - memory["current_bytes"] < incoming + reserve:
            raise MemoryError("Insufficient MoLab RAM headroom within the 90 GiB software budget")
    if memory.get("limit_bytes") is not None:
        if memory["limit_bytes"] - memory["current_bytes"] < incoming + reserve:
            raise MemoryError("Insufficient cgroup RAM headroom")
    return memory


def fetch_video(reader, entry, path, stop=lambda: False, guard=True):
    offset, size = entry
    if path.exists():
        raise FileExistsError("Release the active video before fetching another")
    if guard:
        check_space(path.parent, size)
    h = hashlib.sha256()
    partial = path.with_suffix(".partial")
    try:
        with reader.window(offset, size), partial.open("wb", buffering=0) as output:
            remaining = size
            while remaining:
                if stop():
                    raise InterruptedError("Stop requested during video download")
                if guard:
                    # Existing tmpfs bytes are already charged to the cgroup;
                    # reserve space only for the bytes still to arrive.
                    check_space(path.parent, remaining)
                block = reader.read(min(TRANSFER_CHUNK_BYTES, remaining))
                if not block:
                    raise EOFError("Video data truncated")
                if output.write(block) != len(block):
                    raise OSError("Short write while staging video in RAM")
                h.update(block)
                remaining -= len(block)
                del block
            if guard:
                check_space(path.parent, 0)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    return h.hexdigest()
