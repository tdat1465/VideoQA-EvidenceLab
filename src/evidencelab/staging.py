"""Download pinned NExT-QA inputs to the GPU node's RAM filesystem only."""
from __future__ import annotations

import csv
import http.client
import json
import shutil
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from .data import file_hash, prepare
from .store import atomic_json

VIDEO_REPO = "rhymes-ai/NeXTVideo"
VIDEO_REVISION = "7e8ea8e056742292b95688d92a0773e05df00393"
VIDEO_BYTES = 24253160356
VIDEO_SHA256 = "2e3b1bc3e761122864b46fe3a1790b281301a6ed69de5ca1fceba17c504fa49c"
VIDEO_URL = f"https://huggingface.co/datasets/{VIDEO_REPO}/resolve/{VIDEO_REVISION}/NExTVideo.zip"
ANNOTATION_REVISION = "2432e9724f88ed9f40010e2989f104570a91de4e"
GIB = 1024**3


def require_tmpfs(path: Path):
    path = path.resolve(strict=True)
    result = subprocess.run(["findmnt", "-n", "-o", "FSTYPE", "--target", str(path)],
                            capture_output=True, text=True, check=True)
    if result.stdout.strip() != "tmpfs":
        raise ValueError(f"Refusing data download outside tmpfs: {path}")


def download_verified(url, output: Path, expected_size, expected_hash, attempts=3):
    """Bounded disk/RAM usage; resume partial transfer only with valid HTTP Range."""
    if output.exists():
        if output.stat().st_size == expected_size and file_hash(output) == expected_hash:
            return
        raise ValueError("Existing download does not match pinned source")
    partial = output.with_suffix(output.suffix + ".part")
    for attempt in range(attempts):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected_size:
            raise ValueError("Partial download is larger than pinned source")
        try:
            if offset < expected_size:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=90) as response:
                    append = offset > 0 and response.status == 206
                    if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Server returned a different byte range")
                    if offset and not append:
                        if response.status != 200:
                            raise ValueError("Server could not resume the download")
                        offset = 0
                    received, reported = offset, offset
                    with partial.open("ab" if append else "wb") as out:
                        while True:
                            block = response.read(4 * 1024**2)
                            if not block:
                                break
                            received += len(block)
                            if received > expected_size:
                                raise ValueError("Download exceeds pinned size")
                            out.write(block)
                            if received - reported >= 256 * 1024**2:
                                print(f"Video archive: {received/GIB:.2f}/{expected_size/GIB:.2f} GiB", flush=True)
                                reported = received
                if received != expected_size:
                    raise OSError("Incomplete download")
            print("Checking archive SHA256...", flush=True)
            if file_hash(partial) != expected_hash:
                raise ValueError("Archive SHA256 differs from pinned source; refusing extraction")
            partial.replace(output)
            return
        except (OSError, TimeoutError, http.client.IncompleteRead):
            if attempt + 1 == attempts:
                raise
            print(f"Transfer interrupted; retry {attempt+2}/{attempts} within this RAM session", flush=True)
            time.sleep(2 ** attempt)


def extract_split(archive: Path, target: Path, video_ids: set[str], reserve_gib=16):
    """Extract all videos for a split, reject missing/ambiguous IDs, never guess."""
    target.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        selected = {}
        for info in z.infolist():
            name = PurePosixPath(info.filename.replace("\\", "/"))
            if name.is_absolute() or ".." in name.parts or ":" in info.filename:
                raise ValueError("Unsafe ZIP member")
            if name.suffix.lower() != ".mp4" or name.stem not in video_ids:
                continue
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("ZIP symlinks are not accepted")
            if name.stem in selected:
                raise ValueError(f"Ambiguous archive video: {name.stem}")
            selected[name.stem] = info
        missing = video_ids - selected.keys()
        if missing:
            raise ValueError(f"Archive missing {len(missing)} videos, e.g. {sorted(missing)[:3]}")
        total = sum(info.file_size for info in selected.values())
        if total + reserve_gib * GIB > shutil.disk_usage(target).free:
            raise ValueError("Insufficient free tmpfs for selected videos and reserved headroom")
        for video_id, info in sorted(selected.items()):
            # Keep upstream relative layout so manifests are stable across sessions.
            output = target / PurePosixPath(info.filename)
            output.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, output.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=4 * 1024**2)
        return {"video_count": len(selected), "uncompressed_bytes": total}


def stage_nextqa(target: Path, split: str, provenance: Path):
    if split not in {"train", "val", "test"}:
        raise ValueError("Unknown NExT-QA split")
    target.mkdir(parents=True, exist_ok=False)
    require_tmpfs(target)
    if shutil.disk_usage(target).free < VIDEO_BYTES + 32 * GIB:
        raise ValueError("Need room for 22.59 GiB archive plus 32 GiB staging headroom in tmpfs")
    annotation_url = (f"https://raw.githubusercontent.com/doc-doc/NExT-QA/{ANNOTATION_REVISION}"
                      f"/dataset/nextqa/{split}.csv")
    annotations = target / f"{split}.csv"
    with urllib.request.urlopen(annotation_url, timeout=90) as response:
        content = response.read(32 * 1024**2 + 1)
    if len(content) > 32 * 1024**2:
        raise ValueError("Unexpectedly large annotation response")
    annotations.write_bytes(content)
    with annotations.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"video", "qid", "question", "answer", "type", "a0", "a1", "a2", "a3", "a4"}
    if not rows or not required <= rows[0].keys():
        raise ValueError("Downloaded annotation is not an MCQA CSV")
    video_ids = {row["video"] for row in rows}
    print(f"NExT-QA {split}: {len(rows)} questions, {len(video_ids)} source videos", flush=True)
    archive = target / "NExTVideo.zip"
    download_verified(VIDEO_URL, archive, VIDEO_BYTES, VIDEO_SHA256)
    extracted = extract_split(archive, target / "videos", video_ids)
    archive.unlink()  # Exact downloaded file; release archive RAM before model loading.
    print("Selected videos extracted; archive removed from RAM", flush=True)
    count = prepare("nextqa", annotations, target / "videos", split, target / "manifest.jsonl")
    source = {"dataset": "nextqa", "split": split, "questions": count, **extracted,
              "video_repo": VIDEO_REPO, "video_revision": VIDEO_REVISION,
              "archive_sha256": VIDEO_SHA256, "archive_bytes": VIDEO_BYTES,
              "annotation_revision": ANNOTATION_REVISION, "annotation_sha256": file_hash(annotations)}
    if provenance.exists() and json.loads(provenance.read_text()) != source:
        raise ValueError("Run data provenance differs; use a NEW run directory")
    atomic_json(provenance, source)
    print(f"Prepared complete {split} manifest; runner applies its configured question limit", flush=True)
