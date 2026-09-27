"""Stage user-provided raw videos in RAM and build a normalized manifest.

Does not download restricted datasets or guess an archive's licensing terms.
ZIP/TAR archive traversal, symlinks, device nodes and duplicate files are rejected.
"""
from __future__ import annotations

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

from evidencelab.data import prepare


def member_path(root, name):
    parts = PurePosixPath(name.replace("\\", "/"))
    if parts.is_absolute() or ".." in parts.parts or ":" in name:
        raise ValueError(f"Unsafe archive member: {name}")
    path = (root / str(parts)).resolve()
    if path != root and root not in path.parents:
        raise ValueError("Archive path escapes target")
    return path


def extract(archive, target):
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=False)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            entries = z.infolist()
            total = sum(x.file_size for x in entries)
            if total > shutil.disk_usage(target).free * .8:
                raise ValueError("Insufficient RAM filesystem space for extracted archive + headroom")
            for info in entries:
                path = member_path(target, info.filename)
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Archive symlinks are not accepted")
                if info.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(info) as src, path.open("xb") as dst:
                        shutil.copyfileobj(src, dst)
    else:
        with tarfile.open(archive, "r:*") as tar:
            entries = tar.getmembers()
            if sum(x.size for x in entries) > shutil.disk_usage(target).free * .8:
                raise ValueError("Insufficient RAM filesystem space for extracted archive + headroom")
            for info in entries:
                path = member_path(target, info.name)
                if info.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                elif info.isfile():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(info) as src, path.open("xb") as dst:
                        shutil.copyfileobj(src, dst)
                else:
                    raise ValueError("Only ordinary files and directories are allowed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--dataset", choices=["nextqa", "star"], required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--mapping", type=Path)
    a = p.parse_args()
    extract(a.archive, a.target / "videos")
    n = prepare(a.dataset, a.annotations, a.target / "videos", a.split,
                a.target / "manifest.jsonl", a.mapping)
    print(f"Staged {n} questions at {a.target}")


if __name__ == "__main__":
    main()
