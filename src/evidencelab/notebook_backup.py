"""Small, consistent run snapshots. Never archive caches, media, .env or tokens."""
import json
import os
import shutil
import sqlite3
import tempfile
import zipfile
from pathlib import Path

from .metrics import export
from .store import read_run, atomic_json

ALLOWED = {"results.sqlite3", "contract.json", "summary.json", "predictions.jsonl", "predictions.csv",
           "config.latest.json", "selected-ids.json", "environment.latest.txt", "gpu.latest.txt",
           "archive-index.json"}
MAX_BYTES = 256 * 1024**2


def backup_run(output: Path, archive: Path, index: Path | None = None):
    database = output / "results.sqlite3"
    if not database.is_file():
        raise FileNotFoundError("No journal yet; setup may still be running")
    if archive.resolve().is_relative_to(output.resolve()):
        raise ValueError("Keep backup ZIP outside the active run directory")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=archive.parent, prefix="snapshot-") as temp:
        root = Path(temp)
        source = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
        target = sqlite3.connect(root / "results.sqlite3")
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
        contract, results = read_run(root)
        atomic_json(root / "contract.json", contract)
        atomic_json(root / "config.latest.json", contract["config"])
        export(root, contract, results)
        for name in ("selected-ids.json", "environment.latest.txt", "gpu.latest.txt"):
            path = output / name
            if path.is_file() and not path.is_symlink():
                shutil.copyfile(path, root / name)
        if index is not None and index.is_file():
            shutil.copyfile(index, root / "archive-index.json")
        if sum(p.stat().st_size for p in root.iterdir()) > MAX_BYTES:
            raise ValueError("Run snapshot exceeds 256 MiB guard")
        packed = root / "snapshot.zip"
        with zipfile.ZipFile(packed, "w", zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(root.iterdir()):
                if path.name in ALLOWED:
                    bundle.write(path, path.name)
        os.replace(packed, archive)
    return len(results)


def restore_run(archive: Path, output: Path):
    if output.exists():
        raise FileExistsError("Restore requires a NEW output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        files = bundle.infolist()
        names = [p.filename for p in files]
        if (len(set(names)) != len(names) or "results.sqlite3" not in names
                or any(name not in ALLOWED for name in names)
                or sum(p.file_size for p in files) > MAX_BYTES):
            raise ValueError("Invalid run backup contents")
        with tempfile.TemporaryDirectory(dir=output.parent, prefix="restore-") as temp:
            root = Path(temp)
            for item in files:
                # Fixed top-level allowlist; write bytes, never restore symlinks.
                with bundle.open(item) as source, (root / item.filename).open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024**2)
            contract, results = read_run(root)
            # Regenerate reports from the authoritative, transaction-consistent DB.
            atomic_json(root / "contract.json", contract)
            atomic_json(root / "config.latest.json", contract["config"])
            export(root, contract, results)
            output.mkdir()
            for path in root.iterdir():
                shutil.move(str(path), output / path.name)
    return len(results)
