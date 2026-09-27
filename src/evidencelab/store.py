"""Durable per-question transactions and process locks for 48-hour sessions."""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path


def atomic_json(path: Path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


@contextmanager
def run_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / "run.lock").open("a+b")
    locked = False
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        yield
    finally:
        if locked:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


class Store:
    def __init__(self, directory: Path, contract: dict, resume: bool):
        path = directory / "results.sqlite3"
        existed = path.exists()
        if existed and not resume:
            raise FileExistsError("Run already exists; use --resume or a NEW run directory")
        if not existed and resume:
            raise FileNotFoundError("No run to resume")
        self.db = sqlite3.connect(path)
        try:
            self.db.execute("PRAGMA journal_mode=DELETE")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self.db.execute("CREATE TABLE IF NOT EXISTS results (id TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self.db.execute("CREATE TABLE IF NOT EXISTS events (value TEXT NOT NULL)")
            old = self.db.execute("SELECT value FROM meta WHERE key='contract'").fetchone()
            serialized = json.dumps(contract, sort_keys=True, allow_nan=False)
            if old and old[0] != serialized:
                raise ValueError("Resume contract mismatch: source/config/data/model/runtime changed. Use a new run.")
            if existed and not old:
                raise ValueError("Incomplete run initialization; use a new directory")
            if not old:
                self.db.execute("INSERT INTO meta VALUES ('contract', ?)", (serialized,))
                self.db.commit()
                atomic_json(directory / "contract.json", contract)
        except Exception:
            self.db.close()
            raise

    def done(self):
        return {row[0] for row in self.db.execute("SELECT id FROM results")}

    def put(self, result):
        with self.db:
            self.db.execute("INSERT INTO results VALUES (?, ?)",
                            (result["id"], json.dumps(result, allow_nan=False)))

    def event(self, event):
        with self.db:
            self.db.execute("INSERT INTO events VALUES (?)", (json.dumps(event, allow_nan=False),))

    def results(self):
        return [json.loads(row[0]) for row in self.db.execute("SELECT value FROM results ORDER BY id")]

    def close(self):
        self.db.close()


def read_run(directory: Path):
    path = directory / "results.sqlite3"
    if not path.is_file():
        raise FileNotFoundError(path)
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        contract = json.loads(db.execute("SELECT value FROM meta WHERE key='contract'").fetchone()[0])
        results = [json.loads(row[0]) for row in db.execute("SELECT value FROM results ORDER BY id")]
        return contract, results
    finally:
        db.close()
