"""Run four video-disjoint GPU workers and merge their journals."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import ARMS, HERE, atomic_json, parse_annotations, run_lock, summary
from evidencelab.store import read_run

WORKERS = 4


def split_by_video(samples):
    groups = defaultdict(list)
    for sample in samples:
        groups[sample.video].append(sample)
    shards = [[] for _ in range(WORKERS)]
    loads = [0.] * WORKERS

    def cost(group):
        return max(sample.duration for sample in group) + 30 * len(group)

    for video, group in sorted(groups.items(), key=lambda entry: (-cost(entry[1]), entry[0])):
        rank = min(range(WORKERS), key=lambda i: (loads[i], len(shards[i]), i))
        shards[rank].extend(sorted(sample.id for sample in group))
        loads[rank] += cost(group)
    return shards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--subtitles", type=Path, required=True)
    parser.add_argument("--archive-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session-minutes", type=int, default=720)
    args = parser.parse_args()
    samples = parse_annotations(json.loads(args.annotation.read_text(encoding="utf-8")))
    if len(samples) != 1337 or not args.archive_index.is_file():
        raise ValueError("Require 1337 annotations and the completed pilot archive index")
    shards = split_by_video(samples)
    with run_lock(args.output):
        processes = []
        for rank, ids in enumerate(shards):
            folder = args.output / f"worker-{rank}"
            folder.mkdir(exist_ok=True)
            manifest = folder / "questions.json"
            if manifest.exists() and json.loads(manifest.read_text()) != ids:
                raise ValueError(f"Worker {rank} shard changed")
            atomic_json(manifest, ids)
        for rank, ids in enumerate(shards):
            folder = args.output / f"worker-{rank}"
            command = [sys.executable, "-u", str(HERE / "run.py"),
                       "--annotation", str(args.annotation), "--subtitles", str(args.subtitles),
                       "--archive-index", str(args.archive_index), "--output", str(folder),
                       "--question-ids", str(folder / "questions.json"), "--max-new-questions", "0",
                       "--session-minutes", str(args.session_minutes)]
            with (folder / "worker.log").open("a") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            print(f"Worker {rank} PID {process.pid}, {len(ids)} questions", flush=True)
            atomic_json(args.output / "workers.json", {"pids": [p.pid for p in processes],
                                                       "questions": [len(ids) for ids in shards]})
            if rank < WORKERS - 1:
                time.sleep(8)
        codes = [p.wait() for p in processes]
        rows = []
        for rank, ids in enumerate(shards):
            folder = args.output / f"worker-{rank}"
            if (folder / "results.sqlite3").is_file():
                _, subset = read_run(folder)
                expected = {f"{q}::{arm}" for q in ids for arm in ARMS}
                if not {r["id"] for r in subset} <= expected:
                    raise ValueError(f"Unexpected row in worker {rank}")
                rows.extend(subset)
        report = summary(rows)
        report["worker_exit_codes"] = codes
        atomic_json(args.output / "summary.json", report)
        temp = args.output / "predictions.tmp.jsonl"
        with temp.open("w", encoding="utf-8") as stream:
            for row in sorted(rows, key=lambda r: r["id"]):
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        temp.replace(args.output / "predictions.jsonl")
        return 0 if report["complete"] and all(code == 0 for code in codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
