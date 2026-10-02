"""Submit independent LongVideoBench jobs with separate output directories.

Standard library only on login01; default is a dry run. Dataset/model staging
remains inside each compute allocation. Never submit an already complete run.
"""
import argparse
from contextlib import closing
import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from evidencelab.longvideo_config import LongVideoConfig
from evidencelab.config import source_fingerprint


def plans(root, backend, frames, methods, label, resume, limit=200):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", label):
        raise ValueError("Use a short label with letters, numbers, hyphens or underscores")
    if len(methods) != len(set(methods)):
        raise ValueError("Duplicate methods")
    result = []
    for method in methods:
        suffix = "_full" if limit == 0 else ""
        name = f"configs/lvb_{backend}_{method}{frames}{suffix}.json"
        config = LongVideoConfig.load(REPO / name)
        if (config.backend, config.method, config.frames, config.limit) != (backend, method, frames, limit):
            raise ValueError(f"Unexpected config: {name}")
        scope = "full" if limit == 0 else str(limit)
        run = root / "runs/videoqa" / f"lvb-val-{backend}-{method}{frames}-{scope}-{label}"
        journal = run / "results.sqlite3"
        if journal.exists():
            with closing(sqlite3.connect(journal.resolve().as_uri() + '?mode=ro', uri=True)) as db:
                contract = json.loads(db.execute("SELECT value FROM meta WHERE key='contract'").fetchone()[0])
                count = db.execute("SELECT COUNT(*) FROM results").fetchone()[0]
            if contract['config'] != config.contract() or contract['source_sha256'] != source_fingerprint():
                raise ValueError(f"Source/config changed; use a new label: {run}")
            if count == contract['selected_count']:
                print(f"SKIP complete: {run}")
                continue
        if journal.exists() and not resume:
            raise ValueError(f"Existing journal needs --resume: {run}")
        # --resume is per suite: fresh methods can start alongside partial ones.
        result.append((name, run, journal.exists()))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["molmo2", "llava_video"], default="molmo2")
    p.add_argument("--frames", type=int, choices=[8, 64], default=64)
    p.add_argument("--methods", nargs="+", choices=["uniform", "aks", "focus", "lens"],
                   default=["uniform", "aks", "focus"])
    p.add_argument("--label", required=True)
    p.add_argument("--persist-root", type=Path, default=Path("/media/lnthanh03/DatHa"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--full-validation", action="store_true", help="All 1,337 validation questions; requires 64 frames")
    p.add_argument("--max-new-samples", type=int, default=2)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--test-only", action="store_true")
    mode.add_argument("--submit", action="store_true")
    p.add_argument("--partition", default="batch")
    p.add_argument("--node", choices=["gpu01", "gpu02", "gpu03"])
    p.add_argument("--account")
    args = p.parse_args()
    if args.max_new_samples < 0 or not args.persist_root.is_absolute():
        p.error("Use an absolute persist root and nonnegative max-new-samples")
    if args.full_validation and args.frames != 64:
        p.error("Full-validation suite configs currently use 64 frames")
    items = plans(args.persist_root, args.backend, args.frames, args.methods, args.label, args.resume,
                  limit=0 if args.full_validation else 200)
    if (args.submit or args.test_only) and not os.environ.get("HF_TOKEN"):
        p.error("Export HF_TOKEN before submission")
    records = []
    for config, run, resume in items:
        print(f"CONFIG_FILE={config}\nRUN_DIR={run}\nRESUME={int(resume)} MAX_NEW_SAMPLES={args.max_new_samples}", flush=True)
        if not (args.submit or args.test_only):
            continue
        env = os.environ.copy()
        env.update(PERSIST_ROOT=str(args.persist_root), REPO_ROOT=str(REPO), CONFIG_FILE=config,
                   RUN_DIR=str(run), RESUME=str(int(resume)), MAX_NEW_SAMPLES=str(args.max_new_samples))
        cmd = [sys.executable, str(REPO / "scripts/slurm/submit.py"), "--workflow", "longvideo",
               "--partition", args.partition, "--test-only" if args.test_only else "--parsable"]
        if args.node:
            cmd.extend(["--node", args.node])
        if args.account:
            cmd.extend(["--account", args.account])
        reply = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        if reply.stderr:
            print(reply.stderr, end="", file=sys.stderr)
        print(reply.stdout, end="", flush=True)
        if args.submit:
            record = {"job_id": reply.stdout.strip(), "config": config, "run_dir": str(run), "resume": resume}
            records.append(record)
            # Persist each successful submission even if a later sbatch fails.
            path = REPO / "logs" / f"suite-{args.label}-{record['job_id'].split(';')[0]}.json"
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(record, indent=2) + "\n")
    if not (args.submit or args.test_only):
        print("Dry run only. Use --test-only, then --submit to submit independent jobs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
