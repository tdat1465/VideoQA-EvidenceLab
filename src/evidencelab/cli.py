from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from dataclasses import asdict
from pathlib import Path

from .config import Config
from .data import Sample, digest, file_hash, prepare
from .metrics import compare, export
from .runner import run
from .selectors import AKS_COMMIT, AKS_SHA256
from .store import atomic_json, read_run, run_lock


def fetch_aks(output):
    if output.exists():
        if file_hash(output) != AKS_SHA256:
            raise ValueError("Existing AKS file does not match pinned hash")
        return
    url = f"https://raw.githubusercontent.com/ncTimTang/AKS/{AKS_COMMIT}/frame_select.py"
    content = urllib.request.urlopen(url, timeout=60).read()
    import hashlib
    if hashlib.sha256(content).hexdigest() != AKS_SHA256:
        raise ValueError("Upstream AKS hash mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)


def doctor(config=None):
    from .hardware import execution_profile
    from .runner import runtime_versions
    details = {"runtime": runtime_versions(), **execution_profile(config or Config())}
    print(json.dumps(details, indent=2))


def demo(output):
    """Exercise the complete journal/resume path without weights or dataset downloads."""
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "synthetic.jsonl"
    if manifest.exists():
        raise FileExistsError("Use a new --output directory for demo")
    with manifest.open("w", encoding="utf-8") as f:
        for i in range(12):
            sample = Sample(f"synthetic:{i}", "nextqa", "synthetic", f"video{i//3}.mp4", digest(i//3),
                            "Which event happens next?", ("Sit", "Stand", "Walk", "Run", "Stop"), i % 5, "TN")
            f.write(json.dumps(asdict(sample)) + "\n")
    cfg = Config(backend="mock", method="evidence", limit=0, margin_threshold=1)
    atomic_json(output / "mock.json", asdict(cfg))
    code = run(cfg, manifest, output, output / "run", max_new_samples=4)
    assert code == 75
    code = run(cfg, manifest, output, output / "run", resume=True)
    print("SYNTHETIC ONLY: exercised interruption + resume; accuracy is not a model result.")
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description="Frozen video MCQA experiments on one GPU")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--dataset", choices=["nextqa", "star"], required=True)
    for name in ("annotations", "video-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--mapping", type=Path)
    p = sub.add_parser("run")
    for name in ("config", "manifest", "video-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--aks-file", type=Path)
    p.add_argument("--max-seconds", type=float, default=165600)
    p.add_argument("--max-new-samples", type=int, default=0)
    p = sub.add_parser("report")
    p.add_argument("run", type=Path)
    p = sub.add_parser("compare")
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--output", type=Path)
    p = sub.add_parser("fetch-aks")
    p.add_argument("--output", type=Path, default=Path("third_party/aks/frame_select.py"))
    p = sub.add_parser("demo")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("doctor")
    p.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(f"Prepared {prepare(args.dataset, args.annotations, args.video_root, args.split, args.output, args.mapping)} questions")
    elif args.command == "run":
        return run(Config.load(args.config), args.manifest, args.video_root, args.output, args.resume,
                   args.aks_file, args.max_seconds, args.max_new_samples)
    elif args.command == "report":
        with run_lock(args.run):
            result = export(args.run, *read_run(args.run))
        print(json.dumps(result, indent=2))
    elif args.command == "compare":
        if args.iterations < 1:
            parser.error("--iterations must be positive")
        result = compare(args.left, args.right, args.iterations)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(args.output, result)
        print(json.dumps(result, indent=2))
    elif args.command == "fetch-aks":
        fetch_aks(args.output)
        print(f"Verified AKS {AKS_COMMIT}")
    elif args.command == "doctor":
        doctor(Config.load(args.config) if args.config else None)
    else:
        return demo(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
