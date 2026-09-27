"""Submit one 90-GiB job on gpu01/02/03, excluding gpu04 and all other nodes.

Uses exclusions to support Slurm versions where --nodelist requests every
listed host. Requires only the standard library on the login node.
"""
import argparse
import os
import subprocess
from pathlib import Path

ALLOWED = {"gpu01", "gpu02", "gpu03"}


def submission_args(nodes, partition="batch", node=None, account=None, test_only=False, parsable=False):
    allowed = {node} if node else ALLOWED
    if not allowed <= ALLOWED:
        raise ValueError("Only gpu01, gpu02 and gpu03 are allowed")
    if not set(nodes) & allowed:
        raise ValueError(f"Partition {partition} contains none of the requested nodes")
    excluded = sorted((set(nodes) - allowed) | {"gpu04"})
    args = ["sbatch", f"--partition={partition}", "--nodes=1", "--ntasks=1", "--gres=gpu:1",
            "--cpus-per-task=8", "--mem=90G", "--time=2-00:00:00", "--export=ALL",
            "--exclude=" + ",".join(excluded)]
    if node:
        args.append("--nodelist=" + node)
    if account:
        args.append("--account=" + account)
    if test_only:
        args.append("--test-only")
    if parsable:
        args.append("--parsable")
    return args


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--node", choices=sorted(ALLOWED), help="Optional: request one specific allowed node")
    p.add_argument("--partition", default="batch")
    p.add_argument("--account")
    p.add_argument("--test-only", action="store_true")
    p.add_argument("--parsable", action="store_true")
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    os.environ.setdefault("REPO_ROOT", str(repo))
    info = subprocess.run(["sinfo", "-N", "-h", "-p", a.partition, "-o", "%N"],
                          capture_output=True, text=True, check=True)
    nodes = {line.strip() for line in info.stdout.splitlines() if line.strip()}
    args = submission_args(nodes, a.partition, a.node, a.account, a.test_only, a.parsable)
    (repo / "logs").mkdir(exist_ok=True)
    return subprocess.run(args + ["--chdir=" + str(repo), str(repo / "scripts/slurm/job.sh")], cwd=repo).returncode


if __name__ == "__main__":
    raise SystemExit(main())
