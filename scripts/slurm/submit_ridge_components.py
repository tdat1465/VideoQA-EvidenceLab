"""One GPU allocation for two cached-score component variants; standard library on login."""
import argparse
import getpass
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from contextlib import closing

from submit import ALLOWED, submission_args

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from evidencelab.config import source_fingerprint
from evidencelab.data import file_hash

ROOT = Path("/media/lnthanh03/DatHa")
BASE_SOURCE = "e92cfc762bde69a511beb494a4b5305bd43e89ccdb72d599de938366da1be057"
BASELINE = ROOT / "runs/videoqa/lvb-val-molmo2-ridge64-full-ridge-v1"
SOURCE = ROOT / "runs/videoqa/lvb-ridge-smoothing-v1-dev200"
OUTPUT = ROOT / "runs/videoqa/lvb-ridge-components-v1-dev200"
sys.path.insert(0, str(REPO/"scripts"))
from ridge_component_ablation import validate_source


def placement_args(nodes, node_mode="any"):
    """One GPU on any eligible pool member, or an explicitly selected member."""
    if node_mode != "any" and node_mode not in ALLOWED:
        raise ValueError("Node mode must be any, gpu01, gpu02 or gpu03")
    node = None if node_mode == "any" else node_mode
    eligible = sorted(set(nodes) & (ALLOWED if node is None else {node}))
    # Excluding the other partition members allows Slurm to choose one node;
    # do not pass a three-host required nodelist for a one-node allocation.
    return submission_args(nodes, node=node, memory_gib=90), eligible


def progress(path):
    if not path.exists():
        return 0, False
    with closing(sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True)) as db:
        rows = [json.loads(value) for (value,) in db.execute("SELECT value FROM results")]
    groups = {}
    for row in rows:
        groups.setdefault(row["question_id"], set()).add(row["variant"])
    return sum(len(v) == 2 for v in groups.values()), len(rows) == 400


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["pilot", "dev200"])
    parser.add_argument("--node", choices=["any"] + sorted(ALLOWED), default="any",
                        help="Default any: one GPU on gpu01, gpu02 or gpu03")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--submit", action="store_true")
    action.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    if getpass.getuser() != "lnthanh03" or REPO != ROOT/"code/VideoQA-EvidenceLab-ridge-v1":
        parser.error("Use the lnthanh03 checkout code/VideoQA-EvidenceLab-ridge-v1")
    if source_fingerprint() != BASE_SOURCE:
        parser.error("Frozen baseline source changed")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "src", "scripts", "configs",
                    "pyproject.toml", "requirements*.txt"], cwd=REPO, check=True, capture_output=True)
    summary = json.loads((BASELINE/"summary.json").read_text())
    if not summary["complete"] or summary["completed_count"] != 1337:
        parser.error("Finish the frozen RIDGE baseline first")
    if (OUTPUT/"contract.json").exists():
        old = json.loads((OUTPUT/"contract.json").read_text())
        if (old["experiment_script_sha256"] != file_hash(REPO/"scripts/ridge_component_ablation.py")
                or old["baseline_weights_sha256"] != file_hash(BASELINE/"ridge-weights.json")
                or old["baseline_predictions_sha256"] != file_hash(BASELINE/"predictions.jsonl")):
            parser.error("Ablation code/baseline changed; use a separately versioned experiment")
    cached = validate_source(SOURCE, BASELINE)
    if (OUTPUT/"contract.json").exists():
        if (old["score_source_contract_sha256"] != file_hash(SOURCE/"contract.json")
                or old["score_source_predictions_sha256"] != file_hash(SOURCE/"predictions.jsonl")
                or old["score_inventory"] != cached["inventory"]):
            parser.error("Source score cache/reference changed; do not resume this run")
    count, complete = progress(OUTPUT/"results.sqlite3")
    if complete or (args.mode == "pilot" and count >= 2):
        print("This mode already has enough results; no new job submitted.")
        return 0
    queue = subprocess.check_output(["squeue", "-h", "-u", "lnthanh03", "-o", "%A|%j"], text=True)
    for line in queue.splitlines():
        job, name = line.strip().split("|", 1)
        if name in ("ridge-sm", "ridge-cmp"):
            parser.error(f"A smoothing job {job} is already active; wait before resubmitting")
    nodes = set(subprocess.check_output(["sinfo", "-N", "-h", "-p", "batch", "-o", "%N"], text=True).split())
    placement, eligible = placement_args(nodes, args.node)
    env = os.environ.copy()
    env.update(PERSIST_ROOT=str(ROOT), REPO_ROOT=str(REPO), RUN_DIR=str(OUTPUT),
               BASELINE_RUN=str(BASELINE), SCORE_CACHE_RUN=str(SOURCE), CONFIG_FILE="configs/lvb_molmo2_ridge64_full.json",
               RESUME=str(int((OUTPUT/"results.sqlite3").exists())),
               MAX_NEW_SAMPLES=str(max(0, 2-count) if args.mode == "pilot" else 0))
    print(json.dumps({"mode":args.mode, "run_dir":str(OUTPUT), "baseline_read_only":str(BASELINE),
                      "completed_questions_all_variants":count, "gpu_count":1, "memory_gib":90,
                      "node_mode":args.node, "eligible_nodes":eligible,
                      "sbatch_placement_args":placement,
                      "variants":["vote_only","rescue_only"], "expected_variant_answers":400,
                      "source_cache_read_only":str(SOURCE), "new_blip_calls":0}, indent=2))
    if not (args.submit or args.test_only):
        return 0
    script = str(REPO/"scripts/slurm/ridge_component_job.sh")
    (REPO/"logs").mkdir(exist_ok=True)
    test = placement + ["--test-only"]
    subprocess.run(test + ["--chdir="+str(REPO), script], env=env, check=True)
    if args.test_only:
        return 0
    if not env.get("HF_TOKEN"):
        env["HF_TOKEN"] = getpass.getpass("Hugging Face read token (hidden): ")
    if not env["HF_TOKEN"]:
        parser.error("HF_TOKEN is empty")
    command = placement + ["--parsable"]
    reply = subprocess.check_output(command + ["--chdir="+str(REPO), script], env=env, text=True).strip()
    job = reply.split(";",1)[0]
    record = {"job_id":job, "mode":args.mode, "run_dir":str(OUTPUT), "baseline":str(BASELINE),
              "source_cache_read_only":str(SOURCE), "expected_variant_answers":400,
              "max_new_questions":int(env["MAX_NEW_SAMPLES"]), "memory_gib":90,
              "node_mode":args.node, "eligible_nodes":eligible, "sbatch_placement_args":placement}
    (REPO/"logs"/f"ridge-components-{job}.json").write_text(json.dumps(record, indent=2)+"\n")
    print(f"Submitted one GPU job {job}; output: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
