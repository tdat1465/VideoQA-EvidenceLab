#!/usr/bin/env bash
# Run after git clone / git pull on the school server. No bundle is required.
set -Eeuo pipefail
umask 077
trap 'echo "Dung tai dong $LINENO; xem loi phia tren." >&2' ERR
MODE=${1:-pilot}
case "$MODE" in pilot|dev200|full) ;; *) echo 'Mode: pilot, dev200, full'; exit 2;; esac
[[ $(id -un) = lnthanh03 ]] || { echo 'Can dung tai khoan lnthanh03'; exit 2; }
export PERSIST_ROOT=/media/lnthanh03/DatHa
export REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
[[ "$REPO_ROOT" = "$PERSIST_ROOT/code/VideoQA-EvidenceLab-ridge-v1" ]] || {
  echo 'Dung checkout rieng code/VideoQA-EvidenceLab-ridge-v1'; exit 2;
}
cd "$REPO_ROOT"
[[ $(git symbolic-ref --short HEAD) = codex/ridge-molmo2 ]] || {
  echo 'Sai branch: can codex/ridge-molmo2'; exit 2;
}
git diff --quiet HEAD -- src scripts configs pyproject.toml 'requirements*.txt' || {
  echo 'STOP: source/config da sua; khong submit code chua commit.'; exit 2;
}
EXPECTED_COMMIT=$(git rev-parse HEAD)
export RIDGE_EXPECTED_SOURCE=e92cfc762bde69a511beb494a4b5305bd43e89ccdb72d599de938366da1be057
mkdir -p "$PERSIST_ROOT/metadata/ridge-v1"
exec 9>"$PERSIST_ROOT/metadata/ridge-v1/submit.lock"
flock -n 9 || { echo 'Mot phien khac dang submit RIDGE'; exit 2; }
export RIDGE_MODE="$MODE"
# Pilot and full share the FULL 1337-question contract and run directory.
# The cap only limits new questions in one allocation; dev200 is a separate scope.
PLAN=$(python3 - <<'PY'
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

repo = Path(os.environ['REPO_ROOT'])
root = Path(os.environ['PERSIST_ROOT'])
sys.path.insert(0, str(repo / 'src'))
from evidencelab.config import source_fingerprint
if source_fingerprint() != os.environ['RIDGE_EXPECTED_SOURCE']:
    raise SystemExit('STOP: source fingerprint khong khop package.')
spec = importlib.util.spec_from_file_location('ridge_suite', repo / 'scripts/slurm/submit_longvideo_suite.py')
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)
mode = os.environ['RIDGE_MODE']
with contextlib.redirect_stdout(sys.stderr):
    plans = suite.plans(root, 'molmo2', 64, ['ridge', 'uniform'], 'ridge-v1', True,
                        limit=200 if mode == 'dev200' else 0)
targets = {run for _, run, _ in plans}
queue = subprocess.check_output(['squeue', '-h', '-u', 'lnthanh03', '-o', '%A|%j'], text=True)
for line in queue.splitlines():
    job, name = line.strip().split('|', 1)
    for record in (root / 'code').glob(f'*/logs/suite-*-{job}.json'):
        if Path(json.loads(record.read_text())['run_dir']) in targets:
            raise SystemExit(f'STOP: job {job} dang chay/cho cung RUN_DIR; khong submit trung.')
for config_file, run, resume in plans:
    count = 0
    if resume:
        with contextlib.closing(sqlite3.connect((run / 'results.sqlite3').resolve().as_uri() + '?mode=ro', uri=True)) as db:
            count = db.execute('SELECT COUNT(*) FROM results').fetchone()[0]
    method = suite.LongVideoConfig.load(repo / config_file).method
    if mode == 'pilot' and count >= 2:
        print(f'SKIP pilot da co {count} cau: {run}', file=sys.stderr)
        continue
    print(f'{method}|{max(0, 2-count) if mode == "pilot" else 0}')
PY
)
if [[ -z "$PLAN" ]]; then
  echo 'Mode nay da du ket qua; khong submit them.'
  exit 0
fi
echo "RIDGE package $EXPECTED_COMMIT; mode=$MODE; RAM=90 GiB/job"
if [[ -z "${HF_TOKEN:-}" ]]; then
  read -rsp 'Hugging Face read token: ' HF_TOKEN
  echo
fi
[[ -n "$HF_TOKEN" ]] || { echo 'HF_TOKEN rong'; exit 2; }
export HF_TOKEN
SUITE_ARGS=(--backend molmo2 --frames 64 --label ridge-v1 --resume
            --mem-gib 90 --persist-root "$PERSIST_ROOT")
if [[ "$MODE" != dev200 ]]; then SUITE_ARGS+=(--full-validation); fi
while IFS='|' read -r method new_samples; do
  python3 scripts/slurm/submit_longvideo_suite.py "${SUITE_ARGS[@]}" \
    --methods "$method" --max-new-samples "$new_samples" --test-only
done <<< "$PLAN"
while IFS='|' read -r method new_samples; do
  python3 scripts/slurm/submit_longvideo_suite.py "${SUITE_ARGS[@]}" \
    --methods "$method" --max-new-samples "$new_samples" --submit
done <<< "$PLAN"
squeue -u lnthanh03 -n lvbqa
