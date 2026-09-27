#!/usr/bin/env bash
#SBATCH --job-name=videoqa
#SBATCH --partition=batch
#SBATCH --nodelist=gpu03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Read docs/SERVER.md before submission. Paths and account are explicit env/CLI inputs.
set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit with sbatch, not directly on a login node}"
: "${REPO_ROOT:?Set absolute checkout path}"
: "${RUN_DIR:?Set absolute persistent output directory}"
: "${CONFIG_FILE:?Set config path relative to checkout}"
: "${DATASET:?nextqa or star}"
: "${SPLIT:?val or test}"
: "${ANNOTATIONS_FILE:?Set existing annotation file}"
: "${VIDEO_ARCHIVE:?Set existing zip/tar archive containing original raw videos}"
RESUME=${RESUME:-0}
MAX_NEW_SAMPLES=${MAX_NEW_SAMPLES:-0}
PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python3}
RAM_BASE=${RAM_BASE:-/dev/shm}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}
case "$RESUME" in 0|1) ;; *) echo 'RESUME must be 0 or 1'; exit 2;; esac
[[ "$RUN_DIR" = /* && "$REPO_ROOT" = /* ]] || { echo 'Use absolute paths'; exit 2; }
[[ -f "$ANNOTATIONS_FILE" && -f "$VIDEO_ARCHIVE" && -f "$REPO_ROOT/$CONFIG_FILE" ]]
[[ $(findmnt -n -o FSTYPE --target "$RAM_BASE") = tmpfs ]] || { echo 'RAM_BASE must be tmpfs'; exit 2; }
if findmnt -n -o OPTIONS --target "$RAM_BASE" | grep -qw noexec; then
  echo 'RAM filesystem is noexec; choose an executable tmpfs via RAM_BASE'; exit 2
fi
mkdir -p "$RUN_DIR"
# Hold throughout setup and execution, separately from the Python journal lock.
exec 9>"$RUN_DIR/session.lock"
flock -n 9 || { echo 'Another session owns this run'; exit 2; }
if [[ "$RESUME" = 1 ]]; then
  [[ -f "$RUN_DIR/results.sqlite3" ]] || { echo 'No run journal to resume'; exit 2; }
else
  [[ ! -f "$RUN_DIR/results.sqlite3" ]] || { echo 'Existing run: set RESUME=1'; exit 2; }
fi

TASK_RAM=$(mktemp -d "$RAM_BASE/videoqa.${SLURM_JOB_ID}.XXXXXX")
child=''
stop_requested=0
cleanup() {
  # TASK_RAM is the exact mktemp directory, not a user-supplied cleanup target.
  case "$TASK_RAM" in "$RAM_BASE"/videoqa."$SLURM_JOB_ID".*) rm -rf -- "$TASK_RAM";; esac
}
request_stop() {
  stop_requested=1
  if [[ -n "$child" ]]; then kill -USR1 "$child" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap request_stop USR1 TERM INT
export HF_HOME="$TASK_RAM/hf" XDG_CACHE_HOME="$TASK_RAM/cache" TMPDIR="$TASK_RAM/tmp"
export PIP_CACHE_DIR="$TASK_RAM/pip" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# Slurm owns CUDA_VISIBLE_DEVICES; never replace it with a physical GPU number.
mkdir -p "$TMPDIR" "$TASK_RAM/repo"
cp -a "$REPO_ROOT/src" "$REPO_ROOT/scripts" "$REPO_ROOT/pyproject.toml" \
  "$REPO_ROOT/requirements-server.txt" "$TASK_RAM/repo/"
cp "$REPO_ROOT/$CONFIG_FILE" "$TASK_RAM/config.json"
"$PYTHON_BIN" -m venv "$TASK_RAM/venv"
PY="$TASK_RAM/venv/bin/python"
"$PY" -m pip install --no-cache-dir 'pip==25.2' 'setuptools==80.9.0' 'wheel==0.45.1'
"$PY" -m pip install --no-cache-dir 'torch==2.8.0' 'torchvision==0.23.0' --index-url "$TORCH_INDEX"
"$PY" -m pip install --no-cache-dir -r "$TASK_RAM/repo/requirements-server.txt"
"$PY" -m pip install --no-build-isolation --no-deps -e "$TASK_RAM/repo"
"$PY" -m evidencelab doctor
"$PY" -m pip freeze > "$RUN_DIR/environment.latest.txt"
nvidia-smi > "$RUN_DIR/gpu.latest.txt"
(( stop_requested == 0 )) || exit 75
mapping_args=()
if [[ -n "${VIDEO_MAPPING:-}" ]]; then mapping_args=(--mapping "$VIDEO_MAPPING"); fi
"$PY" "$TASK_RAM/repo/scripts/stage_data.py" --archive "$VIDEO_ARCHIVE" \
  --target "$TASK_RAM/data" --annotations "$ANNOTATIONS_FILE" --dataset "$DATASET" --split "$SPLIT" \
  "${mapping_args[@]}"
"$PY" -m evidencelab fetch-aks --output "$TASK_RAM/aks/frame_select.py"
(( stop_requested == 0 )) || exit 75
# Capture the exact inputs used by this session; SQLite remains the resume authority.
cp "$TASK_RAM/config.json" "$RUN_DIR/config.latest.json"
args=(--config "$TASK_RAM/config.json" --manifest "$TASK_RAM/data/manifest.jsonl" \
  --video-root "$TASK_RAM/data/videos" --output "$RUN_DIR" --max-seconds 165600 \
  --max-new-samples "$MAX_NEW_SAMPLES" --aks-file "$TASK_RAM/aks/frame_select.py")
if [[ "$RESUME" = 1 ]]; then args+=(--resume); fi
"$PY" -m evidencelab run "${args[@]}" &
child=$!
set +e
wait "$child"
status=$?
# A signal interrupts bash wait even while Python finishes its current transaction.
while kill -0 "$child" 2>/dev/null; do
  wait "$child"
  status=$?
done
child=''
set -e
if [[ "$status" = 75 ]]; then
  echo 'Session stopped cleanly. Submit the same RUN_DIR with RESUME=1.'
elif [[ "$status" != 0 ]]; then
  echo "Run failed ($status). Inspect logs; completed questions remain committed."
fi
exit "$status"
