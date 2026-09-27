#!/usr/bin/env bash
#SBATCH --job-name=videoqa
#SBATCH --partition=batch
#SBATCH --nodelist=gpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
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
DATA_MODE=${DATA_MODE:-nextqa_auto}
RESUME=${RESUME:-0}
MAX_NEW_SAMPLES=${MAX_NEW_SAMPLES:-0}
PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python3}
RAM_BASE=${RAM_BASE:-/dev/shm}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}
MIN_RAM_FREE_GIB=${MIN_RAM_FREE_GIB:-64}
case "$RESUME" in 0|1) ;; *) echo 'RESUME must be 0 or 1'; exit 2;; esac
[[ "$RUN_DIR" = /* && "$REPO_ROOT" = /* ]] || { echo 'Use absolute paths'; exit 2; }
[[ -f "$REPO_ROOT/$CONFIG_FILE" ]]
case "$DATA_MODE" in
  nextqa_auto)
    [[ "$DATASET" = nextqa ]] || { echo 'nextqa_auto requires DATASET=nextqa'; exit 2; }
    ;;
  local_archive)
    : "${ANNOTATIONS_FILE:?Set existing annotation file for local_archive mode}"
    : "${VIDEO_ARCHIVE:?Set existing video archive for local_archive mode}"
    [[ -f "$ANNOTATIONS_FILE" && -f "$VIDEO_ARCHIVE" ]]
    ;;
  *) echo 'DATA_MODE must be nextqa_auto or local_archive'; exit 2;;
esac
RAM_BASE=$(readlink -f -- "$RAM_BASE")
[[ $(findmnt -n -o FSTYPE --target "$RAM_BASE") = tmpfs ]] || { echo 'RAM_BASE must be tmpfs'; exit 2; }
if findmnt -n -o OPTIONS --target "$RAM_BASE" | grep -qw noexec; then
  echo 'RAM filesystem is noexec; choose an executable tmpfs via RAM_BASE'; exit 2
fi
mkdir -p "$RUN_DIR"
RUN_DIR=$(readlink -f -- "$RUN_DIR")
[[ "$RUN_DIR/" != "$RAM_BASE/"* ]] || { echo 'RUN_DIR must survive RAM cleanup'; exit 2; }
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
child_signal=TERM
stop_requested=0
cleanup() {
  # TASK_RAM is the exact mktemp directory, not a user-supplied cleanup target.
  case "$TASK_RAM" in "$RAM_BASE"/videoqa."$SLURM_JOB_ID".*) rm -rf -- "$TASK_RAM";; esac
}
request_stop() {
  stop_requested=1
  if [[ -n "$child" ]]; then kill -"$child_signal" "$child" 2>/dev/null || true; fi
}
run_child() {
  (( stop_requested == 0 )) || return 75
  "$@" &
  child=$!
  set +e
  wait "$child"
  local status=$?
  while kill -0 "$child" 2>/dev/null; do
    wait "$child"
    status=$?
  done
  child=''
  set -e
  if (( stop_requested != 0 && status != 0 )); then return 75; fi
  return "$status"
}
trap cleanup EXIT
trap request_stop USR1 TERM INT
export HF_HOME="$TASK_RAM/hf" XDG_CACHE_HOME="$TASK_RAM/cache" TMPDIR="$TASK_RAM/tmp"
export HF_HUB_CACHE="$TASK_RAM/hf/hub" HUGGINGFACE_HUB_CACHE="$TASK_RAM/hf/hub"
export HF_MODULES_CACHE="$TASK_RAM/hf/modules" HF_DATASETS_CACHE="$TASK_RAM/hf/datasets"
export HF_XET_CACHE="$TASK_RAM/hf/xet" TRANSFORMERS_CACHE="$TASK_RAM/hf/transformers"
export PIP_CACHE_DIR="$TASK_RAM/pip" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PIP_CONFIG_FILE=/dev/null
unset PIP_TARGET PIP_PREFIX PIP_USER
export UV_CACHE_DIR="$TASK_RAM/uv" TORCH_HOME="$TASK_RAM/torch" CUDA_CACHE_PATH="$TASK_RAM/cuda"
export TORCH_EXTENSIONS_DIR="$TASK_RAM/torch-extensions" NUMBA_CACHE_DIR="$TASK_RAM/numba"
export TRITON_CACHE_DIR="$TASK_RAM/triton" MPLCONFIGDIR="$TASK_RAM/matplotlib"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# Slurm owns CUDA_VISIBLE_DEVICES; never replace it with a physical GPU number.
mkdir -p "$TMPDIR" "$TASK_RAM/repo"
echo "GPU job $SLURM_JOB_ID on $(hostname); temporary RAM directory: $TASK_RAM"
"$PYTHON_BIN" "$REPO_ROOT/scripts/ram_preflight.py" --ram-root "$TASK_RAM" --min-free-gib "$MIN_RAM_FREE_GIB"
cp -a "$REPO_ROOT/src" "$REPO_ROOT/scripts" "$REPO_ROOT/pyproject.toml" \
  "$REPO_ROOT/requirements-server.txt" "$TASK_RAM/repo/"
cp "$REPO_ROOT/$CONFIG_FILE" "$TASK_RAM/config.json"
"$PYTHON_BIN" -m venv "$TASK_RAM/venv"
PY="$TASK_RAM/venv/bin/python"
run_child "$PY" -m pip install --no-cache-dir 'pip==25.2' 'setuptools==80.9.0' 'wheel==0.45.1'
run_child "$PY" -m pip install --no-cache-dir 'torch==2.8.0' 'torchvision==0.23.0' --index-url "$TORCH_INDEX"
run_child "$PY" -m pip install --no-cache-dir -r "$TASK_RAM/repo/requirements-server.txt"
run_child "$PY" -m pip install --no-build-isolation --no-deps -e "$TASK_RAM/repo"
"$PY" -m evidencelab doctor
"$PY" -m pip freeze > "$RUN_DIR/environment.latest.txt"
nvidia-smi > "$RUN_DIR/gpu.latest.txt"
(( stop_requested == 0 )) || exit 75
if [[ "$DATA_MODE" = nextqa_auto ]]; then
  run_child "$PY" "$TASK_RAM/repo/scripts/stage_nextqa_ram.py" --target "$TASK_RAM/data" \
    --split "$SPLIT" --provenance "$RUN_DIR/data-source.json"
else
  mapping_args=()
  if [[ -n "${VIDEO_MAPPING:-}" ]]; then mapping_args=(--mapping "$VIDEO_MAPPING"); fi
  run_child "$PY" "$TASK_RAM/repo/scripts/stage_data.py" --archive "$VIDEO_ARCHIVE" \
    --target "$TASK_RAM/data" --annotations "$ANNOTATIONS_FILE" --dataset "$DATASET" --split "$SPLIT" \
    "${mapping_args[@]}"
fi
run_child "$PY" -m evidencelab fetch-aks --output "$TASK_RAM/aks/frame_select.py"
(( stop_requested == 0 )) || exit 75
# Capture the exact inputs used by this session; SQLite remains the resume authority.
cp "$TASK_RAM/config.json" "$RUN_DIR/config.latest.json"
args=(--config "$TASK_RAM/config.json" --manifest "$TASK_RAM/data/manifest.jsonl" \
  --video-root "$TASK_RAM/data/videos" --output "$RUN_DIR" --max-seconds 165600 \
  --max-new-samples "$MAX_NEW_SAMPLES" --aks-file "$TASK_RAM/aks/frame_select.py")
if [[ "$RESUME" = 1 ]]; then args+=(--resume); fi
child_signal=USR1
status=0
run_child "$PY" -m evidencelab run "${args[@]}" || status=$?
if [[ "$status" = 75 ]]; then
  echo 'Session stopped cleanly. Submit the same RUN_DIR with RESUME=1.'
elif [[ "$status" != 0 ]]; then
  echo "Run failed ($status). Inspect logs; completed questions remain committed."
fi
exit "$status"
