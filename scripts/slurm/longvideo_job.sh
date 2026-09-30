#!/usr/bin/env bash
#SBATCH --job-name=lvbqa
#SBATCH --partition=batch
#SBATCH --exclude=gpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Read docs/LONGVIDEOBENCH.md before submission. Paths and account are explicit env/CLI inputs.
set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit with sbatch, not directly on a login node}"
# submit.py excludes other nodes before allocation; this is a final guard.
NODE_NAME=${SLURMD_NODENAME:-$(hostname -s)}
case "$NODE_NAME" in gpu01|gpu02|gpu03) ;; *) echo "Unsupported node: $NODE_NAME; use submit.py for gpu01-03 only"; exit 2;; esac
: "${REPO_ROOT:?Set absolute checkout path}"
: "${RUN_DIR:?Set absolute persistent output directory}"
: "${CONFIG_FILE:?Set config path relative to checkout}"
RESUME=${RESUME:-0}
MAX_NEW_SAMPLES=${MAX_NEW_SAMPLES:-0}
PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python3}
RAM_BASE=${RAM_BASE:-/dev/shm}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}
MIN_RAM_FREE_GIB=${MIN_RAM_FREE_GIB:-64}
case "$RESUME" in 0|1) ;; *) echo 'RESUME must be 0 or 1'; exit 2;; esac
[[ "$RUN_DIR" = /* && "$REPO_ROOT" = /* ]] || { echo 'Use absolute paths'; exit 2; }
[[ -f "$REPO_ROOT/$CONFIG_FILE" ]]
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
export HF_XET_CACHE="$TASK_RAM/hf/xet"
unset TRANSFORMERS_CACHE
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120
export PYTORCH_KERNEL_CACHE_PATH="$TASK_RAM/kernel-cache"
export TORCHINDUCTOR_CACHE_DIR="$TASK_RAM/inductor"
export PIP_CACHE_DIR="$TASK_RAM/pip" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PIP_CONFIG_FILE=/dev/null
unset PIP_TARGET PIP_PREFIX PIP_USER
export UV_CACHE_DIR="$TASK_RAM/uv" TORCH_HOME="$TASK_RAM/torch" CUDA_CACHE_PATH="$TASK_RAM/cuda"
export TORCH_EXTENSIONS_DIR="$TASK_RAM/torch-extensions" NUMBA_CACHE_DIR="$TASK_RAM/numba"
export TRITON_CACHE_DIR="$TASK_RAM/triton" MPLCONFIGDIR="$TASK_RAM/matplotlib"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# Slurm owns CUDA_VISIBLE_DEVICES; never replace it with a physical GPU number.
mkdir -p "$TMPDIR" "$TASK_RAM/repo" "$PYTORCH_KERNEL_CACHE_PATH" "$CUDA_CACHE_PATH"
echo "GPU job $SLURM_JOB_ID on $(hostname); temporary RAM directory: $TASK_RAM"
"$PYTHON_BIN" "$REPO_ROOT/scripts/ram_preflight.py" --ram-root "$TASK_RAM" --min-free-gib "$MIN_RAM_FREE_GIB"
cp -a "$REPO_ROOT/src" "$REPO_ROOT/scripts" "$REPO_ROOT/pyproject.toml" \
  "$REPO_ROOT"/requirements*.txt "$TASK_RAM/repo/"
cp "$REPO_ROOT/$CONFIG_FILE" "$TASK_RAM/config.json"
export PYTHONPATH="$TASK_RAM/repo/src"
BACKEND=$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; from evidencelab.longvideo_config import LongVideoConfig; print(LongVideoConfig.load(Path(sys.argv[1])).backend)' "$TASK_RAM/config.json")
case "$BACKEND" in
  molmo2) REQUIREMENTS=requirements-longvideo-molmo.txt;;
  llava_video) REQUIREMENTS=requirements-longvideo-llava.txt;;
  *) echo 'Slurm workflow requires a real backend'; exit 2;;
esac
"$PYTHON_BIN" -m venv "$TASK_RAM/venv"
PY="$TASK_RAM/venv/bin/python"
run_child "$PY" -m pip install --no-cache-dir 'pip==25.2' 'setuptools==80.9.0' 'wheel==0.45.1'
run_child "$PY" -m pip install --no-cache-dir 'requests==2.32.5' 'huggingface-hub==0.35.3'
# Fail on missing dataset authorization/range support before downloading Torch or weights.
run_child "$PY" -m evidencelab.longvideo prepare --config "$TASK_RAM/config.json" --ram-root "$TASK_RAM/data"
run_child "$PY" -m pip install --no-cache-dir 'torch==2.8.0' 'torchvision==0.23.0' --index-url "$TORCH_INDEX"
run_child "$PY" -m pip install --no-cache-dir -r "$TASK_RAM/repo/$REQUIREMENTS"
run_child "$PY" -m pip install --no-build-isolation --no-deps -e "$TASK_RAM/repo"
if [[ "$BACKEND" = llava_video ]]; then
  run_child "$PY" "$TASK_RAM/repo/scripts/fetch_llava_source.py" --target "$TASK_RAM/llava"
  run_child "$PY" -m pip install --no-build-isolation --no-deps -e "$TASK_RAM/llava"
  run_child "$PY" "$TASK_RAM/repo/scripts/smoke_llava_cpu.py"
fi
"$PY" -m pip check
"$PY" -m evidencelab.longvideo doctor --config "$TASK_RAM/config.json"
"$PY" -m pip freeze > "$RUN_DIR/environment.latest.txt"
nvidia-smi > "$RUN_DIR/gpu.latest.txt"
(( stop_requested == 0 )) || exit 75
# Capture the exact inputs used by this session; SQLite remains the resume authority.
cp "$TASK_RAM/config.json" "$RUN_DIR/config.latest.json"
# One small persistent header index is shared across jobs; video bytes never are.
: "${PERSIST_ROOT:?Set /media/lnthanh03/DatHa}"
INDEX_FILE="$PERSIST_ROOT/metadata/longvideobench/60d1c89c1919a198b73be39c2babb213b29d6a5c/index.json"
# Setup/download time also consumes the 48h allocation. Leave five minutes for cleanup.
ELAPSED=$SECONDS
REMAINING=$((172800 - ELAPSED - 300))
(( REMAINING > 0 )) || exit 75
args=(--config "$TASK_RAM/config.json" --annotations "$TASK_RAM/data/lvb_val.json"
  --ram-root "$TASK_RAM/data" --output "$RUN_DIR" --index "$INDEX_FILE"
  --max-seconds "$REMAINING" --max-new-samples "$MAX_NEW_SAMPLES")
if [[ "$RESUME" = 1 ]]; then args+=(--resume); fi
child_signal=USR1
status=0
run_child "$PY" -m evidencelab.longvideo run "${args[@]}" || status=$?
if [[ "$status" = 75 ]]; then
  echo 'Session stopped cleanly. Submit the same RUN_DIR with RESUME=1.'
elif [[ "$status" != 0 ]]; then
  echo "Run failed ($status). Inspect logs; completed questions remain committed."
fi
exit "$status"
