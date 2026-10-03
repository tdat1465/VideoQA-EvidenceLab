#!/usr/bin/env bash
# Submit from the repository root. Configure account/storage per school policy.
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=aks-qa-%j.log
set -euo pipefail
: "${EXPERIMENT:?Set EXPERIMENT to an existing paired selection directory}"
: "${VIDEO_ROOT:?Set VIDEO_ROOT to the benchmark video root}"
: "${VENV:?Set VENV to your prepared Python virtual environment}"
: "${HF_HOME:?Set HF_HOME to approved storage for model weights}"
source "$VENV/bin/activate"
hostname
nvidia-smi
python -m pip freeze
extra=()
if [[ "${RESUME:-0}" == 1 ]]; then extra+=(--resume); fi
if [[ "${OFFLINE:-0}" == 1 ]]; then extra+=(--offline); fi
python run_qa_pair.py --experiment "$EXPERIMENT" --video-root "$VIDEO_ROOT" \
  --model "${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}" --revision "${MODEL_REVISION:-main}" \
  --max-pixels "${MAX_PIXELS:-200704}" "${extra[@]}"
python analyze_qa_pair.py --experiment "$EXPERIMENT"
