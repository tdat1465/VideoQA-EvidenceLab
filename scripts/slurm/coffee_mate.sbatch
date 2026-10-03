#!/usr/bin/env bash
#SBATCH --job-name=coffee-mate
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=coffee-mate-%j.log
set -euo pipefail
: "${REPO_ROOT:?Absolute Coffee-Mate checkout path}"
: "${COMA_VENV:?Prepared isolated Python environment}"
: "${COMA_PREPARED:?Prepared recipe directory}"
: "${HF_HOME:?Cache containing bert-base-uncased}"
cd "$REPO_ROOT"
source "$COMA_VENV/bin/activate"
nvidia-smi
extra=()
if [[ "${COMA_RESUME:-0}" == 1 ]]; then extra+=(--resume); fi
python scripts/coffee_mate.py train --prepared "$COMA_PREPARED" "${extra[@]}"
# Explicit final epoch, no hidden best-on-validation checkpoint search.
COMA_CHECKPOINT="$COMA_PREPARED/training/ckpt_02.pth"
# One GPU avoids DistributedSampler padding/duplicating validation questions.
# Fresh evaluation when training resumed, unless evaluation already exists.
eval_extra=()
if [[ -f "$COMA_PREPARED/evaluation-final/eval_contract.json" ]]; then eval_extra+=(--resume); fi
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} python scripts/coffee_mate.py evaluate \
  --prepared "$COMA_PREPARED" --checkpoint "$COMA_CHECKPOINT" \
  --output "$COMA_PREPARED/evaluation-final" "${eval_extra[@]}"
