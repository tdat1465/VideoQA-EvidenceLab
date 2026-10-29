#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
[[ $(id -un) = lnthanh03 ]] || { echo 'Use lnthanh03'; exit 2; }
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
[[ "$REPO" = /media/lnthanh03/DatHa/code/VideoQA-EvidenceLab-ridge-v1 ]] || exit 2
cd "$REPO"
mkdir -p /media/lnthanh03/DatHa/metadata/ridge-event-drift-v1
exec 9>/media/lnthanh03/DatHa/metadata/ridge-event-drift-v1/submit.lock
flock -n 9 || { echo 'Another diagnostic submission is active'; exit 2; }
export PYTHONDONTWRITEBYTECODE=1
python3 scripts/slurm/submit_ridge_event_drift.py "$@" --submit
