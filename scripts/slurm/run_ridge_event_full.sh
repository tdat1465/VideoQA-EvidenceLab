#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
MODE=${1:-full1337}
if (( $# > 0 )); then shift; fi
case "$MODE" in full1337) ;; *) echo 'Mode: full1337'; exit 2;; esac
[[ $(id -un) = lnthanh03 ]] || { echo 'Use lnthanh03'; exit 2; }
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
[[ "$REPO" = /media/lnthanh03/DatHa/code/VideoQA-EvidenceLab-ridge-v1 ]] || exit 2
cd "$REPO"
mkdir -p /media/lnthanh03/DatHa/metadata/ridge-event-full1337-v1
exec 9>/media/lnthanh03/DatHa/metadata/ridge-event-full1337-v1/submit.lock
flock -n 9 || { echo 'Another event-coverage submission is active'; exit 2; }
export PYTHONDONTWRITEBYTECODE=1
python3 scripts/slurm/submit_ridge_event_full.py "$MODE" "$@" --submit
