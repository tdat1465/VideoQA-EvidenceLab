#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
MODE=${1:-pilot20}
if (( $# > 0 )); then shift; fi
case "$MODE" in pilot20|dev200) ;; *) echo 'Mode: pilot20 or dev200'; exit 2;; esac
[[ $(id -un) = lnthanh03 ]] || { echo 'Use lnthanh03'; exit 2; }
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
[[ "$REPO" = /media/lnthanh03/DatHa/code/VideoQA-EvidenceLab-ridge-v1 ]] || exit 2
cd "$REPO"
mkdir -p /media/lnthanh03/DatHa/metadata/ridge-post-rescue-v1
exec 9>/media/lnthanh03/DatHa/metadata/ridge-post-rescue-v1/submit.lock
flock -n 9 || { echo 'Another post-rescue submission is active'; exit 2; }
export PYTHONDONTWRITEBYTECODE=1
python3 scripts/slurm/submit_ridge_post_rescue.py "$MODE" "$@" --submit
