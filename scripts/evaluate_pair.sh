#!/usr/bin/env bash
# Requires the upstream-compatible lmms-eval/LLaVA-Video environment.
# Generated tasks keep separate immutable annotations and task names.
set -euo pipefail
if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "Usage: bash scripts/evaluate_pair.sh EXPERIMENT_DIR CHECKPOINT K [NPROC=1] [SEED=42]" >&2
  exit 2
fi
experiment=$(cd "$1" && pwd)
checkpoint=$2
budget=$3
nproc=${4:-1}
seed=${5:-42}
python - "$experiment" "$budget" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.loads((root / 'manifest.json').read_text())
if int(sys.argv[2]) != manifest['arguments']['max_num_frames']:
    raise SystemExit('K does not match the selector manifest')
if set(manifest['tasks']) != {'original', 'watershed'}:
    raise SystemExit('Generate tasks using --annotation_path and --video_root first')
PY
for selector in original watershed; do
  task=$(python - "$experiment" "$selector" <<'PY'
import json, sys
from pathlib import Path
print(json.loads((Path(sys.argv[1]) / 'manifest.json').read_text())['tasks'][sys.argv[2]])
PY
)
  output="$experiment/$selector/qa"
  if [[ -e "$output" ]]; then
    echo "QA output already exists: $output; use a fresh experiment to avoid stale checkpoints." >&2
    exit 2
  fi
  accelerate launch --num_processes "$nproc" --main_process_port 12345 -m lmms_eval \
    --model llava_vid \
    --model_args "pretrained=$checkpoint,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=$budget,overwrite=False,use_topk=True,strict_selected_frames=True" \
    --include_path "$experiment/$selector/lmms_task" \
    --tasks "$task" --batch_size 1 --seed "$seed" \
    --log_samples --log_samples_suffix "$selector" --output_path "$output"
done
