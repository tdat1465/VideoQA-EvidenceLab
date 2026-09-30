# Source on login01; contains no token. Input HF_TOKEN with read -rsp as documented.
export PERSIST_ROOT=/media/lnthanh03/DatHa
export REPO_ROOT="$PERSIST_ROOT/code/VideoQA-EvidenceLab-focus"
export CONFIG_FILE=configs/lvb_molmo2_focus64.json
export RUN_DIR="$PERSIST_ROOT/runs/videoqa/lvb-val-molmo2-focus64-200"
export RESUME=0
export MAX_NEW_SAMPLES=2
# Submit with: python3 scripts/slurm/submit.py --workflow longvideo --parsable
