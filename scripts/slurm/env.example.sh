# Save a private copy outside the checkout; edit root if needed.
export PERSIST_ROOT=/media/lnthanh03/DatHa
export REPO_ROOT="$PERSIST_ROOT/code/VideoQA-EvidenceLab"
export RUN_DIR="$PERSIST_ROOT/runs/videoqa/nextqa-val-uniform16-ram"
export CONFIG_FILE=configs/molmo2_uniform16.json
export DATA_MODE=nextqa_auto
export DATASET=nextqa
export SPLIT=val
export PYTHON_BIN=/usr/bin/python3
export RAM_BASE=/dev/shm
export MIN_RAM_FREE_GIB=64
export RESUME=0
# First session commits two real questions; then resume with MAX_NEW_SAMPLES=0.
export MAX_NEW_SAMPLES=2
# No VIDEO_ARCHIVE or ANNOTATIONS_FILE needed in nextqa_auto mode.
# DATA_MODE=local_archive remains available for user-supplied STAR/other archives.
