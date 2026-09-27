# Copy to a private local file and edit. Do not commit machine paths or tokens.
export REPO_ROOT=/absolute/path/VideoQA-EvidenceLab
export RUN_DIR=/absolute/persistent/path/videoqa-runs/nextqa-val-uniform16
export CONFIG_FILE=configs/molmo2_uniform16.json
export DATASET=nextqa
export SPLIT=val
export ANNOTATIONS_FILE=/absolute/path/annotations/val.csv
export VIDEO_ARCHIVE=/absolute/path/NExTVideo.zip
# Optional if annotation IDs and video filenames differ:
# export VIDEO_MAPPING=/absolute/path/annotations/map_vid_vidorID.json
export RESUME=0
# First session: commit two real examples to verify VRAM and tokens; then resume.
export MAX_NEW_SAMPLES=2
