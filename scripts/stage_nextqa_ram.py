"""Must run inside the allocated GPU job, with --target on tmpfs."""
import argparse
from pathlib import Path

from evidencelab.staging import stage_nextqa

p = argparse.ArgumentParser()
p.add_argument("--target", type=Path, required=True)
p.add_argument("--split", choices=["train", "val", "test"], default="val")
p.add_argument("--provenance", type=Path, required=True)
a = p.parse_args()
stage_nextqa(a.target, a.split, a.provenance)
