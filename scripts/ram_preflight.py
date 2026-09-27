"""Lightweight preflight before installing packages or downloading data."""
import argparse
import json
import os
import shutil
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--ram-root", type=Path, required=True)
p.add_argument("--min-free-gib", type=float, default=64)
a = p.parse_args()
gib = 1024**3
info = {}
for line in Path("/proc/meminfo").read_text().splitlines():
    key, value = line.split(":", 1)
    if key == "MemAvailable":
        info["host_mem_available_gib"] = int(value.split()[0]) * 1024 / gib
info["tmpfs_free_gib"] = shutil.disk_usage(a.ram_root).free / gib
allocated = os.environ.get("SLURM_MEM_PER_NODE")
if allocated:
    info["slurm_requested_memory_gib"] = int(allocated) / 1024
print(json.dumps(info, indent=2), flush=True)
if a.min_free_gib <= 0 or any(value < a.min_free_gib for value in info.values()):
    raise SystemExit(f"Need at least {a.min_free_gib:g} GiB for this profile; no fallback to disk")
