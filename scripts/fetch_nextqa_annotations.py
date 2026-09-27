"""Fetch small official MCQA CSVs at an immutable Git commit. No video download."""
import argparse
import urllib.request
from pathlib import Path

REVISION = "2432e9724f88ed9f40010e2989f104570a91de4e"

p = argparse.ArgumentParser()
p.add_argument("--split", choices=["train", "val", "test"], default="val")
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
for name in (a.split + ".csv", "map_vid_vidorID.json"):
    target = a.output / name
    if target.exists():
        raise FileExistsError(target)
    url = f"https://raw.githubusercontent.com/doc-doc/NExT-QA/{REVISION}/dataset/nextqa/{name}"
    with urllib.request.urlopen(url, timeout=60) as src, target.open("xb") as dst:
        dst.write(src.read())
    print(target)
