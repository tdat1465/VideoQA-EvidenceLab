"""Download only pinned upstream code into the job's RAM, verify and safely unpack."""
import argparse
import hashlib
import io
import zipfile
import urllib.request
from pathlib import Path, PurePosixPath

from evidencelab.longvideo import validate_ram_root
from evidencelab.longvideo_config import LLAVA_REVISION

SHA256 = "bab99d2383f9920d97170ed8a50a5b005d44c299f3ac1508ad24db5ae4feebfe"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", required=True, type=Path)
    a = p.parse_args()
    validate_ram_root(a.target.parent)
    url = f"https://codeload.github.com/LLaVA-VL/LLaVA-NeXT/zip/{LLAVA_REVISION}"
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read(32 * 1024**2 + 1)
    if hashlib.sha256(content).hexdigest() != SHA256:
        raise ValueError("Pinned LLaVA source archive hash mismatch")
    a.target.mkdir(exist_ok=False)
    prefix = f"LLaVA-NeXT-{LLAVA_REVISION}/"
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for item in z.infolist():
            if not item.filename.startswith(prefix):
                raise ValueError("Invalid source prefix")
            name = item.filename[len(prefix):]
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
                raise ValueError("Unsafe source path")
            if not (name.startswith("llava/") or name in {"pyproject.toml", "README.md", "LICENSE"}):
                continue
            target = a.target.joinpath(*path.parts)
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(item))
    print(f"Verified LLaVA-NeXT source {LLAVA_REVISION}")


if __name__ == "__main__":
    main()
