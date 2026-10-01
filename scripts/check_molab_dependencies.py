"""Check dependencies and exercise decord's native decoder before MoLab setup succeeds."""
import importlib.metadata
import platform
import subprocess
import sys
import tempfile
from pathlib import Path


def known_decord_tag_mismatch(result, version, wheel, system, machine):
    return (
        result.returncode == 1
        and result.stdout.strip() == "decord 0.6.0 is not supported on this platform"
        and not result.stderr.strip()
        and version == "0.6.0"
        and system == "Linux" and machine == "x86_64"
        and [line for line in wheel.splitlines() if line.startswith("Tag:")]
        == ["Tag: cp36-cp36m-manylinux2010_x86_64"]
    )


def smoke_decode():
    import av
    import decord
    import numpy as np

    decord.bridge.set_bridge("native")
    with tempfile.TemporaryDirectory(prefix="decord-smoke-") as temp:
        path = Path(temp) / "tiny.mp4"
        with av.open(str(path), "w") as container:
            stream = container.add_stream("mpeg4", rate=8)
            stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
            for value in (20, 60, 100, 140):
                frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), value, np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        reader = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=2)
        frames = reader.get_batch([0, 3]).asnumpy()
        if len(reader) != 4 or frames.shape != (2, 48, 64, 3):
            raise RuntimeError("decord decode shape/frame count mismatch")
        if abs(float(frames[0].mean()) - 20) > 10 or abs(float(frames[1].mean()) - 140) > 10:
            raise RuntimeError("decord decode content mismatch")
        del reader
    print("decord native video decode smoke: PASS", flush=True)


def main():
    result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    if result.returncode:
        dist = importlib.metadata.distribution("decord")
        if not known_decord_tag_mismatch(result, dist.version, dist.read_text("WHEEL") or "",
                                        platform.system(), platform.machine()):
            raise subprocess.CalledProcessError(result.returncode, result.args)
        print("Known decord wheel metadata mismatch (internal cp36 tag); requiring native decode smoke.", flush=True)
    smoke_decode()


if __name__ == "__main__":
    main()
