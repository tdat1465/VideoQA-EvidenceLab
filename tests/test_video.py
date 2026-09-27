import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from evidencelab.data import Sample, file_hash
from evidencelab.video import VideoReader


@unittest.skipUnless(importlib.util.find_spec("av"), "install av to exercise actual video decoding")
class VideoTests(unittest.TestCase):
    def test_real_decode_respects_star_clip_and_rejects_changed_video(self):
        import av
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "clip.mp4"
            with av.open(str(path), "w") as out:
                stream = out.add_stream("mpeg4", rate=10)
                stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
                for i in range(40):
                    frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), i*5, dtype=np.uint8), format="rgb24")
                    for packet in stream.encode(frame):
                        out.mux(packet)
                for packet in stream.encode():
                    out.mux(packet)
            s = Sample("q", "star", "val", "clip.mp4", file_hash(path), "What?", ("a", "b"), 0,
                       "Interaction", 1.05, 2.25)
            reader = VideoReader(root, 8)
            frames, times = reader.read(s)
            self.assertTrue(frames)
            self.assertLessEqual(len(frames), 8)
            self.assertTrue(all(s.start <= t < s.end for t in times))
            self.assertEqual(times, sorted(set(times)))
            self.assertIs(reader.read(s)[0], frames)
            with self.assertRaisesRegex(ValueError, "outside video"):
                reader.read(replace(s, end=9))
            with path.open("ab") as f:
                f.write(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                reader.read(s)


if __name__ == "__main__":
    unittest.main()
