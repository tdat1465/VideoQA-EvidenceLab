import hashlib
import io
import json
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from evidencelab.data import load_manifest
from evidencelab.staging import download_verified, extract_split, require_tmpfs, stage_nextqa


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status, self.headers = status, headers or {}


class DownloadTests(unittest.TestCase):
    def test_interrupted_transfer_retries_from_committed_partial_bytes(self):
        class Interrupted(Response):
            def read(self, n=-1):
                if self.tell() == 0:
                    return super().read(2)
                raise OSError("connection lost")
        payload = b"abcdef"
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "videos.zip"
            responses = [Interrupted(payload), Response(payload[2:], 206, {"Content-Range": "bytes 2-5/6"})]
            with patch("evidencelab.staging.urllib.request.urlopen", side_effect=responses) as urlopen, \
                 patch("evidencelab.staging.time.sleep"):
                download_verified("https://example.invalid", output, 6, hashlib.sha256(payload).hexdigest())
            self.assertEqual(output.read_bytes(), payload)
            self.assertEqual(urlopen.call_args.args[0].get_header("Range"), "bytes=2-")

    def test_download_and_resume_verify_bytes(self):
        payload = b"video fixture bytes"
        sha = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "videos.zip"
            output.with_suffix(".zip.part").write_bytes(payload[:5])
            with patch("evidencelab.staging.urllib.request.urlopen",
                       return_value=Response(payload[5:], 206, {"Content-Range": "bytes 5-18/19"})) as urlopen:
                download_verified("https://example.invalid/video", output, len(payload), sha)
                self.assertEqual(urlopen.call_args.args[0].get_header("Range"), "bytes=5-")
            self.assertEqual(output.read_bytes(), payload)
            self.assertFalse(output.with_suffix(".zip.part").exists())

    def test_server_ignores_range_restarts_without_duplicate_bytes(self):
        payload = b"abcdefghij"
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "videos.zip"
            output.with_suffix(".zip.part").write_bytes(b"abc")
            with patch("evidencelab.staging.urllib.request.urlopen", return_value=Response(payload)):
                download_verified("https://example.invalid", output, len(payload), hashlib.sha256(payload).hexdigest())
            self.assertEqual(output.read_bytes(), payload)

    def test_wrong_hash_and_wrong_range_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "videos.zip"
            with patch("evidencelab.staging.urllib.request.urlopen", return_value=Response(b"abc")):
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    download_verified("https://example.invalid", output, 3, "0" * 64)
            self.assertFalse(output.exists())
            with patch("evidencelab.staging.urllib.request.urlopen",
                       return_value=Response(b"def", 206, {"Content-Range": "bytes 0-2/6"})):
                with self.assertRaisesRegex(ValueError, "different byte range"):
                    download_verified("https://example.invalid", output, 6, "0" * 64)

    def test_disk_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch("evidencelab.staging.subprocess.run",
                       return_value=subprocess.CompletedProcess([], 0, "ext4\n", "")):
                with self.assertRaisesRegex(ValueError, "outside tmpfs"):
                    require_tmpfs(Path(temp))


class SplitStagingTests(unittest.TestCase):
    def test_extract_only_split_videos(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with zipfile.ZipFile(root / "all.zip", "w") as z:
                z.writestr("NExTVideo/01/100.mp4", b"validation")
                z.writestr("NExTVideo/01/200.mp4", b"training")
            result = extract_split(root / "all.zip", root / "videos", {"100"}, reserve_gib=0)
            self.assertEqual(result["video_count"], 1)
            self.assertEqual(len(list((root / "videos").rglob("*.mp4"))), 1)
            self.assertEqual((root / "videos/NExTVideo/01/100.mp4").read_bytes(), b"validation")

    def test_missing_and_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with zipfile.ZipFile(root / "missing.zip", "w") as z:
                z.writestr("100.mp4", b"one")
            with self.assertRaisesRegex(ValueError, "missing"):
                extract_split(root / "missing.zip", root / "missing", {"200"}, reserve_gib=0)
            with zipfile.ZipFile(root / "duplicate.zip", "w") as z:
                z.writestr("a/100.mp4", b"one")
                z.writestr("b/100.mp4", b"two")
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                extract_split(root / "duplicate.zip", root / "duplicate", {"100"}, reserve_gib=0)

    def test_archive_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with zipfile.ZipFile(root / "bad.zip", "w") as z:
                z.writestr("../100.mp4", b"escape")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                extract_split(root / "bad.zip", root / "videos", {"100"}, reserve_gib=0)

    def test_full_staging_removes_zip_preserves_all_questions_and_provenance(self):
        csv = ("video,qid,question,answer,type,a0,a1,a2,a3,a4\n"
               "100,1,Why?,0,CW,a,b,c,d,e\n100,2,When?,1,TN,a,b,c,d,e\n").encode()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with zipfile.ZipFile(root / "fixture.zip", "w") as z:
                z.writestr("NExTVideo/01/100.mp4", b"video bytes")
            def download(url, destination, size, sha):
                shutil.copyfile(root / "fixture.zip", destination)
            with patch("evidencelab.staging.require_tmpfs"), \
                 patch("evidencelab.staging.shutil.disk_usage", return_value=shutil._ntuple_diskusage(256*1024**3, 0, 256*1024**3)), \
                 patch("evidencelab.staging.urllib.request.urlopen", return_value=Response(csv)), \
                 patch("evidencelab.staging.download_verified", side_effect=download):
                stage_nextqa(root / "ram", "val", root / "data-source.json")
            self.assertFalse((root / "ram/NExTVideo.zip").exists())
            rows = load_manifest(root / "ram/manifest.jsonl")
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(s.split == "val" for s in rows))
            self.assertEqual(json.loads((root / "data-source.json").read_text())["video_count"], 1)


if __name__ == "__main__":
    unittest.main()
