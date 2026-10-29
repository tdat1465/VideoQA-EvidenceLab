"""Lossless, per-video tmpfs frames from two agreeing sequential decode passes."""
from __future__ import annotations
import hashlib
from pathlib import Path
import shutil

POLICY = {'decoder': 'decord', 'api': 'next()', 'threads': 1, 'passes': 2,
          'storage': 'lossless RGB PNG; no resize', 'max_cache_bytes': 32 * 1024**3}


def fingerprint(array):
    if str(array.dtype) != 'uint8' or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError('Require native uint8 RGB frames')
    return {'sha256': hashlib.sha256(array.tobytes()).hexdigest(), 'shape': list(array.shape)}


def sequential_frames(path, count, fps):
    import decord
    video = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
    if len(video) != count or float(video.get_avg_fps()) != fps:
        raise ValueError('Decoder metadata changed')
    while True:
        try:
            frame = video.next()
        except StopIteration:
            return
        yield frame


class Frame:
    def __init__(self, array):
        self.array = array

    def asnumpy(self):
        return self.array


class FrameCache:
    """Random access only to verified lossless images, never to a video decoder."""
    def __init__(self, root, manifest):
        self.root, self.manifest = Path(root), manifest

    def __len__(self):
        return self.manifest['source_frames']

    def get_avg_fps(self):
        return self.manifest['fps']

    def __getitem__(self, index):
        import numpy as np
        from PIL import Image
        index = int(index)
        expected = self.manifest['frames'][str(index)]
        with Image.open(self.root / f'{index}.png') as image:
            array = np.array(image.convert('RGB'))
        if fingerprint(array) != expected:
            raise ValueError('Canonical cached pixels changed')
        return Frame(array)


def build(path, root, count, fps, indices, video_sha, guard, stop=lambda: False,
          factory=sequential_frames):
    """Verify EOF/count and every requested native pixel in two fresh passes."""
    from PIL import Image
    indices = sorted(set(indices))
    if not indices or indices[0] < 0 or indices[-1] >= count or count <= 0 or fps <= 0:
        raise ValueError('Invalid canonical frame grid')
    root = Path(root)
    root.mkdir()  # caller owns an exact temporary child, never a shared directory
    manifest = {'policy': POLICY, 'source_frames': count, 'fps': fps, 'video_sha256': video_sha,
                'indices': indices, 'frames': {}}
    wanted, cache_bytes = set(indices), 0
    try:
        for repeat in range(2):
            stream, observed = iter(factory(path, count, fps)), 0
            try:
                for index, native in enumerate(stream):
                    if stop():
                        raise InterruptedError('Stop during canonical decode')
                    observed += 1
                    if observed > count:
                        raise ValueError('Sequential decoder returned too many frames')
                    if index not in wanted:
                        continue
                    array = native.asnumpy()
                    actual = fingerprint(array)
                    if repeat:
                        if actual != manifest['frames'][str(index)]:
                            raise ValueError(f'Sequential repeat pixels differ at frame {index}')
                    else:
                        guard(root, array.nbytes * 2)
                        target = root / f'{index}.png'
                        Image.fromarray(array).save(target, compress_level=1)
                        cache_bytes += target.stat().st_size
                        if cache_bytes > POLICY['max_cache_bytes']:
                            raise MemoryError('Canonical cache exceeds 32 GiB; no disk fallback')
                        manifest['frames'][str(index)] = actual
                    del array
            finally:
                if hasattr(stream, 'close'):
                    stream.close()
            if observed != count or len(manifest['frames']) != len(indices):
                raise ValueError('Sequential EOF/count/grid mismatch')
        # Read back every PNG before allowing any scoring or answer inference.
        cache = FrameCache(root, manifest)
        for index in indices:
            if stop():
                raise InterruptedError('Stop during canonical cache verification')
            cache[index]
        return cache
    except BaseException:
        shutil.rmtree(root)
        raise
