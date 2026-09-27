from __future__ import annotations

from pathlib import Path

from .data import Sample, file_hash


def resolve_video(root: Path, sample: Sample) -> Path:
    root = root.resolve()
    path = (root / sample.video).resolve()
    if root not in path.parents or not path.is_file():
        raise FileNotFoundError(f"Video missing or outside video_root: {sample.video}")
    return path


class VideoReader:
    def __init__(self, root: Path, candidates: int, max_side: int = 448):
        self.root, self.candidates, self.max_side = root, candidates, max_side
        self.verified = set()
        self.last_key, self.last_value = None, None

    def read(self, sample: Sample):
        path = resolve_video(self.root, sample)
        stat = path.stat()
        signature = (str(path), stat.st_size, stat.st_mtime_ns, sample.video_sha256)
        if signature not in self.verified:
            if file_hash(path) != sample.video_sha256:
                raise ValueError(f"Video changed since manifest preparation: {sample.video}")
            self.verified.add(signature)
        key = (signature, sample.start, sample.end)
        if key == self.last_key:
            return self.last_value
        import av
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            duration = (float(stream.duration * stream.time_base) if stream.duration is not None
                        else float(container.duration / av.time_base) if container.duration else None)
            if duration is None or duration <= 0:
                raise ValueError(f"Video has no usable duration: {sample.video}")
            end = sample.end if sample.end is not None else duration
            if sample.start >= duration or end > duration + 0.5:
                raise ValueError(f"Annotation clip is outside video: {sample.id}")
            end = min(duration, end)
            # Sample inside bins; never expose frames beyond STAR's situation interval.
            targets = [sample.start + (i + 0.5) * (end - sample.start) / self.candidates
                       for i in range(self.candidates)]
            images, times, cursor = [], [], 0
            if sample.start > 0:
                container.seek(int(sample.start / stream.time_base), stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.time is None:
                    raise ValueError("Decoder did not return presentation timestamps")
                t = float(frame.time)
                if t >= end:
                    break
                if t < sample.start or t < targets[cursor]:
                    continue
                image = frame.to_image().convert("RGB")
                image.thumbnail((self.max_side, self.max_side))
                images.append(image)
                times.append(t)
                while cursor < len(targets) and targets[cursor] <= t:
                    cursor += 1
                if cursor == len(targets):
                    break
        if not images:
            raise ValueError(f"No frames decoded inside clip: {sample.id}")
        self.last_key, self.last_value = key, (images, times)
        return images, times
