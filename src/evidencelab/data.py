"""Normalize public MCQA annotations; never send ground-truth graphs to a model."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def relative_video(value: str) -> str:
    value = str(value).replace("\\", "/")
    p = PurePosixPath(value)
    if not value or p.is_absolute() or ".." in p.parts or ":" in value:
        raise ValueError(f"Video must be a relative path inside video_root: {value!r}")
    return p.as_posix()


@dataclass(frozen=True)
class Question:
    """Only information available at inference time. Deliberately no answer field."""
    id: str
    question: str
    choices: tuple[str, ...]


@dataclass(frozen=True)
class Sample:
    id: str
    dataset: str
    split: str
    video: str
    video_sha256: str
    question: str
    choices: tuple[str, ...]
    answer: int | None
    question_type: str
    start: float = 0.0
    end: float | None = None

    def __post_init__(self):
        relative_video(self.video)
        if not self.id or not self.question.strip() or not self.split:
            raise ValueError("Missing id, question or split")
        if not 2 <= len(self.choices) <= 5 or any(not x.strip() for x in self.choices):
            raise ValueError("Require 2-5 nonempty choices")
        if self.answer is not None and (type(self.answer) is not int or
                                       not 0 <= self.answer < len(self.choices)):
            raise ValueError("Answer must be a zero-based choice index or null")
        if not math.isfinite(self.start) or self.start < 0:
            raise ValueError("Invalid clip start")
        if self.end is not None and (not math.isfinite(self.end) or self.end <= self.start):
            raise ValueError("Clip end must be greater than start")
        if len(self.video_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.video_sha256):
            raise ValueError("Require SHA256 of the actual video bytes")

    def public_question(self) -> Question:
        return Question(self.id, self.question, self.choices)


def load_manifest(path: Path) -> list[Sample]:
    samples, seen = [], set()
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            item["choices"] = tuple(item["choices"])
            sample = Sample(**item)
            if sample.id in seen:
                raise ValueError(f"Duplicate sample id {sample.id}")
            seen.add(sample.id)
            samples.append(sample)
    if not samples:
        raise ValueError("Empty dataset")
    return samples


def video_index(root: Path) -> dict[str, str]:
    """Index filenames, rejecting ambiguous stem mappings rather than guessing."""
    root = root.resolve()
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mkv", ".webm"}:
            if root not in path.resolve().parents:
                raise ValueError("Video symlink escapes video root")
            if path.stem in result:
                raise ValueError(f"Ambiguous video stem {path.stem}; use a separate video root")
            result[path.stem] = path.relative_to(root).as_posix()
    return result


def prepare(dataset: str, annotations: Path, video_root: Path, split: str,
            output: Path, mapping: Path | None = None) -> int:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    idx = video_index(video_root)
    remap = json.loads(mapping.read_text(encoding="utf-8")) if mapping else {}
    if annotations.suffix == ".parquet":
        import pyarrow.parquet as pq
        rows = pq.read_table(annotations).to_pylist()
    elif dataset == "nextqa":
        with annotations.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = json.loads(annotations.read_text(encoding="utf-8-sig"))
    hashed, result, seen = {}, [], set()
    for row in rows:
        if dataset == "nextqa":
            vid = str(row["video"])
            choices = tuple(str(row[f"a{i}"]) for i in range(5))
            answer = int(row["answer"]) if row.get("answer") not in (None, "") else None
            qid = f"nextqa:{split}:{vid}:{row['qid']}"
            category, start, end = str(row["type"]), 0.0, None
        elif dataset == "star":
            vid = str(row["video_id"])
            options = sorted(row["choices"], key=lambda x: int(x["choice_id"]))
            if len({int(x["choice_id"]) for x in options}) != len(options):
                raise ValueError("Duplicate STAR choice_id")
            choices = tuple(x["choice"] for x in options)
            target = row.get("answer")
            answer = None
            if target is not None:
                matches = [i for i, c in enumerate(choices) if c.strip() == str(target).strip()]
                if len(matches) != 1:
                    raise ValueError(f"STAR answer is not one unambiguous choice: {row['question_id']}")
                answer = matches[0]
            qid = f"star:{split}:{row['question_id']}"
            category = row["question_id"].split("_")[0]
            # Full Charades videos must be cropped to the annotated situation.
            start, end = float(row["start"]), float(row["end"])
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        key = str(remap.get(vid, vid))
        relative = idx.get(Path(key).stem)
        if relative is None:
            raise FileNotFoundError(f"No raw video for {vid} (mapped to {key})")
        if relative not in hashed:
            hashed[relative] = file_hash(video_root / relative)
        sample = Sample(qid, dataset, split, relative, hashed[relative], row["question"],
                        choices, answer, category, start, end)
        if sample.id in seen:
            raise ValueError(f"Duplicate sample id {sample.id}")
        seen.add(sample.id)
        result.append(sample)
    if not result:
        raise ValueError("Empty annotation file")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as f:
        for item in result:
            f.write(json.dumps(asdict(item), ensure_ascii=False, allow_nan=False) + "\n")
    return len(result)


def choose_subset(samples: list[Sample], limit: int, seed: int) -> list[Sample]:
    """Stable selection independent of file order; limit=0 means the whole split."""
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    ordered = sorted(samples, key=lambda x: digest([seed, x.id]))
    return ordered[:limit] if limit else ordered
