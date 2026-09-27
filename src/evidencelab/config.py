from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .data import digest, file_hash


@dataclass(frozen=True)
class Config:
    backend: str = "molmo2"
    model_id: str = "allenai/Molmo2-4B"
    revision: str = "042abfa7a38879a376cec03d949eff0aefaa0600"
    method: str = "uniform"
    frames: int = 16
    initial_frames: int = 8
    candidate_frames: int = 64
    margin_threshold: float = 0.15
    neighbor_seconds: float = 1.0
    relevance_weight: float = 0.25
    max_input_tokens: int = 12000
    max_image_pixels: int = 200704
    clip_id: str = "openai/clip-vit-base-patch32"
    clip_revision: str = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
    clip_device: str = "cpu"
    clip_batch_size: int = 16
    seed: int = 18
    limit: int = 200
    aks_t1: float = 0.8
    aks_t2: float = -100.0
    aks_depth: int = 5
    prompt_version: str = "letter-score-v1"
    notes: str = field(default="", compare=False)

    def __post_init__(self):
        if self.backend not in {"molmo2", "qwen", "mock"}:
            raise ValueError("backend must be molmo2, qwen or mock")
        if self.method not in {"uniform", "clip_topk", "aks", "evidence", "uniform_refine"}:
            raise ValueError("Unknown method")
        for name in ("frames", "initial_frames", "candidate_frames", "max_input_tokens",
                     "max_image_pixels", "clip_batch_size", "aks_depth"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.initial_frames <= self.frames <= self.candidate_frames:
            raise ValueError("Require initial_frames <= frames <= candidate_frames")
        if not 0 <= self.margin_threshold <= 1 or self.neighbor_seconds <= 0:
            raise ValueError("Invalid evidence search parameters")
        if not 0 <= self.relevance_weight <= 1:
            raise ValueError("relevance_weight must be in [0,1]")
        if type(self.limit) is not int or self.limit < 0 or type(self.seed) is not int:
            raise ValueError("Invalid limit/seed")
        if self.clip_device not in {"cpu", "cuda"}:
            raise ValueError("clip_device must be cpu or cuda")
        if self.backend != "mock" and not re.fullmatch(r"[a-f0-9]{40}", self.revision):
            raise ValueError("Pin model revision to a full Hugging Face commit SHA")
        if not re.fullmatch(r"[a-f0-9]{40}", self.clip_revision):
            raise ValueError("Pin CLIP revision to an immutable SHA")
        if self.prompt_version != "letter-score-v1":
            raise ValueError("Unknown prompt version")
        # Reject NaN in every numeric setting.
        digest(asdict(self))

    @classmethod
    def load(cls, path: Path):
        return cls(**json.loads(path.read_text(encoding="utf-8-sig")))

    def contract(self):
        result = asdict(self)
        result.pop("notes")
        return result


def source_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    files = sorted(root.rglob("*.py"))
    repo = root.parent.parent
    extra = [repo / "requirements-server.txt", repo / "pyproject.toml"]
    return digest([(p.relative_to(repo).as_posix(), file_hash(p)) for p in files + extra if p.exists()])
