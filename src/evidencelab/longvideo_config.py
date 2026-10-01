"""Separate protocol for LongVideoBench; old NExT-QA configs stay unchanged."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import re

from .data import digest

FOCUS_REVISION = "d469757cd89976117467294fd1f177026d1a627d"
FOCUS_SHA256 = "404e266328f051c0377688eac72fe5f24b777f52c008bf1103337b261c34beeb"
LLAVA_REVISION = "bce12e479bc4dfee2b9c50c88137b01ff51bd483"
DATASET_REVISION = "60d1c89c1919a198b73be39c2babb213b29d6a5c"
LENS_REVISION = "a3868ab0c50afd34078c350a9daed9382ab2d251"


@dataclass(frozen=True)
class LongVideoConfig:
    backend: str = "molmo2"
    model_id: str = "allenai/Molmo2-4B"
    revision: str = "042abfa7a38879a376cec03d949eff0aefaa0600"
    dtype: str = "auto"
    method: str = "focus"
    frames: int = 64
    limit: int = 200
    seed: int = 18
    selector_seed: int = 42
    max_input_tokens: int = 20000
    max_image_pixels: int = 200704
    # Kept for the existing paired-comparison contract: no fixed candidate pool.
    candidate_frames: int = 0
    prompt_version: str = "letter-score-v1"
    protocol: str = "lvb-video-only-letter-v1"
    dataset_revision: str = DATASET_REVISION
    blip_id: str = "Salesforce/blip-itm-large-coco"
    blip_revision: str = "19502f1e215844f7e48bd48473f86932486d3441"
    blip_batch_size: int = 4
    blip_implementation: str = "hf-itm-fp32-lavis-preprocess-v1"
    vision_id: str = "google/siglip-so400m-patch14-384"
    vision_revision: str = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
    focus_params: dict = field(default_factory=dict)
    lens_implementation: str = "upstream-lazy-1fps-ssim32-v1"

    def __post_init__(self):
        if self.backend not in {"molmo2", "llava_video", "mock"}:
            raise ValueError("LongVideoBench backend must be molmo2, llava_video or mock")
        if self.method not in {"uniform", "focus", "lens"}:
            raise ValueError("LongVideoBench method must be uniform, focus or lens")
        if self.lens_implementation != "upstream-lazy-1fps-ssim32-v1":
            raise ValueError("Unknown LENS implementation")
        if self.dtype not in {"auto", "bfloat16", "float16"}:
            raise ValueError("Invalid dtype")
        for key in ("frames", "max_input_tokens", "max_image_pixels", "blip_batch_size"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"Invalid {key}")
        if self.frames > 64:
            raise ValueError("This initial protocol is limited to 64 frames")
        if type(self.limit) is not int or self.limit < 0:
            raise ValueError("limit=0 means full validation; otherwise use a positive integer")
        if type(self.seed) is not int or type(self.selector_seed) is not int or self.selector_seed < 0:
            raise ValueError("Invalid random seed")
        for key in ("revision", "dataset_revision", "blip_revision", "vision_revision"):
            if not re.fullmatch("[a-f0-9]{40}", getattr(self, key)):
                raise ValueError(f"Pin {key} to a full commit")
        if self.dataset_revision != DATASET_REVISION:
            raise ValueError("Dataset revision requires a reviewed archive layout update")
        if (self.protocol != "lvb-video-only-letter-v1" or self.prompt_version != "letter-score-v1"
                or self.candidate_frames != 0
                or self.blip_implementation != "hf-itm-fp32-lavis-preprocess-v1"):
            raise ValueError("Unknown protocol")
        if self.backend == "llava_video" and self.model_id != "lmms-lab/LLaVA-Video-7B-Qwen2":
            raise ValueError("This backend implements LLaVA-Video-7B-Qwen2")
        # Initially fix upstream defaults; future ablations must explicitly validate params.
        if self.focus_params:
            raise ValueError("Only upstream FOCUS defaults are supported in this first protocol")
        digest(self.contract())

    @classmethod
    def load(cls, path: Path):
        return cls(**json.loads(path.read_text(encoding="utf-8-sig")))

    def contract(self):
        return asdict(self)
