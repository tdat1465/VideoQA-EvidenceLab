"""LENS research adapter: verified upstream core, lazy video, bounded SSIM batches.

See docs/LENS.md for changes from upstream and separate third-party licenses.
"""
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

from .data import file_hash
from .focus_adapter import question_seed
from .longvideo_config import LENS_REVISION
from .longvideo_data import check_space

MANIFEST_SHA = "360c1a5557e777a6d70fbb8016860580a75d6875f9f16d96939cb7733e68e27a"
MASK_SHA = "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"
MASK_URL = f"https://openaipublic.azureedge.net/clip/models/{MASK_SHA}/ViT-L-14-336px.pt"


def load_lens():
    root = Path(__file__).parent / "_vendor/lens"
    if file_hash(root / "manifest.json") != MANIFEST_SHA:
        raise ValueError("LENS manifest hash mismatch")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["revision"] != LENS_REVISION:
        raise ValueError("LENS revision mismatch")
    for name, expected in manifest["files"].items():
        if file_hash(root / name) != expected:
            raise ValueError(f"LENS source hash mismatch: {name}")
    loaded = []
    for name in ("sampling", "prompts"):
        spec = importlib.util.spec_from_file_location(f"evidencelab_lens_{name}", root / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded.append(module)
    # The unmodified masking source uses absolute API_CLIP imports.
    for name, module in list(sys.modules.items()):
        if name.startswith("API_CLIP"):
            for location in getattr(module, "__path__", [getattr(module, "__file__", "")]):
                if not Path(location).resolve().is_relative_to(root.resolve()):
                    raise RuntimeError("Another API_CLIP package shadows the pinned LENS source")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return loaded


def parse_ratio(response):
    """Upstream float parsing and 0.5 fallback, with explicit NaN/Inf handling."""
    try:
        value = float(response.strip())
        if not math.isfinite(value):
            raise ValueError("nonfinite ratio")
    except (ValueError, TypeError):
        return .5, True
    return max(0., min(1., value)), False


def candidate_indices(total, fps):
    import numpy as np
    if total < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid video metadata")
    # Match upstream LLaVA loader's approximately 1 fps, without get_batch(all).
    step = max(1, round(fps))
    if (total + step - 1) // step > 9999:
        return np.linspace(0, total - 1, 9999, dtype=int).tolist()
    return list(range(0, total, step))


def lens_query(question):
    # Upstream LVB selection prompt includes choices, never the correct label.
    options = "".join(f"({chr(65+i)}) {c}\n" for i, c in enumerate(question.choices))
    return f"Question: {question.question}\nOptions:\n{options}Respond with only the letter of the correct option.\n"


def neighbors(anchor, count):
    lo, hi = max(0, anchor - 1), min(count, anchor + 3)
    positions = list(range(lo, hi))
    while len(positions) < min(4, count):
        if lo > 0:
            lo -= 1
            positions.insert(0, lo)
        elif hi < count:
            positions.append(hi)
            hi += 1
    # Explicit short-video boundary padding, rather than crashing on <4 frames.
    while len(positions) < 4:
        positions.append(positions[-1])
    return positions[:4]


def hyperframe(images):
    import cv2
    import numpy as np
    from PIL import Image
    arrays = [np.asarray(image) for image in images]
    height, width = arrays[0].shape[:2]
    grid = np.concatenate((np.concatenate(arrays[:2], axis=1),
                           np.concatenate(arrays[2:], axis=1)), axis=0)
    return Image.fromarray(cv2.resize(grid, (width, height), interpolation=cv2.INTER_LINEAR))


def bounded_ssim(frames, device, batch_size=32, guard=lambda: None):
    """Upstream Gaussian SSIM equation, with at most batch_size pairs on GPU.

    Input is CPU float32 at the exact upstream area-resized resolution <=224.
    Local statistics stay on CPU; only each working batch resides on the GPU.
    """
    import torch
    import torch.nn.functional as F
    count, channels, _, _ = frames.shape
    coords = torch.arange(11, device=device).float() - 5
    g = torch.exp(-coords.square() / (2 * 1.5**2))
    g /= g.sum()
    kernel = (g.view(1, 1, 11, 1) * g.view(1, 1, 1, 11)).expand(channels, 1, 11, 11).contiguous()
    means, variances = torch.empty_like(frames), torch.empty_like(frames)
    data_range = 255. if frames.max() > 1 else 1.
    c1, c2 = (.01 * data_range)**2, (.03 * data_range)**2
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            guard()
            batch = frames[start:start+batch_size].to(device)
            mean = F.conv2d(batch, kernel, padding=5, groups=channels)
            means[start:start+batch_size] = mean.cpu()
            variances[start:start+batch_size] = (F.conv2d(batch.square(), kernel, padding=5,
                                                       groups=channels) - mean.square()).cpu()
        matrix = torch.empty((count, count), dtype=torch.float32)
        for i in range(count):
            guard()
            frame_i, mu_i, var_i = (x[i:i+1].to(device) for x in (frames, means, variances))
            for start in range(0, count, batch_size):
                batch, mu, var = (x[start:start+batch_size].to(device) for x in (frames, means, variances))
                product = mu_i * mu
                cov = F.conv2d(frame_i * batch, kernel, padding=5, groups=channels) - product
                value = ((2*product+c1) * (2*cov+c2) /
                         ((mu_i.square()+mu.square()+c1) * (var_i+var+c2))).mean(dim=(1, 2, 3))
                matrix[i, start:start+batch_size] = value.cpu()
    return (matrix + 1) / 2


def download_mask(root, guard):
    import requests
    path = root / "lens-clip.pt"
    if path.exists():
        if file_hash(path) != MASK_SHA:
            raise ValueError("CLIP checkpoint checksum mismatch")
        return path
    partial = path.with_suffix(".partial")
    try:
        with requests.get(MASK_URL, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            with partial.open("wb", buffering=0) as output:
                sha, size = hashlib.sha256(), 0
                for block in response.iter_content(4 * 1024**2):
                    guard()
                    size += len(block)
                    if size > 2 * 1024**3:
                        raise ValueError("CLIP checkpoint exceeds size guard")
                    output.write(block)
                    sha.update(block)
        if sha.hexdigest() != MASK_SHA:
            raise ValueError("CLIP checkpoint checksum mismatch")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    return path


class LensSelector:
    def __init__(self, config, ram_root, stop=lambda: False):
        import torch
        from transformers import BlipForImageTextRetrieval, BlipProcessor
        self.config, self.root, self.stop = config, ram_root, stop
        self.sampling, self.prompts = load_lens()
        from API_CLIP.clip_prs.utils.openai_models import load_openai_model
        from API_CLIP.clip_prs.utils.transform import image_transform
        from API_CLIP.clip_prs.utils.tokenizer import tokenize
        from API_CLIP.hook import hook_prs_logger
        from API_CLIP.main import gen_mask, merge_mask, invtrans, merge, toImg
        self.mask_functions = gen_mask, merge_mask, invtrans, merge, toImg
        self.guard()
        self.processor = BlipProcessor.from_pretrained(config.blip_id, revision=config.blip_revision)
        self.blip = BlipForImageTextRetrieval.from_pretrained(
            config.blip_id, revision=config.blip_revision, torch_dtype=torch.float32).to("cuda:0")
        self.blip.eval().requires_grad_(False)
        checkpoint = download_mask(ram_root, self.guard)
        self.clip = load_openai_model(str(checkpoint), precision="fp32", device="cuda:0")
        self.clip.eval().requires_grad_(False)
        self.preprocess, self.tokenizer = image_transform(336, is_train=False), tokenize
        self.prs = hook_prs_logger(self.clip, "cuda:0", layer_index=22)

    def guard(self, incoming=0):
        if self.stop():
            raise InterruptedError("Stop requested during LENS selection")
        check_space(self.root, incoming)

    def mask(self, image, query):
        import torch
        from PIL import Image
        gen_mask, merge_mask, invtrans, merge, toImg = self.mask_functions
        self.guard()
        with torch.inference_mode():
            _, attention, tokens = gen_mask(self.clip, self.prs, self.preprocess, "cuda:0",
                                            self.tokenizer, [image], [query])
            # Reject degenerate masks; do not silently turn NaNs into black pixels.
            mask = merge_mask(attention[0], tokens[0], kernel_size=3, enhance_coe=10)
            if not torch.isfinite(mask).all():
                raise ValueError("LENS CLIP mask is nonfinite")
            mask = invtrans(toImg(mask.cpu().unsqueeze(0)), image, method=Image.Resampling.BICUBIC)
            result = merge(mask.convert("L"), image.convert("RGB"), 200).convert("RGB")
        self.prs.reinit()
        return result

    def select(self, video, question, backend):
        import torch
        import torch.nn.functional as F
        from PIL import Image
        pool = candidate_indices(len(video), float(video.get_avg_fps()))
        query = lens_query(question)
        self.guard()
        allocation = backend.generate_text(self.prompts.ALLOCATION_PROMPT.format(
            USER_QUERY=question.question, TOTAL_BUDGET=self.config.frames), max_new_tokens=16)
        ratio, fallback = parse_ratio(allocation["text"])
        spatial_budget = round(ratio * self.config.frames)
        temporal_budget = self.config.frames - spatial_budget
        scores = []
        for start in range(0, len(pool), self.config.blip_batch_size):
            self.guard()
            images = [Image.fromarray(video[i].asnumpy()).convert("RGB")
                      for i in pool[start:start+self.config.blip_batch_size]]
            # Match LENS's HF BLIP processor, unlike the FOCUS LAVIS-preprocess port.
            inputs = self.processor(images=images, text=[query]*len(images), padding=True,
                                    return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                logits = self.blip(**inputs).itm_score
                scores.extend(logits.softmax(-1)[:, 1].cpu().tolist())
            del images, inputs, logits
        scores = torch.tensor(scores, dtype=torch.float32)
        if not torch.isfinite(scores).all():
            raise ValueError("Invalid LENS relevance scores")
        seed = question_seed(self.config.selector_seed, question.id)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            spatial = self.sampling.select_anchor_frames_watershed(scores, spatial_budget)
            if len(spatial) < spatial_budget:
                spatial = scores.topk(min(spatial_budget, len(pool))).indices.sort().values.tolist()
            pre = self.sampling.uniform_sampling(pool, query, min(len(pool), max(128, 8*self.config.frames)))
            temporal, refined = [], None
            if temporal_budget:
                small = []
                for i in pre:
                    self.guard()
                    pixels = torch.from_numpy(video[pool[i]].asnumpy()).permute(2, 0, 1).float()[None]
                    h, w = pixels.shape[-2:]
                    if max(h, w) > 224:
                        scale = 224 / max(h, w)
                        pixels = F.interpolate(pixels, (max(1, int(h*scale)), max(1, int(w*scale))), mode="area")
                    small.append(pixels[0])
                matrix = bounded_ssim(torch.stack(small), "cuda:0", guard=self.guard)
                del small, pixels
                # CPU graph avoids upstream's many GPU scalar synchronizations.
                # Same graph equation and parameters; record this implementation.
                refined, _, _, _ = self.sampling._graph_diffusion_refine(scores[pre], matrix, .8, 10, .3)
                anchors = self.sampling.select_anchor_frames_watershed(refined, self.config.frames)
                if len(anchors) < self.config.frames:
                    anchors = refined.topk(min(self.config.frames, len(pre))).indices.sort().values.tolist()
                ordered = torch.tensor(anchors)[refined[anchors].argsort(descending=True)].tolist()
                for i in ordered:
                    if len(temporal) >= temporal_budget:
                        break
                    if pre[i] not in spatial:
                        temporal.append(i)
                for i in ordered:
                    if len(temporal) >= temporal_budget:
                        break
                    if i not in temporal:
                        temporal.append(i)
        views = []
        for i in spatial:
            self.guard()
            image = Image.fromarray(video[pool[i]].asnumpy()).convert("RGB")
            views.append((pool[i], self.mask(image, query), "spatial", [pool[i]]))
        for i in temporal:
            self.guard()
            sources = [pool[pre[j]] for j in neighbors(i, len(pre))]
            images = [Image.fromarray(video[j].asnumpy()).convert("RGB") for j in sources]
            views.append((pool[pre[i]], hyperframe(images), "temporal", sources))
        if not views:
            raise ValueError("LENS returned no views")
        views.sort(key=lambda view: view[0])
        indices = [v[0] for v in views]
        details = {"lens_revision": LENS_REVISION, "lens_ratio": ratio,
                   "lens_ratio_fallback": fallback, "lens_allocation": allocation,
                   "auxiliary_vlm_calls": 1, "auxiliary_input_tokens": allocation["input_tokens"],
                   "lens_spatial_budget": spatial_budget, "lens_temporal_budget": temporal_budget,
                   "lens_spatial_views": len(spatial), "lens_temporal_views": len(temporal),
                   "lens_candidate_frames": len(pool), "lens_graph_candidates": len(pre) if temporal_budget else 0,
                   "lens_views": [{"kind": v[2], "anchor": v[0], "source_frames": v[3]} for v in views],
                   "source_frame_presentations": sum(len(v[3]) for v in views),
                   "unique_source_frames": len({i for v in views for i in v[3]}),
                   "scorer_frame_evaluations": len(pool), "scorer_unique_frames": len(pool),
                   "scorer_batches": math.ceil(len(pool)/self.config.blip_batch_size),
                   "selector_rng_seed": seed, "frame_index_space": "source-video-anchors-of-transformed-views"}
        return indices, [v[1] for v in views], details
