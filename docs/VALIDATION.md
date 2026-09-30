# Validation status

This repository is an experimental implementation. CPU mechanics tests are separate from GPU/model validation.

## LongVideoBench integration — 2026-09-30

**36/36 CPU tests passed** on Windows, Python 3.12.14, including the existing real PyAV
decode test. New tests exercise the unmodified, SHA-verified FOCUS algorithm on an
18,000-frame synthetic timeline with a deterministic relevance scorer, exact tar/PAX
offsets across multipart boundaries, range validation/fail-closed behavior, cancellation,
annotation/label isolation, deterministic subsets, and complete/interrupted journal equivalence.
Bash syntax checks passed for both Slurm job scripts. Python compilation and git whitespace checks passed.

The pinned official LLaVA-NeXT source was imported with Torch 2.8.0+cpu and
Transformers 4.45.1. `scripts/smoke_llava_cpu.py` passed using tiny random weights:
the actual upstream video expansion produced 21 tokens for three synthetic frames,
and our last-position LM-head logits matched one-token generation. This smoke check
also runs automatically during LLaVA Slurm setup. No pretrained LLaVA weights were loaded locally.
The local test environment reuses existing CPU Torch read-only; it is not the server CUDA environment.

**Not yet verified:** full-model LLaVA or Molmo2 FOCUS generation on the allocated GPU,
BLIP parity with legacy LAVIS, LongVideoBench video decoding/downloads on the school network,
and peak system RAM. The official dataset currently requires accepted access conditions
and an authorized HF token. Anonymous access returned 401, so real annotations, tar headers,
and video bytes were not inspected here; the source revision and 31-part sizes were read
from public metadata, and the range/index implementation was tested against synthetic tar fixtures.
Pilot must pass before collecting research results. No new LongVideoBench accuracy is claimed.

The user reported a completed **NExT-QA Uniform-16 + Molmo2-4B baseline: 173/200 (86.5%)**,
on gpu01/A100 with BF16. This is a prior observed baseline, separate from the new workflow.
Historical validation notes below describe the state at their original dates.

## Historical checks

Checked locally on 2026-09-25 (Python 3.12.14, Windows): **13/13 CPU tests passed**, including real PyAV decoding; both real Molmo2 and Qwen processors accepted irregular timestamp inputs with single-token option letters. Torch was 2.8.0+cpu, torchvision 0.23.0+cpu, transformers 4.57.1. Processor input shapes for three synthetic frames: Molmo2 input IDs `[1,327]` and video patches `[3,729,588]`; Qwen input IDs `[1,254]` and three image grids. Model weights were not loaded. Slurm syntax was checked with Bash; no scheduler execution was performed here.

```bash
pip install -e . numpy==2.2.6 Pillow==11.3.0 av==15.1.0
python -m unittest discover -s tests -v
python -m evidencelab demo --output runs/demo
bash -n scripts/slurm/job.sh
```

Tests cover annotation conversion, choice ordering, exclusion of ground-truth graphs, STAR time windows, actual synthetic-video decoding, altered video rejection, stable subsets, selection budgets, gating and the uniform second-pass control, transactional resume equivalence, contract mismatch rejection, run locking and unsafe archive paths. The actual decoder test is explicitly skipped if PyAV is absent. Demo accuracy is synthetic and has no research interpretation.

GitHub Actions runs the CPU test suite on Python 3.10 and 3.12. It does not download model weights or allocate GPUs. Optional processor-only smoke checks may download processor/tokenizer files, but cannot establish that full model generation works or fits 32 GB VRAM.

```bash
python scripts/smoke_processor.py
python scripts/smoke_processor.py --backend qwen
python -m evidencelab fetch-aks --output third_party/aks/frame_select.py
python scripts/smoke_aks.py third_party/aks/frame_select.py
```

The AKS smoke check compares wrapper output against the pinned upstream implementation on 20 nondegenerate score arrays and exercises constant/short-input guards. It is algorithm parity on synthetic scores, not paper benchmark reproduction.

Rechecked on 2026-09-27: all 13 CPU tests and Bash syntax passed; the actual `fetch-aks` command verified the pinned raw source hash, and all 20 AKS parity cases plus the constant/short-input guards passed.

RAM staging update, 2026-09-27: **22/22 CPU tests passed** after adding download hash/range checks, interrupted-transfer retry and partial-transfer resume, split-only extraction, missing/duplicate video rejection, archive removal, provenance and refusal to download onto an ordinary disk filesystem. HTTP Range inspection of the real pinned NExTVideo ZIP fetched only 607,872 bytes of directory metadata, confirmed 5,440 unique MP4 stems and zero missing validation/test videos. The complete 24.25 GB archive was not downloaded on this workstation; its full hash will be checked in the allocated job. Requesting 90 GiB from Slurm is a proposed allocation, not a measured peak or a verified scheduler entitlement.

Before collecting research results, run the documented two-question GPU pilot and inspect outputs. Then run the fixed 200-question validation subset; only after that commit to full-split evaluation or a claim of improvement.

90-GiB/node-pool update: **28/28 CPU tests passed**. New cases verify that only gpu01/02/03 are eligible, gpu04 and unrelated partition nodes are excluded, exactly one node is requested, forbidden/absent node requests fail, precision selection handles non-BF16 GPUs, Python patch differences are allowed, and cross-node resume refuses a changed precision. FP16/BF16 full-model inference and actual scheduler allocation on those nodes remain untested here.
