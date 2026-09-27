# Validation status

This repository is an experimental implementation. CPU mechanics tests are separate from GPU/model validation. No NExT-QA/STAR accuracy, GPU memory, or runtime result has been obtained on the school's GPU yet.

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

RAM staging update, 2026-09-27: **22/22 CPU tests passed** after adding download hash/range checks, interrupted-transfer retry and partial-transfer resume, split-only extraction, missing/duplicate video rejection, archive removal, provenance and refusal to download onto an ordinary disk filesystem. HTTP Range inspection of the real pinned NExTVideo ZIP fetched only 607,872 bytes of directory metadata, confirmed 5,440 unique MP4 stems and zero missing validation/test videos. The complete 24.25 GB archive was not downloaded on this workstation; its full hash will be checked in the allocated job. Requesting 192 GiB from Slurm is a proposed allocation, not a measured peak or a verified scheduler entitlement.

Before collecting research results, run the documented two-question GPU pilot and inspect outputs. Then run the fixed 200-question validation subset; only after that commit to full-split evaluation or a claim of improvement.
