# Provenance and third-party terms

Original code in this repository is MIT licensed. That license does not relicense any downloaded dataset, model weights or upstream code.

| Reference | Pinned version / purpose |
|---|---|
| [tdat1465/DyGEnc](https://github.com/tdat1465/DyGEnc) | `78c7a47cd4e26c92fa31894ec551f058908a5c58`; operational inspiration for tmpfs, Slurm and resume; no model code vendored |
| [allenai/Molmo2-4B](https://huggingface.co/allenai/Molmo2-4B) | `042abfa7a38879a376cec03d949eff0aefaa0600`; remote processor/model code and weights loaded at this revision |
| [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | `cc594898137f460bfe9f0759e9844b3ce807cfb5`; optional multi-image backend |
| [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`; frozen cosine relevance scorer |
| [ncTimTang/AKS](https://github.com/ncTimTang/AKS) | `b0b8a58fedf1d05d78151e2969cecde60e83721d`; runtime import of `frame_select.py` |
| [NExT-QA annotations](https://github.com/doc-doc/NExT-QA) | `2432e9724f88ed9f40010e2989f104570a91de4e`; immutable CSV download |
| [rhymes-ai/NeXTVideo](https://huggingface.co/datasets/rhymes-ai/NeXTVideo) | Third-party video mirror, `7e8ea8e056742292b95688d92a0773e05df00393`; RAM-only download, SHA256 checked before extraction |
| [STAR](https://github.com/csbobby/STAR_Benchmark) | Annotation adapter verified against the official JSON schema on 2026-09-25 |
| [A.I.R.](https://github.com/UCF-AIR/A.I.R.) | Conceptual inspiration; no code imported, no claim of reproducing its algorithm |

AKS's inspected repository did not include a LICENSE file. Its source is not redistributed here. `fetch-aks` retrieves the exact upstream file into an ignored directory and verifies SHA256 `594557130aa702fb1c3eafae9df120514d36e9684443b415e0483dce14b4f149` (raw LF bytes; a Git checkout with CRLF conversion has a different hash). Consult the authors regarding reuse/redistribution terms. Short/constant sequence guards and small-budget settings are documented modifications of the evaluation wrapper, not original paper results.

Molmo2 needs `trust_remote_code=True`; the immutable commit makes the executable source explicit. Review upstream licenses/model cards and dataset terms before use. No downloaded weights, annotations or videos are included in this repository. The licensed FOCUS algorithm below is the only vendored external implementation.

NExTVideo archive SHA256: `2e3b1bc3e761122864b46fe3a1790b281301a6ed69de5ca1fceba17c504fa49c`, size 24,253,160,356 bytes. The hash is the pinned Hugging Face LFS SHA256; full bytes are verified by each GPU staging job. The mirror is also linked by [NVIDIA's dataset preparation notes](https://huggingface.co/datasets/nvidia/Nemotron-VLM-Dataset-v2/blob/main/nextqa/README.md). Original NExT-QA authors' video link remains in their official repository.

## FOCUS / LongVideoBench additions

- `src/evidencelab/_vendor/focus.py` is the **unmodified** algorithm from
  [NUS-HPC-AI-Lab/FOCUS](https://github.com/NUS-HPC-AI-Lab/FOCUS), commit
  `d469757cd89976117467294fd1f177026d1a627d`. SHA256:
  `404e266328f051c0377688eac72fe5f24b777f52c008bf1103337b261c34beeb`.
  Its Apache-2.0 license is included alongside it as `FOCUS-LICENSE.txt`.
- `focus_adapter.py` contains our integration. BLIP scoring uses the Salesforce
  Transformers conversion, explicit `[ENC]`, 35-token truncation and the LAVIS
  evaluation image/caption preprocessing. Numerical parity with legacy LAVIS
  has **not** been established. No AKS code is copied for this scorer.
- LLaVA-NeXT source is fetched into job RAM from commit
  `bce12e479bc4dfee2b9c50c88137b01ff51bd483`, verified by archive SHA256, and
  installed without training dependencies. The source and license are unmodified.
  A scoped loader hook pins the official SigLip tower's revision; weights are frozen.
- LongVideoBench is accessed through its official gated Hugging Face dataset at
  `60d1c89c1919a198b73be39c2babb213b29d6a5c`. Users must accept its access conditions.
  Neither dataset contents nor pretrained model weights are redistributed here.
