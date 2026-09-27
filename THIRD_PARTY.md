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
| [STAR](https://github.com/csbobby/STAR_Benchmark) | Annotation adapter verified against the official JSON schema on 2026-09-25 |
| [A.I.R.](https://github.com/UCF-AIR/A.I.R.) | Conceptual inspiration; no code imported, no claim of reproducing its algorithm |

AKS's inspected repository did not include a LICENSE file. Its source is not redistributed here. `fetch-aks` retrieves the exact upstream file into an ignored directory and verifies SHA256 `594557130aa702fb1c3eafae9df120514d36e9684443b415e0483dce14b4f149` (raw LF bytes; a Git checkout with CRLF conversion has a different hash). Consult the authors regarding reuse/redistribution terms. Short/constant sequence guards and small-budget settings are documented modifications of the evaluation wrapper, not original paper results.

Molmo2 needs `trust_remote_code=True`; the immutable commit makes the executable source explicit. Review upstream licenses/model cards and dataset terms before use. No downloaded weights, external implementation, annotations or videos are included in this repository.
