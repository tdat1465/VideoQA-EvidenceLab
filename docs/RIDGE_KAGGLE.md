# RIDGE cached ablation on Kaggle T4 x2

This is a **new FP16 v2 experiment**, with Molmo2 weights sharded across both T4 GPUs using Accelerate's `balanced` mapping. It uses a separate v2 output and contract; v1 Kaggle and BF16 A100 answer journals remain separate. The 376 existing school answers remain in their original run; finish the remaining 24 on the school server if completing that experiment.

The provided importable notebook is `outputs/ridge-kaggle-v1/RIDGE_Kaggle_T4x2.ipynb` in the local workspace, accompanied by `ridge-kaggle-input.zip`. The current ZIP contains approximately 7.2 MB of code, author selection source, baseline metadata, precomputed BLIP scores, cached Qwen preference weights and an archive-safe vocabulary copy. It contains no video bytes, answer model weights or credentials. Upload it to a **private** Kaggle Dataset and attach that Dataset to the notebook. No upload has been performed by the assistant. Older 5.9 MB bundles remain supported.

## Notebook setup

1. Create a Kaggle notebook and import `RIDGE_Kaggle_T4x2.ipynb` using its notebook import/upload action.
2. Select **GPU T4 x2** and turn **Internet** on. Both GPUs must remain visible. Availability and quotas depend on the account; inspect the session panel rather than assuming a fixed weekly allowance.
3. Create a private Kaggle Dataset from `ridge-kaggle-input.zip` and use **Add Input** to attach it. The notebook accepts both the unchanged ZIP and files automatically expanded by Kaggle, finding the bundle manifest under `/kaggle/input`.
4. Add the existing Hugging Face read token to Kaggle **Secrets** under `HF_TOKEN`, then enable that secret for the notebook. Ensure this HF account has accepted LongVideoBench access conditions. Never paste a token into code or output.
5. Run the cells in order. The notebook kernel may use a newer Python version: uv 0.8.22 downloads a separate Python 3.12.11 and creates `/kaggle/temp/ridge-v1/venv-py312-v2`. That worker environment installs Torch 2.8.0 with CUDA 12.6 and the existing pinned Molmo requirements. The notebook kernel and its packages remain unchanged. Internet access is required for the Python download as well as dependencies.

The first cell prints GPU names and checks the input ZIP's hashes. The preflight validates all 200 source caches and historical baseline consistency. The backend then verifies two T4 GPUs and native FP16 CUDA operations. CPU/disk weight offload is not accepted by this profile; videos and model downloads use Kaggle's temporary disk, separately from saved results.

The notebook carries a hash-checked copy of the v2 runner. After verifying all input bundle files, it installs that runner in the working copy. Existing input Datasets can therefore be reused without uploading code again. The actual runner hash and video I/O profile participate in the v2 resume contract.

## Host RAM during video processing

The initial port retained 64 original-resolution PIL images and then built another large array with `np.stack`. A Kaggle session was reported killed for host memory after model load; the precise failing native allocation was not captured. These duplicate RGB buffers are a concrete source of avoidable peak RAM.

Version 2 decodes original RGB frames one at a time into a scratch file, releases the Decord reader, and passes a writable NumPy disk mapping directly to the unchanged native video processor. It keeps all 64 selected frames, original pixel values, timestamps, resizing settings and FP16 scoring. Decoder concurrency is reduced to one thread. The scratch filesystem must not be tmpfs/ramfs. Frame staging requires enough disk space plus a 4 GiB reserve, and may be slower than the original path.

Phase logs record worker RSS and container cgroup usage/headroom when readable. A 1.25 GiB headroom guard stops between allocations and lets the runner save results. It cannot guarantee protection against an abrupt native allocation or platform termination. Disk mappings still use reclaimable page cache; this is not zero-RAM processing. A new pilot is required on T4 x2.

If Dataset processing expands the nested CLIP `.txt.gz` asset, the loader restores its exact bytes from a base64 sidecar in newer bundles. With older bundles, it downloads only that missing asset from the fixed OpenAI CLIP commit `a1d071733d7111c9c014f024669f959182114e33` and requires its original SHA256. All bundle checks still run. The existing input Dataset can therefore be reused; missing score/code files and modified assets are rejected.

## Pilot and full run

Initially keep `MODE = 'pilot'` in the last code cell. It processes the first two questions, producing **10/1,000** answers. This checks that the actual Molmo2 forward path works on T4 x2. CPU checks alone cannot establish GPU compatibility, memory sufficiency or numerical stability.

After the pilot has finite probabilities, 64-frame outputs and no model-sharding error, change the same cell to `MODE = 'full'` and rerun it. It reads the existing SQLite journal and processes the remaining 198 questions without repeating committed answers. The model is loaded again for this second worker, while downloaded files remain cached during the same Kaggle session. If an error occurs, retain its log and checkpoint before changing the protocol.

The default experiment runs five variants, serially within one process:

| Variant | Purpose |
|---|---|
| `sigma2` | Baseline rerun in the same FP16 Kaggle environment |
| `sigma1` | Best fixed-sigma configuration from the earlier BF16 development results, rerun as a tuning control |
| `multiscale_v1` | Combined prototype rerun under the same precision |
| `vote_only` | Region-label consensus without fine-peak rescue |
| `rescue_only` | Original sigma2 labels plus supported fine-peak rescue |

Target: **200 questions × 5 = 1,000 answers**, each using the frozen 64-frame budget. These are new Kaggle results. No BLIP scoring or Qwen weight generation is performed; both are explicitly historical cached inputs. The checkpoint does not import the school's BF16 answer journal or treat old predictions as new controls.

No quantization, lower frame budget, silent truncation or precision fallback is enabled. FP16 answer probabilities are checked for finiteness before commit. The input prompt, video processor and option-letter scoring come from the frozen backend. The portable entry point has its own platform and precision contract, without falsifying Slurm environment variables or weakening the school runner's guards.

## Checkpoint and continuation

Outputs:

```text
/kaggle/working/ridge-kaggle-v2/run/summary.json
/kaggle/working/ridge-kaggle-v2/run/predictions.jsonl
/kaggle/working/ridge-kaggle-v2/run/results.sqlite3
/kaggle/working/ridge-kaggle-v2/run/curves/
/kaggle/working/ridge-kaggle-v2/ridge-kaggle-checkpoint.zip
/kaggle/working/ridge-kaggle-v2/session.log
```

SQLite commits each variant independently. Human-readable exports and a transaction-consistent ZIP update at run initialization, after every question and on worker exit. The ZIP contains the answer journal, contract, curves, exports and archive metadata index when available. It excludes model weights, HF caches, active video files, RGB scratch mappings and tokens.

The worker defaults to six hours, then stops between variants/questions and exports. This is a worker budget, not a promise the entire experiment fits one session. Kaggle's session limit also includes environment setup and downloads. The notebook's Interrupt handler asks the worker to stop safely; a forced platform termination can prevent the latest export/ZIP even though SQLite has more committed rows.

Before ending a session, download the checkpoint ZIP and session log or save a notebook version with outputs. For a new session, attach the original input bundle and a private Dataset containing the latest `ridge-kaggle-checkpoint.zip`; the restore cell accepts the ZIP or its automatically expanded contents. Keep the notebook/code and environment frozen. Runtime or device-map changes cause contract mismatch rather than mixed results. Kaggle cannot recover unsaved working storage after it is discarded.

The first archive-index build may take time because videos are read by bounded HTTP Range from the official multipart archive. Only one active video is stored at a time. The metadata index is included in subsequent checkpoints so it can be reused. No full dataset archive is staged.

Completion is `completed_variant_answers=1000`, `complete=true`, and `n=200` for every variant. Compare the five variants on identical question IDs **within this FP16 run**. FP16-vs-BF16 or T4-vs-A100 comparisons are separate environment analyses, not selector-only ablations. Cached answer timings exclude prior BLIP/Qwen and do not represent end-to-end speedups.

## Validation and current limitation

The port has nine runner tests and four notebook tests covering contracts, hardware mapping, interrupted transactions, restore/resume, finite probabilities, exact staged RGB order, decoder release, cgroup guards and the direct mapped-array processor call. The notebook's Python cells and ZIP hashes are checked locally. An additional CPU check uses the actual Molmo2 video processor from revision `042abfa7a38879a376cec03d949eff0aefaa0600`: 64 synthetic RGB frames of shape 240×426×3 produce bitwise-identical pixel values, pooling indices and grids, with identical irregular timestamps, through both stacked and disk-mapped inputs. It uses CPU Torch 2.8.0 and local NumPy 1.26.4; it does not validate real video decoding, the pinned Kaggle NumPy environment, GPU inference or peak RAM on Kaggle. **No real Kaggle/T4 inference has been executed by the assistant.** Pilot results must be checked before interpreting accuracy.

Primary references checked 22/10/2026:

- [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks)
- [Kaggle Dataset documentation](https://www.kaggle.com/docs/datasets)
- [Kaggle CLI documentation, including Secrets usage](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
- [NVIDIA Kaggle GPU notebook setup](https://docs.nvidia.com/datascience/deployment/26.10/platforms/kaggle/)
- [Accelerate device-map design](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference)
- [Molmo2-4B model card](https://huggingface.co/allenai/Molmo2-4B)
- [uv Python installation](https://docs.astral.sh/uv/guides/install-python/)
