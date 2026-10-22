# RIDGE cached ablation on Kaggle T4 x2

This is a **new FP16 experiment**, with Molmo2 weights sharded across both T4 GPUs using Accelerate's `balanced` mapping. It is not a continuation of the BF16 A100 school run. The 376 existing school answers remain in their original run; finish the remaining 24 on the school server if completing that experiment.

The provided importable notebook is `outputs/ridge-kaggle-v1/RIDGE_Kaggle_T4x2.ipynb` in the local workspace, accompanied by `ridge-kaggle-input.zip`. The ZIP contains approximately 6 MB of code, author selection source, baseline metadata, precomputed BLIP scores and cached Qwen preference weights. It contains no video bytes, answer model weights or credentials. Upload it to a **private** Kaggle Dataset and attach that Dataset to the notebook. No upload has been performed by the assistant.

## Notebook setup

1. Create a Kaggle notebook and import `RIDGE_Kaggle_T4x2.ipynb` using its notebook import/upload action.
2. Select **GPU T4 x2** and turn **Internet** on. Both GPUs must remain visible. Availability and quotas depend on the account; inspect the session panel rather than assuming a fixed weekly allowance.
3. Create a private Kaggle Dataset from `ridge-kaggle-input.zip` and use **Add Input** to attach it. The notebook accepts both the unchanged ZIP and files automatically expanded by Kaggle, finding the bundle manifest under `/kaggle/input`.
4. Add the existing Hugging Face read token to Kaggle **Secrets** under `HF_TOKEN`, then enable that secret for the notebook. Ensure this HF account has accepted LongVideoBench access conditions. Never paste a token into code or output.
5. Run the cells in order. The Python version must be 3.10–3.12. An isolated environment under `/kaggle/temp/ridge-v1/venv` installs Torch 2.8.0 with CUDA 12.6 and the existing pinned Molmo requirements. It does not replace the notebook kernel's packages.

The first cell prints GPU names and checks the input ZIP's hashes. The preflight validates all 200 source caches and historical baseline consistency. The backend then verifies two T4 GPUs and native FP16 CUDA operations. CPU/disk weight offload is not accepted by this profile; videos and model downloads use Kaggle's temporary disk, separately from saved results.

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
/kaggle/working/ridge-kaggle-v1/run/summary.json
/kaggle/working/ridge-kaggle-v1/run/predictions.jsonl
/kaggle/working/ridge-kaggle-v1/run/results.sqlite3
/kaggle/working/ridge-kaggle-v1/run/curves/
/kaggle/working/ridge-kaggle-v1/ridge-kaggle-checkpoint.zip
/kaggle/working/ridge-kaggle-v1/session.log
```

SQLite commits each variant independently. Human-readable exports update after each question and on worker exit; a transaction-consistent ZIP updates every five questions and on worker exit. The ZIP contains the answer journal, contract, curves, exports and archive metadata index when available. It excludes model weights, HF caches, active video files and tokens.

The worker defaults to six hours, then stops between variants/questions and exports. This is a worker budget, not a promise the entire experiment fits one session. Kaggle's session limit also includes environment setup and downloads. The notebook's Interrupt handler asks the worker to stop safely; a forced platform termination can prevent the latest export/ZIP even though SQLite has more committed rows.

Before ending a session, download the checkpoint ZIP and session log or save a notebook version with outputs. For a new session, attach the original input bundle and a private Dataset containing the latest `ridge-kaggle-checkpoint.zip`; the restore cell accepts the ZIP or its automatically expanded contents. Keep the notebook/code and environment frozen. Runtime or device-map changes cause contract mismatch rather than mixed results. Kaggle cannot recover unsaved working storage after it is discarded.

The first archive-index build may take time because videos are read by bounded HTTP Range from the official multipart archive. Only one active video is stored at a time. The metadata index is included in subsequent checkpoints so it can be reused. No full dataset archive is staged.

Completion is `completed_variant_answers=1000`, `complete=true`, and `n=200` for every variant. Compare the five variants on identical question IDs **within this FP16 run**. FP16-vs-BF16 or T4-vs-A100 comparisons are separate environment analyses, not selector-only ablations. Cached answer timings exclude prior BLIP/Qwen and do not represent end-to-end speedups.

## Validation and current limitation

The port has five CPU tests covering separate FP16 contracts, rejection of BF16 answer journals, multi-GPU mapping checks, interrupted per-variant transactions, ZIP restore/resume and rejection of nonfinite answers. The notebook's Python cells and ZIP hashes are checked locally. The pinned Molmo2 source was inspected for its supported split modules and image-feature device transfer. **No real Kaggle/T4 inference has been executed by the assistant.** Pilot results must be checked before interpreting accuracy.

Primary references checked 22/10/2026:

- [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks)
- [Kaggle Dataset documentation](https://www.kaggle.com/docs/datasets)
- [Kaggle CLI documentation, including Secrets usage](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
- [NVIDIA Kaggle GPU notebook setup](https://docs.nvidia.com/datascience/deployment/26.10/platforms/kaggle/)
- [Accelerate device-map design](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference)
- [Molmo2-4B model card](https://huggingface.co/allenai/Molmo2-4B)
