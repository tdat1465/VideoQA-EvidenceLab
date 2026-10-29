# Diagnose the o2F-N42Ufo4_2 control mismatch

Prepared 2026-10-29. Full1337 jobs 80271 and 80364 stopped at the sigma2 frame-reproduction guard for `longvideobench:val:o2F-N42Ufo4_2`. The current cache selects 60/64 of the old baseline frames. There are 5,665 completed records on 1,133 questions; resubmitting the full worker reloads the same failing cache. Keep that guard and cache unchanged.

## What is known and what is missing

The runtime/configuration, weights, video hash and pinned code match. Given the cached question scores, the original author selector and full worker produce the same new frames. Original baseline job 79154 processed questions 0, 1 and 2 consecutively on the same video reader. The full worker additionally performs event-query scoring and reads frames for multiple answer variants between those questions. Decoder state/history is therefore a hypothesis to test, not an established cause. Historical raw BLIP scores and pixel hashes were not included in the baseline export.

## One isolated GPU diagnostic

```bash
cd /media/lnthanh03/DatHa/code/VideoQA-EvidenceLab-ridge-v1
git pull --ff-only https://github.com/tdat1465/VideoQA-EvidenceLab.git codex/ridge-molmo2
bash scripts/slurm/run_ridge_event_drift.sh --node gpu03
```

Default: one GPU on gpu03, eight CPUs, 90 GiB RAM, four-hour maximum session including setup. gpu03 matches the node and A100 PCIe family used for the original baseline and failing first full job. This does not pin the same physical GPU UUID; the report records the actual UUID and CUDA flags. `--node any` is supported if necessary, but hardware differences must then be considered. An active RIDGE/full/diagnostic job blocks submission. There is no artificial GPU keepalive.

The launcher validates the stopped full snapshot and runs `sbatch --test-only` before submitting one job. A hidden HF read token is requested if needed. Models, libraries and the one video remain in tmpfs and are cleaned when the allocation ends.

This diagnostic writes only under:

`/media/lnthanh03/DatHa/runs/videoqa/lvb-ridge-event-drift-o2f-v1/session-JOBID`

It does not edit full-run contracts, scores, predictions or SQLite. It hashes the full contract/predictions and failing score before and after the diagnostic. A changed snapshot invalidates diagnostic completion. No result from this diagnostic is inserted into full evaluation.

## Four fixed cases

1. `cold_a`: new reader, score question 2 directly.
2. `cold_b`: another new reader, repeat question 2.
3. `baseline_history`: new reader; score question 0, decode its original baseline answer frames and run the answerer; repeat for question 1; then score question 2.
4. `full_history`: new reader; score each preceding question, replay its event-query forwards and the answer-frame accesses/forwards recorded as actually performed in the full run, then score question 2.

The history replay uses recorded frame IDs to reproduce decoder access history; it does not change the selector or claim these are new evaluation answers. Every actual candidate array and normalized BLIP image tensor for question 2 is hashed, together with score vectors and selected sigma2 frames. The model, tokenizer IDs, input question, candidate grid, weights and video SHA are recorded. Pixel/tensor hashing does not alter returned arrays or model tensors, but it adds CPU work; this is not a speed benchmark.

The cases share one loaded model session and reproduce histories within this one video. Earlier videos and global CUDA allocator history of a day-long job are not reproduced. Matching software versions alone does not establish bit-identical GPU computation. There is still no historical pixel/score fingerprint to compare directly with the original baseline.

Interpret report comparisons as observations: changed raw-pixel hashes establish that decoded arrays differed across tested histories; unchanged raw/tensor hashes with changed scores localize drift downstream of those recorded inputs. Neither observation automatically explains every historical run. If all cases agree, further evidence is needed.

## Monitor and download

```bash
squeue -u lnthanh03 -n ridge-evtdiag -o '%.18i %.9T %.30R'
```

Logs: `logs/ridge-evtdiag-JOBID.out` and `.err`. Each completed case is exported immediately; interrupted jobs may still have a partial report.

After the worker exits, download the single self-contained report from Windows PowerShell, replacing `80399` with the submitted job ID:

```powershell
powershell -ExecutionPolicy Bypass -File "E:\Codex\Documents\ChatGPT\KLTN\research-ridge\VideoQA-EvidenceLab\scripts\download_ridge_event_drift.ps1" -JobId 80399 -Destination "E:\Codex\Documents\ChatGPT\KLTN\outputs\ridge-event-drift-results-v1"
```

One SCP connection, normally one password prompt. The final local file is `drift-JOBID.json`. Check `complete=true` and `snapshot_unchanged=true`; if incomplete, retain the partial report and inspect its error type and available cases. The diagnostic does not automatically resume full1337 or modify its failed cache.
