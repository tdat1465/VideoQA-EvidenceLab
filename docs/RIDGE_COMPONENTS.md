# RIDGE component ablation after completed smoothing dev200

The previous run completed 200 questions with five variants. Sigma2 scored 122/200 and multiscale_v1 scored 120/200. This experiment separates the two mechanisms in that prototype. It is a diagnostic development ablation; neither component is a validated improvement.

## One job for two components

Run on the school login terminal:

```bash
cd /media/lnthanh03/DatHa/code/VideoQA-EvidenceLab-ridge-v1
git pull --ff-only https://github.com/tdat1465/VideoQA-EvidenceLab.git codex/ridge-molmo2
bash scripts/slurm/run_ridge_components.sh dev200 --node any
```

Enter the Hugging Face read token in the terminal if prompted. It remains hidden. Submit **one job, one GPU**, selected from gpu01/02/03, with 8 CPUs, 90 GiB RAM and 48 hours. The job name is `ridge-cmp`.

The run produces **200 questions × 2 variants = 400 new answers**. Both variants are evaluated serially in the same allocation. The old five variants are read-only references, not rerun. No extra pilot job is required; optional `pilot` mode stops after two complete questions and resumes with `dev200` in the same output directory.

- `vote_only`: majority region labels across sigma1/2/4, fallback to sigma2; sigma2 ranking and slope values; no fine peak rescue.
- `rescue_only`: sigma2 region labels, then the exact frozen v1 supported fine peak rescue rule and ranking replacement; no majority-vote changes. The sigma1/4 analyses are still used to identify fine-only peaks under the same rule as v1.

Both use the existing author allocation and selection helpers, frozen preference weights and 64-frame budget. Thresholds are unchanged. With no rescue, `rescue_only` reproduces sigma2 selection; this was checked on all 97 such questions in the actual source cache. Those matching frame sets provide a useful prediction consistency check when the GPU job is run.

## Immutable inputs and isolated output

Required existing server directories:

```text
/media/lnthanh03/DatHa/runs/videoqa/lvb-val-molmo2-ridge64-full-ridge-v1
/media/lnthanh03/DatHa/runs/videoqa/lvb-ridge-smoothing-v1-dev200
```

The second directory must retain `contract.json`, `manifest.json`, `summary.json`, `predictions.jsonl` and all **200 files under `scores/`**, with `complete=true` and 1,000 prior answers. Do not move or edit these source files. The preflight verifies completeness, hashes, baseline parity, video/candidate metadata, shared score identities and the frozen manifest. It fails if a score is missing or corrupt; there is no fallback rescoring.

New output:

```text
/media/lnthanh03/DatHa/runs/videoqa/lvb-ridge-components-v1-dev200
```

Only Molmo2 is loaded for new answering. **No BLIP or Qwen model is loaded or invoked.** Video files must still be fetched and decoded to answer from the newly selected frames. Large libraries, model weights, archives and active video bytes stay in allocation tmpfs, as before. Small journals, curves and exports persist in the new output. The runner and source inventory are part of its contract; keep them frozen while resuming.

## Follow progress and resume

```bash
squeue -u lnthanh03 -n ridge-cmp
tail -n 20 logs/ridge-cmp-JOBID.out
```

Replace `JOBID` with the ID printed by the launcher. If the worker stops cleanly, run the same `dev200 --node any` command after the old job leaves the queue. Committed answers remain; only unfinished variants run. Do not submit a duplicate. A dry scheduler validation is available through:

```bash
python3 scripts/slurm/submit_ridge_components.py dev200 --node any --test-only
```

When finished, summary must show `complete=true`, `completed_variant_answers=400`, and `n=200` for each variant. `one_time_scored_questions` and `one_time_scoring_seconds` are both zero because BLIP happened in the source experiment. `available_score_questions=200`; `reused_score_questions` counts source questions represented in the new answers. The timing scope excludes historical BLIP/Qwen and reports setup/transfer separately. Do not call cached per-answer timing an end-to-end speedup.

`reproduction_on_identical_sigma2_frames` reports how many new results use exactly the old sigma2 frames and how many reproduce its prediction. This is not a new standalone sigma2 control run. No GPU result for either component has been obtained before these instructions are delivered.

## Download after worker exit

In local Windows PowerShell:

```powershell
cd 'E:\Codex\Documents\ChatGPT\KLTN'
powershell -ExecutionPolicy Bypass -File '.\research-ridge\VideoQA-EvidenceLab\scripts\download_ridge_components.ps1'
```

The downloader retrieves new exports, logs and curves. It does not duplicate the source score cache. Send the resulting `outputs\ridge-components-results-<timestamp>` path for checking and comparison with all five previous variants on the same IDs.

## Local validation before GPU use

Four CPU tests cover all 200 actual cached curves, exact component isolation and the 97 no-rescue controls, a supported synthetic shoulder, corrupt/missing-cache rejection, and mocked interrupted/resumed inference. The mocked workflow asserts that the BLIP constructor is never called, source and baseline files remain byte-identical, committed answers survive and no new score directory is created. Existing frozen smoothing and scheduler tests also remain applicable. These checks establish implementation behavior, not component accuracy.
