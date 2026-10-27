# Event coverage v1 — full LongVideoBench validation

Prepared 2026-10-27. This is a new run on all 1,337 questions / 753 videos. The completed 1,000-record dev200 run remains read-only. No full1337 GPU result is available yet.

## Fixed protocol

- Same Molmo2-4B revision, BF16, answer prompt, BLIP revision/preprocessing, candidate grid and question-only Qwen weights as the frozen baseline.
- Five variants: `sigma2`, `sigma1`, `time16_sigma1`, `eventmean_sigma1`, `eventcover_sigma1`. Target: **6,685 records**, 1,337 complete questions. The record count is not the number of independent answer forwards.
- Reuse the 200 frozen question score caches and matching event score caches from the completed development runs. Do not copy any old answer into the new run. Each answer is inferred in this run or reused from an identical ordered-frame result for the same question within this run.
- 361 questions have fewer than 64 candidate frames. For a grid of at most 64 candidates, every variant uses all candidates, just as the original baseline did; there is no padding, fabricated relevance vector or unnecessary BLIP scoring. The observed full set has no exactly-64 grid.
- The other 976 questions use the unchanged event helper and 64 unique chronological frames. Of these, 200 have reusable question scores; 776 need fresh question-only BLIP scoring. Eligible new event descriptions also need scoring. Full-set query counts and latency have not been measured.
- Same event parser, support thresholds and coverage allocation as dev200. No tuning based on the full labels. Labels are used for reporting and control validation, not frame selection or query generation.
- Recomputed sigma2 frames must match the old full baseline before answer inference. Same-frame predictions must match the baseline. On the 200 development questions, all five variants must reproduce their frozen frames and predictions. A mismatch stops the run for investigation; do not loosen checks or overwrite the journal.

## Independent output and resume

Output:

`/media/lnthanh03/DatHa/runs/videoqa/lvb-ridge-event-coverage-full1337-v1`

Read-only inputs:

- `lvb-val-molmo2-ridge64-full-ridge-v1` — completed full sigma2 baseline and Qwen weights.
- `lvb-ridge-smoothing-v1-dev200` — completed question score caches and sigma1/sigma2 development controls.
- `lvb-ridge-event-coverage-v1-dev200` — completed event score caches and five development references.

The full run has a separate contract, SQLite journal, curves and score records. The contract pins worker/helper hashes, inputs, runtime, precision, manifest and the small-grid policy. Changed pinned code requires a new run version. Source cache timers are not charged as new scoring; fresh question and event scoring have separate timers.

Each variant commits separately. Interrupted scoring of a question that has no completed score cache is redone. Completed score caches and committed variants are reused on restart. An active RIDGE job blocks another submission; a completed full run does not submit again. Runtime, models and active video remain in tmpfs; persistent storage holds only code, metadata, small score caches and results.

## Submit on the school login node

```bash
cd /media/lnthanh03/DatHa/code/VideoQA-EvidenceLab-ridge-v1
git pull --ff-only https://github.com/tdat1465/VideoQA-EvidenceLab.git codex/ridge-molmo2
bash scripts/slurm/run_ridge_event_full.sh full1337 --node any
```

One job uses one A100 on gpu01/02/03, eight CPUs and 90 GiB RAM; gpu04 is excluded. The total requested wall time is 48 hours including setup. This does not change the school's GPU idle policy and there is no artificial keepalive. If the job is interrupted, run the **same command** to resume. No full-run ETA is promised.

The launcher performs standard-library validation and `sbatch --test-only` before submitting one job. It asks for a hidden Hugging Face read token if `HF_TOKEN` is absent. SSH authentication and submission are performed by the user.

To validate placement without submitting a job:

```bash
python3 scripts/slurm/submit_ridge_event_full.py full1337 --node any --test-only
```

Monitor:

```bash
squeue -u lnthanh03 -n ridge-evtfull -o '%.18i %.9T %.30R'
```

Logs: `logs/ridge-evtfull-JOBID.out` and `logs/ridge-evtfull-JOBID.err`. Submission metadata: `logs/ridge-event-full-JOBID.json`.

After worker exit, check the summary:

```bash
python3 - <<'PY'
import json
from pathlib import Path
r = Path('/media/lnthanh03/DatHa/runs/videoqa/lvb-ridge-event-coverage-full1337-v1')
s = json.loads((r/'summary.json').read_text())
for key in ('completed_variant_answers', 'target_variant_answers', 'completed_questions',
            'complete', 'variants', 'control_reproduction', 'development_reproduction',
            'subsets', 'question_score_origins', 'actual_answer_forwards'):
    print(key, s[key])
PY
```

Completion: **6,685/6,685 records**, **1,337/1,337 questions**, `complete=true`.

## Download to drive E from Windows PowerShell

Run after the worker exits:

```powershell
powershell -ExecutionPolicy Bypass -File "E:\Codex\Documents\ChatGPT\KLTN\research-ridge\VideoQA-EvidenceLab\scripts\download_ridge_event_full.ps1" -Destination "E:\Codex\Documents\ChatGPT\KLTN\outputs\ridge-event-full1337-results-v1"
```

Keep the same destination to resume missing/changed files. One SSH inventory plus one SFTP transfer connection is used, normally two password prompts. SHA256 checks validate exports. The downloader does not export the SQLite journal or change server results.

## Evaluation and limits

Compare event coverage principally against sigma1, then sigma2, Time16 and event mean; include the old Uniform baseline on identical IDs. Report all 1,337 questions and the **1,137 questions / 672 videos outside the dev200 video set** separately. The summary's `unseen_videos` means disjoint from the 81 dev200 videos, not a claim that no prior full-baseline/error statistics have been viewed. The all-validation score includes development questions and is not an independent test score.

Use paired video-cluster uncertainty, eligible/fallback subgroups and both gains and losses. Report fresh question scoring, fresh event scoring and answer reuse separately. Full1337 is an evaluation expansion, not a new method or proof of improvement. The dev200 result was 128/200 versus sigma1 127/200, with a video-cluster CI containing zero.

Offline validation before submission: standard-library preflight, exact replay of 1,000 dev200 frame sets and 1,805 all-candidate frame sets (361 questions × five variants), CPU/mock tests for transactional resume and control mismatch, read-only journal progress, one-GPU submission guards, Bash syntax, and an interrupted SFTP fixture. These checks do not replace GPU inference or pixel inspection.
