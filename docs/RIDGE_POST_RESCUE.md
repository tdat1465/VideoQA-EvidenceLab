# Bounded post-selection rescue: frozen pilot20 and dev200

This is an exploratory new sampler, not a validated improvement or a novelty claim. It addresses the observed global changes caused by relabeling before RIDGE selection. Original smoothing/component scripts, baseline, preferences and BLIP caches remain unchanged and read-only.

The sampler first reproduces sigma2's 64 selected candidates. It proposes only frozen fine-peak rescue cores not covered by that initial set. A maximum of one or two swaps is allowed. A donor must be an original selected frame within 12 candidate steps, have another retained neighbor within two candidate steps, and occupy the same one of 16 fixed candidate-index bins as the proposed frame. Original first/last selected frames and previously covered rescue cores are protected. No global labels, allocations or temporal bins are rebuilt. If no eligible donor exists, the selection remains unchanged. These are heuristics, not proof that the removed frame is semantically redundant.

| Variant | Behavior |
|---|---|
| `sigma2` | Recomputed same-session control; original frame set and predicted choice must match the frozen baseline |
| `post_rescue1` | At most one local swap after sigma2 selection |
| `post_rescue2` | At most two; the first accepted swap is shared with post_rescue1 |

The frozen cohort remains the original 200 development questions. `pilot20` selects 20 questions by whole-video groups with seed 20261022, using only IDs, video names and candidate counts. It does not select the seven outcome-changing cases. It happens to include a previously inspected case because that video is sampled; it is not held-out data. CPU preview shows 2/20 pilot questions change one frame; the one- and two-swap variants have the same selections in this pilot. This pilot checks execution/reproducibility, not efficacy or the two-swap GPU case. Across all 200 questions, 38 have swaps; post_rescue2 performs two swaps in 15. No accuracy was used to tune this rule or seed. Earlier previews of larger metadata-only cohorts are not accuracy experiments; the protocol stays at the first frozen 20-question cohort.

## Run on the school server

```bash
cd /media/lnthanh03/DatHa/code/VideoQA-EvidenceLab-ridge-v1
git pull --ff-only https://github.com/tdat1465/VideoQA-EvidenceLab.git codex/ridge-molmo2
python3 scripts/slurm/submit_ridge_post_rescue.py pilot20 --node any --test-only
bash scripts/slurm/run_ridge_post_rescue.sh pilot20 --node any
squeue -u lnthanh03 -n ridge-post
```

The test-only command validates the frozen cache and asks Slurm to validate placement; it does not allocate a GPU. The launcher submits exactly one job using one GPU, 8 CPUs and 90 GiB RAM on gpu01/gpu02/gpu03, excluding gpu04. The default total wall-time limit is **48 hours including setup**, with a five-minute export/cleanup margin. The job exits when work completes; it does not retain the GPU for the full requested duration. The GPU idle policy is independent of total wall time. It blocks a duplicate active RIDGE job. Input HF_TOKEN is requested hidden in the server terminal when needed; do not paste it into logs or chat. Heavy libraries, model weights and active videos live only inside allocation tmpfs; Slurm's CUDA_VISIBLE_DEVICES is preserved.

New output: `/media/lnthanh03/DatHa/runs/videoqa/lvb-ridge-postrescue-v1-dev200`. Logs: `logs/ridge-post-JOBID.out`, `.err`, and `logs/ridge-postrescue-JOBID.json`. Existing 400 component results are untouched. One job runs all three variants serially. Sigma1/sigma4 controls are read from the compatible frozen source and reported on the same completed question IDs; they are not rerun.

## Expected pilot completion

Pilot creates **60 variant records**, each of the three variants having n=20. Its summary has:

```text
completed_variant_answers: 60
complete: false
pilot20.completed_questions: 20
pilot20.complete: true
```

The overall contract targets 600 records for dev200, so overall complete=false at pilot is intentional. A finished pilot exits zero. An interrupted session exits 75 and the same launcher resumes committed per-variant records. No second two-question smoke job is required. Model/environment setup still consumes time in every allocation; cached BLIP/Qwen scores are reused, not model files from previous allocations.

Before scaling, require 20 sigma2 controls to match baseline frames and choices, exact 64-frame outputs, valid swap limits and preserved bin/core coverage. Inspect `swap_counts`, `matched_reference_variants` and the paired predictions. A small accuracy increase on these 20 development questions is not a go/no-go significance test. If all checks pass and further evaluation is warranted, run:

```bash
bash scripts/slurm/run_ridge_post_rescue.sh dev200 --node any
```

This continues the same journal and fills missing variant records; immediately after pilot, 60 records are retained and 540 remain. Do not change the sampler, thresholds, cache or precision while resuming. Full completion is 600/600 and n=200 per variant. If the method is refined after viewing these results, use a separately versioned run and later evaluate on video-disjoint data not used to choose the method.

## Download in local PowerShell

```powershell
& 'E:\Codex\Documents\ChatGPT\KLTN\research-ridge\VideoQA-EvidenceLab\scripts\download_ridge_post_rescue.ps1'
```

The downloader displays both overall 60/600 and pilot20 20/20 completion at pilot. It retrieves logs, contract, summary, curves and predictions. Small question/option text is included in the new prediction export so future error analysis can use the actual question. Keep data exports private. No videos, weights or duplicate source score cache are downloaded.

## Local verification

Four sampler tests cover nested caps, same-bin swaps, no-donor fallback, label-blind whole-video sampling and interrupted transaction resume. Three submission tests cover read-only SQLite progress, one-GPU placement, the 48-hour default, custom hour/day formatting including valid 60-minute and longer jobs, invalid wall-time rejection, and duplicate blocking. All 200 real score caches are checked: 200 sigma2 selections reproduce baseline exactly; all 400 modified selections satisfy the cap, bin, anchor, uncovered-core and protected-coverage invariants. The source's original frozen validation also passes. Source trace and selection preview live in local `outputs/ridge-postrescue-v1/`.

## Resume after the school idle cancellation

The export for job 79497 contains 238/600 committed results, with 79 complete three-variant questions and one sigma2-only question. All three methods answer 59/79 of the complete common questions correctly. The original 60 pilot records are unchanged, all 238 selection replays match, and three two-swap GPU cases now exist. There are 362 answers left. The worker and experiment contract are unchanged by the scheduling update, so the existing journal remains resumable.

The school reports cancellation after 60 minutes without GPU activity. Previous exports only contain a GPU snapshot taken before model loading; it cannot establish a 60-minute idle interval. The temporary 55-minute wall-time limit introduced in ed250be was too restrictive and is removed. Launchers default to 48 hours total wall time and retain a five-minute cleanup margin. They submit one job at a time, without automatic resubmission, and exit when complete. Model downloads still occur in allocation tmpfs on each session; a longer job avoids repeating setup merely because of the former 55-minute limit. Run the dev200 command again only after an interrupted job ends, until 600/600. Use `--session-minutes 480` for an explicit eight-hour limit; the permitted range is 10-2880 minutes, independent of the 60-minute idle policy. Changes apply to newly submitted jobs; pulling code does not extend an already submitted 55-minute allocation.

Per-session files `gpu-util-JOBID.csv`, `gpu-context-JOBID.json`, and `phases-JOBID.log` provide read-only utilization samples, the CUDA GPU UUID/Slurm assignment, and phase timestamps. Samples cover the GPUs visible to nvidia-smi; correlate UUID before interpreting physical and logical indices. No dummy compute is used to maintain an allocation. The downloader retrieves these files when present, including for older runs without telemetry.

Implementation references: [Slurm sbatch](https://slurm.schedmd.com/sbatch.html) and [NVIDIA nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html). The school-specific idle policy is supplied by the user, not inferred from these documents.
