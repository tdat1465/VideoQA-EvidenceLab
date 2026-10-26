# RIDGE: event decomposition and evidence coverage v1

Development prototype prepared 2026-10-26. The first GPU pilot stopped at 92 records: 18 questions have all five variants, and a nineteenth has both controls. Both controls reproduced frames and predictions on 19/19 available questions. The user accepted this reviewed partial pilot to continue dev200; this is not evidence of an accuracy improvement. The earlier question/options experiment is complete and stays read-only. This is a new run, not a continuation of its 400 records.

## Purpose and controls

The previous mean-of-options score changed many frames but achieved 123/200 versus sigma2 122/200 and sigma1 127/200. This experiment asks whether separate event evidence helps more than averaging queries, and whether any improvement is explained by time coverage alone.

| Variant | Selection | Purpose |
|---|---|---|
| `sigma2` | Original question scores, original weights, sigma=2 | Reproduce the full baseline |
| `sigma1` | Original question scores, original weights, sigma=1 | Reproduce the strongest previous dev setting |
| `time16_sigma1` | 16 uniform candidate anchors, fill from sigma1 | Isolate time coverage |
| `eventmean_sigma1` | Average supported event scores, blend 50/50 with question scores, run sigma1 | Isolate separate event queries followed by averaging |
| `eventcover_sigma1` | Same 16 time anchors, up to 16 event/context additions, fill from sigma1 | Test separate event coverage |

The three experimental variants all fall back to sigma1 if public-text parsing fails or no event passes support checks. The time control uses exactly the same support gate as event coverage. In the local text preview, 16/200 questions qualify for parsing; 184/200 fall back. Selection never uses dataset type, answer labels, earlier prediction outcomes or question ID. Report effects both on the full fixed dev200 and on the text-eligible subset; a subset result is not a full-dataset improvement.

## Public-text queries

The question must explicitly mention a sequence/order of scenes/events/scenarios. Every choice must start with `First` and contain explicit punctuation-delimited markers such as `Next`, `Then`, `followed by`, `Finally` or `lastly`, with an unambiguous final clause. Unsupported or over-budget syntax fails closed.

Extract clauses from **all choices**, remove the ordering markers, normalize punctuation/case, deduplicate exact normalized descriptions and sort the resulting union. Choice order and repeated wording do not weight a description more heavily. No option order is used as a guessed answer. Example: five permutations of three exactly matching events yield three queries, not fifteen. This is exact text deduplication, not semantic coreference/paraphrase resolution. Different descriptions of one scene can remain distinct; references such as “the same woman” can remain ambiguous. The union is capped at 12 descriptions; it can contain distractor events that do not exist in the video.

Each description keeps all tokenizer tokens in chunks of at most 33 content tokens plus 2 special tokens. Use the immutable BLIP helper from the previous run, average chunks within each event, and retain the separate score vector for every event. A small transformed frame batch is reused across text queries. Each BLIP forward still re-encodes vision, so this does not claim an image-embedding speed optimization. Old Qwen weights and old question-only scores are reused; there are no new Qwen calls.

## Support and allocation

An event is supported only if its raw BLIP maximum is at least 0.5, its raw range is at least 0.1, and at least two samples in the maximum's radius-1 neighborhood have min/max-normalized score at least 0.5. These thresholds are frozen **heuristics**, not calibrated proof that an event exists. Flat curves, low scores and isolated one-sample spikes fail these checks.

For each supported event, Gaussian-smooth its raw curve with sigma=1, min/max-normalize it and map responses below 0.5 to zero. These normalized curves are retrieval responses, not probabilities or confidence estimates. With `S` the current frame set, use `mean_e max_{i in S} response[e,i]` as the greedy objective.

For `eventcover_sigma1`:

1. Keep 16 evenly spaced candidate indices, including both ends.
2. Greedily add the frame with the largest remaining event coverage gain. Add at most `min(8, supported event count)` centers; centers must differ by at least two candidate indices. Stop if gain is zero. This does not force one center per description.
3. Add neighboring candidates at offsets -1 and +1 around centers, with at most 16 event/context additions beyond the time anchors.
4. Fill to exactly 64 unique frames from the original sigma1 set, prioritizing its smoothed question relevance and resolving ties by index. At least 32 original sigma1 frames remain. Sort frames chronologically before passing them to Molmo2.

`time16_sigma1` applies the same fill with the 16 anchors and no event additions. `eventmean_sigma1` uses the supported events' raw scores with equal weight per event, mixed 50/50 with old question scores, then the original sigma1 selector. It does not add time anchors. Mean/coverage comparisons therefore concern complete allocation designs; the time control specifically tests whether event additions help beyond time anchors.

All variants use the old candidate grid, old question-based Qwen preferences, unchanged answer prompt and 64-frame budget. Neighbor offsets are candidate steps (approximately seconds at this grid), not adjacent raw video frames. This prototype cannot retrieve evidence between existing candidates or fix a missing fine grid.

## Integrity, resume and cost

- New run: `/media/lnthanh03/DatHa/runs/videoqa/lvb-ridge-event-coverage-v1-dev200`.
- Immutable score source: `lvb-ridge-smoothing-v1-dev200`; full baseline: `lvb-val-molmo2-ridge64-full-ridge-v1`.
- Same fixed dev200 and pilot20 manifests as previous runs. Pilot has 20 questions/100 variant records, including 5 text-eligible questions. Dev200 has 200 questions/1000 records, including 16 text-eligible questions.
- Offline token preview: 15 event queries/chunks and 15,795 frame-text pairs for pilot20; 69 queries/73 chunks and 90,507 pairs for dev200. These counts are not measured latency or accuracy.
- Contract pins worker, event helper, frozen scoring helper, source/config/runtime/precision, manifests, old weights and score hashes. Changes to pinned code require a new run version; never delete the old journal to mix settings. Submission-policy changes may resume the same run when all pinned contract hashes still match.
- SQLite commits every variant. Completed event scores are reused after restart. Interrupted scoring of the active question is redone. The active video/model/runtime are held in tmpfs; only small results/cache metadata persist.
- Exact ordered-frame matches **within this new run and question** reuse that run's deterministic answer. Such rows record `reused_answer_from`, `answer_inference_performed=false` and zero answer-forward seconds. They are valid variant outcomes, not independent model calls. Original-run answers are used only to check control reproduction, never copied as new answers.
- Controls stop the run if sigma1/sigma2 frames or predictions differ from frozen references. Same-frame experimental predictions must also match those references.
- A single job requests one GPU, 8 CPUs and 90 GiB RAM, chooses gpu01/02/03 and excludes gpu04. Default total wall time is 48 hours including setup, independent of the school's 60-minute idle policy. There is no artificial GPU keepalive.

## Run on the school server

```bash
cd /media/lnthanh03/DatHa/code/VideoQA-EvidenceLab-ridge-v1
git pull --ff-only https://github.com/tdat1465/VideoQA-EvidenceLab.git codex/ridge-molmo2
bash scripts/slurm/run_ridge_event_coverage.sh pilot20 --node any
```

If `HF_TOKEN` is absent, the launcher asks for the Hugging Face read token with hidden input. Run the wrapper on the login node; model work runs only inside the allocated GPU job. The launcher validates the frozen data and uses `sbatch --test-only` before submitting **one** job.

Monitor:

```bash
squeue -u lnthanh03 -n ridge-event -o '%.18i %.9T %.30R'
```

Use the submitted job ID to read `logs/ridge-event-JOBID.out` and `logs/ridge-event-JOBID.err`. Re-run the **same command** to resume a stopped pilot; an active job is blocked, and a completed mode does not submit again.

After the worker exits, inspect:

```bash
python3 - <<'PY'
import json
from pathlib import Path
r = Path('/media/lnthanh03/DatHa/runs/videoqa/lvb-ridge-event-coverage-v1-dev200')
s = json.loads((r/'summary.json').read_text())
for k in ('completed_variant_answers', 'pilot20', 'control_reproduction',
          'parser_eligible_questions', 'supported_event_questions',
          'eventcover_same_frames_as_sigma1', 'actual_answer_forwards', 'variants'):
    print(k, s[k])
PY
```

Pilot completion should read 100/1000 records and `pilot20.complete=true`; global `complete=false` is expected. Before dev200, review control matches, parsed descriptions, accepted events, actual frame changes and runtime. A pilot accuracy increase does not establish a gain. If every eligible event is rejected or coverage never changes frames, investigate before spending on dev200; do not tune thresholds using pilot correct answers and overwrite this run.

After a technically valid pilot, continue the same run:

```bash
bash scripts/slurm/run_ridge_event_coverage.sh dev200 --node any
```

By default, the launcher requires the pilot to be complete. An explicitly reviewed partial pilot with at least 18 complete questions can instead be accepted with:

```bash
bash scripts/slurm/run_ridge_event_coverage.sh dev200 --node any --accept-partial-pilot
```

This flag changes only the submission prerequisite. It records the pilot completion count and acceptance in the job metadata, retains every source/contract/control check, and resumes the existing SQLite journal. The full target is 1000/1000, including the missing pilot records. From the audited 92-record snapshot, 908 records remain; this is not the number of independent answer forwards. Repeating the same dev200 command resumes committed results without restarting completed variants. Worker, selector, thresholds, manifest, model, and frame budget are unchanged.

Download from **Windows PowerShell**, after worker exit:

```powershell
powershell -ExecutionPolicy Bypass -File "E:\Codex\Documents\ChatGPT\KLTN\research-ridge\VideoQA-EvidenceLab\scripts\download_ridge_event_coverage.ps1" -Destination "E:\Codex\Documents\ChatGPT\KLTN\outputs\ridge-event-coverage-results-v1"
```

Repeat with the same destination to fetch only missing/changed files. One SSH inventory connection plus one SFTP transfer connection is used, normally two password prompts. SHA256 checks validate every export. Counts refresh when the worker exits; the downloader does not export the SQLite journal.

## Evaluation

Compare paired gains/losses against sigma1 **and** sigma2, then time16 and eventmean. Also compare the existing Uniform baseline on the same fixed IDs. Report changes on eligible and fallback questions, group uncertainty by video, and charge new event scoring as well as answer time. Event response coverage is a selector diagnostic, not annotation of true evidence coverage. Analyze gained and lost questions with video evidence before assigning causes. This dev200 has already informed design; any final improvement claim needs video-disjoint evaluation after freezing the method.
