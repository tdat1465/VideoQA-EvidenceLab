# FORTE on Molmo2-4B / LongVideoBench

Independent implementation of arXiv:2610.00573v1 (2026-10-01), Appendices A-C.
This is an adaptation to the existing EvidenceLab protocol, not author code or
a reproduction of the paper's Qwen3-VL-8B accuracy numbers. No training is used.

## Fixed experiment

| Item | Value |
| --- | --- |
| Backbone | Frozen Molmo2-4B, revision 042abfa7a38879a376cec03d949eff0aefaa0600 |
| Data | LongVideoBench val, revision 60d1c89c1919a198b73be39c2babb213b29d6a5c |
| Inputs | Video frames + question + choices, no audio/subtitle/transcript |
| Scorer | Existing pinned HF BLIP ITM large COCO, FP32; question only |
| Scoring allowance rho | 0.5 frame-query evaluations per second |
| Final frames K | 64, or all original frames if video has fewer than 64 |
| GP | RBF length scale 10 s, FP64, half-width 128, recovery cap 1024 |
| Acquisition / selection weights | 3.5 / 1.0 |
| Initialization / adaptive rounds | Reference grid at rho/2, four rounds |
| Answer evaluation | Existing letter-token logits, lvb-video-only-letter-v1 |
| Question subset | 200 fixed questions, seed 18; full config has 1337 questions |

The measured scorer budget B is the count of the unique reference grid, not K.
The frame universe is every original frame. T=(n-1)/fps. A stride
max(1,floor(fps/rho)) gives max(1,floor(n/stride)) reference points. Initialization
uses rho/2. Remaining evaluations are divided evenly over the four rounds.
Each GP fit normalizes all measured scores again; constant scores become zero.
Predictions are clipped to the normalized observation range. No variance/UCB is used.
GP means remain fixed inside a round; coverage changes after each acquisition.
Final exact optimization uses measured scores only, over the scored pool.
It does not optimize over unmeasured frames. If B<K, preserve the scored selection
and supplement with farthest original frames, earlier on ties, without scoring them.

The GP uses banded Cholesky and solves, without a dense inverse. At each width,
jitter increases from 1e-12 to 1e-3 in powers of ten. On failure the width doubles
up to min(1024,n_observations-1), restarting jitter. Failure at the cap raises an
error. Prediction temporaries are limited to 1024 query frames at a time.
The optimizer uses rolling DP rows, a monotone deque and predecessor indices,
with O(KB) time and O(KB) backtracking storage. It never forms a B-by-B gap matrix.

## Reconstruction choices and protocol differences

- Paper v1 states code will be released at publication; no official repository
  was identified during preparation. The algorithm is reconstructed from the PDF.
- The reference grid's integer mapping is not specified completely in the PDF.
  This implementation uses EvidenceLab's endpoint-spanning `uniform`, Python
  round, midpoint for one point. This choice is recorded in the contract.
- A GP neighborhood is insertion position (clamped to the last observation)
  plus/minus the recovered width, clipped to valid observation indices. The PDF
  does not specify exact edge indexing. Binary insertion lookup is used rather
  than the paper's chronological pointer sweep.
- Acquisition uses a heap of gap maxima. Only the split gap is recomputed. This
  gives the same greedy decisions as recomputing every candidate after each
  acquisition, with an earlier-source-index tie break. It is not the author's
  position-aware candidate heap implementation.
- The paper's default scorer is LAVIS BLIP; this run retains the existing HF
  port, special-token fix, question cleaning and 35-token truncation. Numerical
  equivalence with LAVIS has not been established.
- The paper generates and parses answers. This run preserves Molmo2 letter
  logits so comparisons within EvidenceLab keep the same answer protocol.
- Historical AKS sometimes supplies fewer than 64 frames; FORTE follows the
  paper's preserve-and-supplement rule. Report actual frame counts when comparing.

## Deployment and experimental sequence

Use the packaged `outputs/deploy-forte.ps1` from the KLTN workspace in PowerShell.
It uploads a Git bundle and launcher, then asks for SSH/HF credentials in your
terminal. The new server checkout is
`/media/lnthanh03/DatHa/code/VideoQA-EvidenceLab-forte-v1`.
The old lens-textfix checkout and completed exports are not used for resume here.

1. Default `pilot`: two questions each for FORTE and a fresh Uniform control;
   90 GiB, 1 GPU, 8 CPUs, max 48 h, allowed nodes gpu01-gpu03. Pilot exits 75
   after its two committed questions because the 200-question journal is partial.
2. Inspect logs, source indices, counts, precision, memory and stage timings.
   `python3 scripts/audit_forte_run.py RUN_DIR` audits FORTE exports with stdlib.
3. `-Mode dev200`: resume the SAME 200-question runs, unlimited new questions
   within each allocation. This is a development subset, not untouched evaluation.
4. `-Mode full`: independent full-validation FORTE and Uniform runs; resume if
   partial, skip if complete. Use all 1337 questions and video-cluster bootstrap.

Run directories:

```
/media/lnthanh03/DatHa/runs/videoqa/lvb-val-molmo2-forte64-200-forte-r05-v1
/media/lnthanh03/DatHa/runs/videoqa/lvb-val-molmo2-uniform64-200-forte-r05-v1
/media/lnthanh03/DatHa/runs/videoqa/lvb-val-molmo2-forte64-full-forte-r05-v1
/media/lnthanh03/DatHa/runs/videoqa/lvb-val-molmo2-uniform64-full-forte-r05-v1
```

Changing source/config requires a new checkout/label; do not edit or bypass stored
contracts. The old full runs have a different source fingerprint. Formal strict
paired comparison therefore uses the fresh Uniform control from this checkout.
Existing FOCUS/AKS results remain historical references unless rerun here.

All large data, weights, runtime and cache remain in compute-node tmpfs, one video
at a time; persistent storage holds code, metadata and question results only.
No model/GPU run has been executed locally. CPU checks do not establish GPU
compatibility, benchmark accuracy, or a speed improvement.

## Recorded evidence

Each prediction records original frame count/FPS, B distinct scorer evaluations, initial/scored frame IDs,
raw measured scores, selected original IDs/timestamps, per-round acquired IDs,
GP width/jitter/recovery attempts and fit/predict times, acquisition time, final
optimization time, and supplemental-frame count. Existing total selection timing
is synchronized before measurement. BLIP decode/inference stays in batches of 4.

`tests/test_forte.py` checks the optimizer against 1680 exhaustive small cases and
independent quadratic DP; GP against dense/local references; greedy acquisition
against full recomputation; numerical recovery, question-only scoring, exact
budget use, short-video supplementation, controlled stop, and runner resume.

Paper: https://arxiv.org/abs/2610.00573v1
