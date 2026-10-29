# Canonical RGB v2: fix decoder-state dependence

Job 80381 completed all eight decoder-policy cases. Each policy repeated exactly
(RGB hashes, BLIP tensors, scores and selected indices). Sequential Decord with
one and two threads also agreed exactly. All sequential decoders returned 33,909
frames; PyAV presentation timestamps matched ordinal/30 with maximum error 0.

Indexed Decord differed from sequential Decord at 231/1,130 candidate RGB frames.
Its selection matched the old baseline 64/64, whereas both sequential libraries
matched only 24/64. Matching the old baseline alone therefore cannot validate a
decoder policy. PyAV and Decord RGB conversion differed at 1,113 candidates; this
does not establish that either library decoded the wrong picture. Evidence is
limited to this regression video. The previous 80368 history experiment showed
an actual input-pixel change under the old indexed access pattern.

## Protocol

The new run is `lvb-ridge-event-canonical-v2`, with its own SQLite journal.
Keep full1337-v1 and its 5,665 committed answers unchanged. Do not resume v1 by
deleting a score, changing the sigma2 guard or substituting a convenient curve.

For each video, v2 uses two fresh sequential Decord passes with one thread, starting
at zero and draining to EOF without seeking. Both frame counts and every required
native RGB hash must agree. Cache the union of the frozen 1 FPS grid and native
Uniform64 indices as lossless PNG, without resizing, in allocation tmpfs only.
Read back and verify every PNG. Both BLIP and Molmo2 read these same cached pixels;
no video-reader state is used during scoring or answering. Keep persistent frame
hash/shape manifests, but no PNG/video files. On resume, rebuild and require an
identical manifest before accepting a score cache or adding results.

The per-video PNG cache has a 32 GiB ceiling and checks actual tmpfs/cgroup free
space before writes. It stops on insufficient RAM; there is no persistent-disk
fallback. The allocation remains one GPU, eight CPUs, 90 GiB, gpu01/02/03 only.
Setup, dependencies, models and video/image caches use the existing tmpfs workflow.

Six variants: Uniform64, sigma2, sigma1, time16_sigma1, eventmean_sigma1 and
eventcover_sigma1. All BLIP scores and answers originate in v2. Frozen Qwen text
weights are reused because their input does not include pixels. Answers may be
reused for identical selected frames within this v2 journal only. Uniform64 uses
64 evenly spaced native source frames, including short candidate-grid videos;
RIDGE uses all candidates when its grid has <=64 frames.

Old baseline comparisons are descriptive metadata, with no equality assertion:
v2 changed the input pixels and needs fresh controls. Event parsing and selection
remain frozen. No method uses labels to select frames.

## Run on the school server

```bash
cd /media/lnthanh03/DatHa/code/VideoQA-EvidenceLab-ridge-v1
git pull --ff-only github codex/ridge-molmo2
bash scripts/slurm/run_ridge_event_canonical.sh pilot23 --node gpu03 --session-minutes 360
```

Pilot23 is the fixed original pilot20 plus all three questions from regression
video `o2F-N42Ufo4.mp4`: 23 questions across 10 videos, 138 records. It exercises
all six methods, sequential pixel repeatability and shared-cache answering. A
successful pilot is `pilot23.complete=true`; the overall `complete` remains false
until all 8,022 records exist. Inspect/download the pilot before starting full.

Resume an interrupted pilot with the exact same command. After pilot completion:

```bash
bash scripts/slurm/run_ridge_event_canonical.sh full1337 --node any
```

Full is gated on all 138 pilot records. It continues the same v2 journal, so pilot
answers are retained. Resume full with this same command. No old v1 answers are
copied. A clean session stop can return 75 and still preserve committed records.
Do not run multiple RIDGE experiments simultaneously. No GPU keepalive is used.

Download from Windows PowerShell after the worker exits:

```powershell
Set-Location 'E:\Codex\Documents\ChatGPT\KLTN\research-ridge\VideoQA-EvidenceLab'
powershell -ExecutionPolicy Bypass -File .\scripts\download_ridge_event_canonical.ps1 -Destination 'E:\Codex\Documents\ChatGPT\KLTN\outputs\ridge-event-canonical-results-v2'
```

The downloader verifies SHA256, retains complete local files on interruption,
and downloads pixel manifests along with scores, curves and exported results.
Use the same destination to continue a download. Expected pilot 138/8022 and
23/23 pilot questions; expected full 8022/8022 and 1337/1337 questions.

## Reporting and validation

Report all validation plus engineering videos (dev200's 81 videos plus the decoder
regression video) and the remaining videos separately. The latter is not a pristine
test set: old full baseline/partial results have already been inspected. A single
video diagnostic does not establish whole-dataset decoder equivalence or gains.

Local validation: standard-library preflight against the downloaded frozen inputs;
eight CPU tests covering two-pass EOF/RGB, PNG round trips/corruption, cancellation,
cache budget, native Uniform64, pilot gating, same-run resume/answer reuse, and Slurm submission gating;
mock downloader interruption/checksum/path tests; Bash syntax checks. Real Decord,
BLIP and Molmo2 integration is the server pilot, not a locally completed GPU run.

Diagnostic report SHA256:
`d56fac26dbe23d894b3e39ef4fd05af6249c1e3434774f58eb1abffce145890e`.
