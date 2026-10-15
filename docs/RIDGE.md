# Author RIDGE on Molmo2-4B / LongVideoBench

This experiment integrates the unmodified author selector, not a paper-only
reimplementation. The answer model is Molmo2-4B and uses this repository's
video-only letter-logit protocol; this is not reproduction of the paper's
Qwen/InternVL/LLaVA accuracy numbers. No proposed improvements are enabled yet.

- Author repo: https://github.com/Xssq999/RIDGE
- Commit: `0f710c7304e806ccedc27b0d177241cb14d77bba`.
- `frame_select.py` SHA256: `7897d51979def23f83f4ce8e406b27b981f44ed6d8b1886eaff41241c1c83744`.
- `weight.py` SHA256: `0048dfc74e9a26febb14c97207bc3da98f04fb8b84202f9c40fcbe88128e744c`.

Files are fetched into compute RAM and verified before import. We do not bundle
the unlicensed upstream source or relicense it as our MIT code.

## Fixed adaptation

| Setting | Value |
|---|---|
| Answerer/data/scorer revisions | Same pinned Molmo2, LVB val and HF BLIP as existing configs |
| Final K | 64; if the candidate pool is shorter, all candidates, no padding |
| Candidate grid | Existing AKS floor-fps grid, approximately 1fps; subsecond guard |
| BLIP query | Question only; no choices, labels, category or duration |
| RIDGE core | Author `gradselect`, default sigma=2, adaptive_sigma=false; other defaults unchanged |
| Weight model | Qwen/Qwen3-8B at `b968826d9c46dd6066d109eabc6255188de91218` |
| Weight prompt | Original author SYSTEM_PROMPT and USER_TEMPLATE, question only |
| Generation | Native resolved BF16/FP16, do_sample=false, thinking=false, 512 new tokens |
| Answers | One Molmo2 call; chronological original frame IDs/timestamps |

Qwen loads first on the one Slurm-owned GPU, writes weights atomically per
question, then releases its model and dedicated RAM cache before Molmo2/BLIP
load. No scheduler rewrites CUDA_VISIBLE_DEVICES. No weights/video/env downloads
run on the login node or persistent disk. Per-run weights are small metadata.

Weight generation follows the author's prompt and generation recipe but uses
an immutable revision, per-question durable cache, and strict validation.
The upstream defaults/clamping for invalid JSON are intentionally rejected;
missing/noninteger/out-of-range fields stop the run. No silent uniform fallback.

`ridge-weights.json` binds prompts, source, runtime, resolved dtype, model revision,
annotation/subset hash and each question. Checksums bind result rows to exact
weights. The run lock covers cache writes. On interruption, valid prior weights
are reused; altered cache/contracts refuse resume. Keep the entire run directory.

Timing: per-question elapsed covers decode, BLIP, RIDGE and answer. Qwen model
setup, full weight-preparation phase, per-question generation time/tokens and
weight-stage VRAM are recorded separately in cache/traces/events. Include this
auxiliary cost when comparing efficiency; selection CPU time alone is not total
cost. BLIP parity with the author's scoring environment remains unverified.

## Slurm pilot and full

The Git workflow uses the `codex/ridge-molmo2` branch and a separate checkout:

```bash
cd /media/lnthanh03/DatHa/code
git clone --single-branch --branch codex/ridge-molmo2 https://github.com/tdat1465/VideoQA-EvidenceLab.git VideoQA-EvidenceLab-ridge-v1
cd VideoQA-EvidenceLab-ridge-v1
bash scripts/slurm/run_ridge_server.sh pilot
```

For later updates, use `git pull --ff-only origin codex/ridge-molmo2`. The
launcher checks branch, clean source, pinned fingerprint and active runs before
submission. Use `full` only after inspecting pilot. An offline bundle launcher
is also available; both use the same planning logic.

The workflow uses the run label `ridge-v1`, requests
one GPU, 8 CPUs, 90GiB RAM, and only gpu01-03. It submits RIDGE and Uniform as
independent jobs, checks sbatch test-only before submission, and refuses active
duplicate run directories or changed source/config.

Pilot commits at most two new questions per method under the full 1337-question
contract. Full resumes the SAME run; dev200 uses a separate directory/contract.
Exit 75 is a controlled partial stop and may appear as FAILED in Slurm; inspect
summary/logs. Only complete=true and completed_count=1337 establish full accuracy.

GPU inference has not been executed locally. CPU tests with fake models/HF/video
are wiring and resume tests, not real VideoQA results. The author example smoke
executes actual upstream CPU selection but no model or real video inference.
