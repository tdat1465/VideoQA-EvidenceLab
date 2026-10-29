# Validate decoding policies on the drift video

This is a separate diagnostic after job 80368 established different RGB/tensor data at frame 11220 (374 seconds). It does not change full1337 v1, its guards, or the 5,665 committed records.

## Fixed cases and evidence

Four policies, each repeated from a fresh reader:

- Decord indexed reads, one decoder thread.
- Decord sequential `next()`, one decoder thread.
- Decord sequential `next()`, two decoder threads.
- PyAV sequential `container.decode(video=0)`, one decoder thread.

Sequential readers decode every preceding frame from ordinal zero, never seek, reject backwards requests and retain only the current frame. RGB conversion is done for requested candidates and five probes around frame 11220. They drain to EOF after scoring to record the actual decoded count. PyAV also records presentation timestamps and their difference from the reference ordinal/FPS timeline. A count or PTS mismatch must be assessed before comparing frame identities across decoders.

All cases score the same public question with frozen BLIP ITM FP32 and original candidate grid/weights. **Molmo2 and Qwen are not loaded; there are no answer forwards.** Histories and allocator state therefore differ from the original Molmo2 job; this diagnostic tests candidate decoding policies, not exact replay of the full job.

The report includes all candidate RGB/tensor hashes, score vectors, repeat comparisons, cross-policy comparisons, source-frame probes and comparisons with the cold/failing conditions in job 80368. Each sequential probe also has a 32×18 RGB thumbnail, encoded in base64, for comparing image content if library color conversion prevents exact hash matching. Probe matches can help identify a possible shifted/duplicated frame, but no decoder is automatically labelled ground truth. RGB hashes may differ across libraries because of color conversion; this alone is not evidence of a wrong frame. Selection overlap with old sigma2 is a diagnostic, not a criterion for choosing the policy.

The reference `session-80368/report.json` is pinned by SHA256. Full-run contract, prediction export and failed cache hashes must remain unchanged. Output is diagnostic-only, never a replacement score cache or evaluation result. Repeatability on one video does not validate every video or permit mixing decoder protocols in an existing run.

## Run on the school server

```bash
cd /media/lnthanh03/DatHa/code/VideoQA-EvidenceLab-ridge-v1
git pull --ff-only https://github.com/tdat1465/VideoQA-EvidenceLab.git codex/ridge-molmo2
bash scripts/slurm/run_ridge_decode_policy.sh --node gpu03 --reference-job 80368
```

One GPU, eight CPUs, 90 GiB RAM, four-hour maximum including setup. Models, libraries and video stay in tmpfs. Default node gpu03 matches the preceding diagnostic's A100 PCIe family. The launcher checks for active RIDGE/diagnostic jobs and runs Slurm's placement test before one submission. No artificial keepalive or full-run resume is performed.

```bash
squeue -u lnthanh03 -n ridge-decpol -o '%.18i %.9T %.30R'
```

Logs: `logs/ridge-decpol-JOBID.out` and `.err`. The self-contained report is under `/media/lnthanh03/DatHa/runs/videoqa/lvb-ridge-decode-policy-o2f-v1/session-JOBID/report.json`. Completed cases are saved immediately; partial reports survive controlled interruption.

After worker exit, download from Windows PowerShell, replacing JOBID:

```powershell
powershell -ExecutionPolicy Bypass -File "E:\Codex\Documents\ChatGPT\KLTN\research-ridge\VideoQA-EvidenceLab\scripts\download_ridge_decode_policy.ps1" -JobId JOBID
```

The file is saved to `E:\Codex\Documents\ChatGPT\KLTN\outputs\ridge-decode-policy-results-v1\decode-JOBID.json`. One SCP connection is used. Completion means `complete=true`, `snapshot_unchanged=true`, `reference_unchanged=true` and eight recorded cases; it does not mean a decoder policy has passed all criteria.

Implementation references: Decord 0.6.0 `VideoReader.next()` source distributed in the pinned wheel; [PyAV decoding API](https://pyav.org/docs/stable/api/container.html#av.container.InputContainer.decode) and [codec thread configuration](https://pyav.org/docs/stable/api/codec.html#av.codec.context.CodecContext.thread_count). The server still uses the pinned `av==15.1.0` runtime; no package upgrade is introduced.
