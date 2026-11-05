# LongVideoBench experiments (2026-11-05)

Results on the 1,337-question LongVideoBench validation set. These experiments used validation data during development; they do not establish independent-test performance. Video-MME preflight did not produce an accuracy result.

| Run | Records | Description |
|---|---:|---|
| molmo2-subtitle | 4,011 | event coverage sigma2, one subtitle anchor, multiple subtitle anchors |
| qwen | 2,674 | Qwen2.5-VL-7B RIDGE sigma2 vs multi-anchor |
| llava | 2,674 | LLaVA-OneVision-7B RIDGE sigma2 vs multi-anchor |
| internvl | 2,674 | InternVL3-8B RIDGE sigma2 vs multi-anchor |
| molmo2-relational | 5,348 | sigma2, event coverage and relational ablations |
| molmo2-canonical | 8,022 | canonical six-arm 64-frame evaluation |

The per-question records include frame indices, model predictions and correctness. Some runs reuse predictions when frame sets are identical; rows are not all independent forward passes. See the linked audit and report files for methodology and limits. For the subtitle Molmo2 run, the checkpoint contained 4,011 distinct question/arm records and five zero worker exit codes; the archive CRC was verified. Multi-anchor changed frames for 427 questions vs event coverage, and moved 25 wrong-to-correct and 12 correct-to-wrong (911 to 924 correct).

`manifest.json` records each exported file's source and SHA-256. The relational and canonical exports omit benchmark question text, choices, correct answer, detailed scores and execution metadata; originals and large score caches remain in local archives. The smaller exports retain their original prediction fields. Raw videos, subtitles, annotations, credentials, runtime logs, caches and checkpoints are not distributed. LongVideoBench is available from [the official dataset page](https://huggingface.co/datasets/longvideobench/LongVideoBench) under CC BY-NC-SA 4.0, subject to its access conditions.
