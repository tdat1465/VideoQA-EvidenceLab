# Protected subtitle selection on LongVideoBench

The Molmo2-4B run completed all 1,337 validation questions with four workers.
The checkpoint contains 5,348 results. Every stored correctness flag matched
the original validation annotation when recomputed from the predicted option.

| Method | Correct | Accuracy |
|---|---:|---:|
| RIDGE sigma 2 | 901 / 1,337 | 67.39% |
| Event coverage sigma 2 | 910 / 1,337 | 68.06% |
| Multi-anchor v1 | 923 / 1,337 | 69.04% |
| Protected v2 | 920 / 1,337 | 68.81% |

Protected v2 gained 1.42 percentage points over RIDGE and lost 0.22 points
against multi-anchor v1. The video bootstrap 95% interval for the latter
difference was −0.98 to +0.52 points.

`checkpoint.zip` contains the four SQLite journals, shard assignments, run
contracts, score caches, pixel manifests, archive index, predictions, and
summary. The same summary is available as `summary.json` beside the archive.

SHA-256 for `checkpoint.zip`:

```text
022463745fd28bc39af26e13ca063bb3fceb67eaa64a8af0822eb1ed82bdac93
```

To open the journals or resume a checkpoint with the same schema, extract the
archive into the full run output directory. Use the pinned LongVideoBench
annotation and subtitle archive with the environment recorded in each worker's
contract. The supervisor preserves each worker's question assignment and
skips predictions already recorded in SQLite.

```bash
unzip results/longvideobench-full4/checkpoint.zip -d runs/subtitle-protected-v2-full4
python experiments/subtitle_coverage_v2/full.py \
  --annotation /path/to/lvb_val.json \
  --subtitles /path/to/subtitles.tar \
  --archive-index runs/subtitle-protected-v2-full4/index/archive-index.json \
  --output runs/subtitle-protected-v2-full4
```
