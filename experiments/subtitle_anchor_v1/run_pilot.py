"""Exploratory, visual-only Molmo2 comparison at a fixed frame budget."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path("/marimo/ridge-canonical")
PILOT = ROOT / "subtitle-pilot"
sys.path.insert(0, str(ROOT / "relational-w5-v1"))
from worker_sigma2 import (
    frames_full, load_full_inputs, LongVideoConfig, HFBackend, Question,
    MultipartReader, HTTPRangeSource, archive_index, fetch_video, check_space,
    TRANSFER_CHUNK_BYTES, Image, decord, digest,
)


def select(base, count, fps, record, limit):
    offset = float(record["offset"] or 0)
    anchors = []
    for matches in record["matches"]:
        candidates = []
        for _, _, start, end in matches["candidates"]:
            t = (start + end) / 2 - offset
            if 0 <= t < count / fps:
                candidates.append(t)
        anchors.append(candidates[:limit])
    times = list(dict.fromkeys(t for rank in range(limit) for group in anchors for t in group[rank:rank+1]))
    # Spread across subtitle options before allocating a second location per option.
    targeted = list(dict.fromkeys(
        max(0, min(count - 1, round((t + shift) * fps)))
        for shift in (0, 1, -1, 2, -2, 4, -4, 8, -8)
        for t in times
    ))[:16]
    return sorted((targeted + [i for i in base if i not in targeted])[:len(base)])


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-rank", type=int, choices=range(5))
    args = parser.parse_args()
    full = args.full_rank is not None
    torch.set_num_threads(2)
    if full:
        torch.cuda.set_per_process_memory_fraction(0.18)
    inputs = load_full_inputs(ROOT)
    folder = ROOT / "subtitle-full-v1" if full else PILOT
    plan = json.loads((folder / ("anchors.json" if full else "anchor-window-preflight.json")).read_text())
    rows = {}
    with (ROOT / "results-relational-w5-v1/predictions.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r["variant"] == "eventcover_sigma2":
                rows[r["question_id"]] = r
    if full:
        plan = [r for r in plan if int(hashlib.sha256(rows[r["id"]]["video"].encode()).hexdigest(), 16) % 5 == args.full_rank]
    output = folder / (f"predictions-{args.full_rank}.jsonl" if full else "pilot-predictions.jsonl")
    done = set()
    if output.exists():
        done = {(r["id"], r["arm"]) for line in output.read_text().splitlines() if (r := json.loads(line))}
    reader = MultipartReader(HTTPRangeSource())
    active = {rows[x["id"]]["video"] for x in plan if not full or any(q["candidates"] for q in x["matches"])}
    members = archive_index(reader, ROOT / "archive-index.json", active) if active else {}
    reader.clear_cache()
    reader.block_size = TRANSFER_CHUNK_BYTES
    backend = HFBackend(LongVideoConfig(**inputs["config"]))
    with output.open("a") as f:
        for rec in plan:
            row = rows[rec["id"]]
            arms = ("eventcover_sigma2", "top1_anchor", "multi_anchor")
            if all((rec["id"], arm) in done for arm in arms):
                continue
            if full and not any(q["candidates"] for q in rec["matches"]):
                for arm in arms:
                    if (rec["id"], arm) not in done:
                        f.write(json.dumps(dict(id=rec["id"], arm=arm, frames=row["frame_indices"],
                            prediction=row["prediction"], correct=row["correct"], kind=rec["kind"],
                            seconds=0, reused_eventcover=True)) + "\n")
                f.flush()
                continue
            print("VIDEO", rec["id"], flush=True)
            with tempfile.TemporaryDirectory(dir="/dev/shm", prefix="ridge-subtitle-") as temp:
                root = Path(temp)
                path = root / "video.mp4"
                sha = fetch_video(reader, members["videos/" + row["video"]], path)
                reader.clear_cache()
                assert sha == inputs["baselines"][rec["id"]]["video_sha256"]
                native = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
                count, fps = len(native), float(native.get_avg_fps())
                del native
                base = row["frame_indices"]
                sets = [base, select(base, count, fps, rec, 1), select(base, count, fps, rec, 3)]
                assert 0 < len(base) <= 64
                assert all(len(indices) == len(set(indices)) == len(base) for indices in sets)
                cache = frames_full.build(path, root / "pixels", count, fps, set().union(*sets), sha, check_space)
                references = list((ROOT / "reference-full1337-v1").glob("shard-*/pixels/" + digest(row["video"]) + ".json"))
                assert len(references) == 1
                reference = json.loads(references[0].read_text())
                assert all(reference["frames"][str(i)] == cache.manifest["frames"][str(i)] for i in base)
                question = Question(rec["id"], row["question"], tuple(row["choices"]))
                predictions = {tuple(base): row["prediction"]} if full else {}
                for arm, indices in zip(arms, sets):
                    if (rec["id"], arm) in done:
                        continue
                    begin = time.monotonic()
                    key = tuple(indices)
                    if key not in predictions:
                        backend.video_duration = count / fps
                        predictions[key] = backend.answer(question, [Image.fromarray(cache[i].asnumpy()) for i in indices], [i/fps for i in indices]).prediction
                    prediction = predictions[key]
                    result = dict(id=rec["id"], arm=arm, frames=indices, prediction=prediction,
                                  correct=prediction == row["answer"], kind=rec["kind"], seconds=time.monotonic()-begin,
                                  reused_eventcover=full and indices == base)
                    f.write(json.dumps(result) + "\n")
                    f.flush()
    print("PILOT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
