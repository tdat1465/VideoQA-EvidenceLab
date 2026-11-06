"""Fresh paired Molmo2 inference for the protected subtitle replacement experiment."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import gc
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src"),
               str(REPO / "experiments/subtitle_anchor_v1/relational")]

from selection import POLICY, protected_select, subtitle_targets
from evidencelab.data import digest, file_hash
from evidencelab.store import Store, atomic_json, run_lock
from evidencelab.longvideo_data import parse_annotations
from evidencelab.subtitle_anchors import candidates, queries, select, windows

ARMS = ("ridge_sigma2", "eventcover_sigma2", "multi_anchor_v1", "protected_v2")
ANNOTATION_SHA = "d32bf2e0cbbdb4fa2e99666a5a49b5a19bca2ceec69222c9a7d3aad567d1fa23"
RESULTS = REPO / "results/longvideobench-20261105"


@contextmanager
def session_stop(minutes):
    stopped = []
    deadline = time.monotonic() + minutes * 60
    old_handlers = {s: signal.signal(s, lambda signum, frame: stopped.append(signum))
                    for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield lambda: bool(stopped) or time.monotonic() >= deadline
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def read_predictions(folder):
    path = RESULTS / folder / "predictions.jsonl"
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    expected = next(r for r in manifest if r["file"] == f"{folder}/predictions.jsonl")
    if file_hash(path) != expected["sha256"]:
        raise ValueError(f"Published frames changed: {path}")
    groups = defaultdict(dict)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        ident = row.get("question_id", row["id"])
        arm = row.get("variant", row.get("arm"))
        if ident in groups[arm]:
            raise ValueError("Duplicate published prediction")
        groups[arm][ident] = row
    return groups


def prepare(args):
    if file_hash(args.annotation) != ANNOTATION_SHA:
        raise ValueError("Annotation differs from the independently audited lvb_val.json")
    raw = json.loads(args.annotation.read_text(encoding="utf-8"))
    samples = parse_annotations(raw)
    ids = {s.id for s in samples}
    if len(ids) != 1337:
        raise ValueError("Require all 1337 validation questions")
    relational = read_predictions("molmo2-relational")
    subtitle = read_predictions("molmo2-subtitle")
    frozen = {"ridge_sigma2": relational["sigma2"],
              "eventcover_sigma2": relational["eventcover_sigma2"],
              "multi_anchor_v1": subtitle["multi_anchor"]}
    for arm, rows in frozen.items():
        if set(rows) != ids:
            raise ValueError(f"Incomplete frame manifest: {arm}")
        for sample in samples:
            row = rows[sample.id]
            frames = row.get("frame_indices", row.get("frames"))
            if (frames != sorted(set(frames)) or not 0 < len(frames) <= 64
                    or row["correct"] != (row["prediction"] == sample.answer)):
                raise ValueError(f"Invalid frozen baseline: {sample.id}")
            row["frames"] = frames
    if args.anchors:
        records = json.loads(args.anchors.read_text())
        if len(records) != 1337 or len({r["id"] for r in records}) != 1337:
            raise ValueError("Require the complete original 1337-question anchors.json")
        plan = {r["id"]: r for r in records}
        plan_source = {"anchors_sha256": file_hash(args.anchors)}
    else:
        plan = {}
        with tarfile.open(args.subtitles) as archive:
            for row in raw:
                query_texts = queries(row["question"], row["candidates"], row["question_category"])
                spans = []
                if query_texts and row.get("subtitle_path"):
                    with archive.extractfile("subtitles/" + row["subtitle_path"]) as stream:
                        spans = list(windows(json.load(stream)))
                ident = "longvideobench:val:" + str(row["id"])
                plan[ident] = {"id": ident, "kind": row["question_category"],
                               "offset": row["starting_timestamp_for_subtitles"],
                               "matches": [{"candidates": candidates(q, spans)} for q in query_texts]}
        plan_source = {"subtitles_sha256": file_hash(args.subtitles)}
    if set(plan) != ids or any(plan[s.id]["kind"] != s.question_type for s in samples):
        raise ValueError("Subtitle plan IDs/categories differ from annotations")
    sources = {folder: file_hash(RESULTS / folder / "predictions.jsonl")
               for folder in ("molmo2-relational", "molmo2-subtitle")}
    return samples, frozen, plan, {"annotation_sha256": ANNOTATION_SHA,
                                   "frame_sources": sources, "plan_sha256": digest(plan),
                                   "plan_source": plan_source}


def summary(rows, target_questions=1337):
    groups = defaultdict(dict)
    for row in rows:
        groups[row["question_id"]][row["variant"]] = row
    complete = {q: group for q, group in groups.items() if set(group) == set(ARMS)}
    counts = {arm: {"n": sum(r["variant"] == arm for r in rows),
                    "correct": sum(r["variant"] == arm and r["correct"] for r in rows)}
              for arm in ARMS}
    for value in counts.values():
        value["accuracy"] = 100 * value["correct"] / value["n"] if value["n"] else None
    comparisons = {}
    for arm in ARMS[:-1]:
        gains = sum(not g[arm]["correct"] and g["protected_v2"]["correct"] for g in complete.values())
        losses = sum(g[arm]["correct"] and not g["protected_v2"]["correct"] for g in complete.values())
        item = {"paired_n": len(complete), "gains": gains, "losses": losses,
                "delta_pp": 100 * (gains - losses) / len(complete) if complete else None}
        if len(complete) == 1337:
            import numpy as np
            by_video = defaultdict(lambda: [0, 0])
            for ident in sorted(complete):
                g = complete[ident]
                values = by_video[g[arm]["video"]]
                values[0] += int(g["protected_v2"]["correct"]) - int(g[arm]["correct"])
                values[1] += 1
            values = np.array([by_video[v] for v in sorted(by_video)])
            rng, deltas = np.random.default_rng(20261102), []
            for _ in range(100):
                indexes = rng.integers(0, len(values), size=(100, len(values)))
                totals = values[indexes].sum(axis=1)
                deltas.extend(100 * totals[:, 0] / totals[:, 1])
            item["video_bootstrap_95ci_pp"] = np.quantile(deltas, [.025, .975]).tolist()
        comparisons[arm] = item
    return {"complete": len(complete) == target_questions, "completed_questions": len(complete),
            "target_questions": target_questions, "records": len(rows), "arms": counts,
            "protected_v2_vs": comparisons,
            "fresh_answer_forwards": sum(r["reused_from"] is None for r in rows),
            "note": "Partial runs are pilot diagnostics. Validation was inspected during development."}


def export(output, store, target_questions=1337):
    rows = store.results()
    result = summary(rows, target_questions)
    atomic_json(output / "summary.json", result)
    temporary = output / "predictions.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(output / "predictions.jsonl")
    return result


def commit_arms(store, sample, sets, stamp, details, answer, stop):
    previous = {r["variant"]: r for r in store.results() if r["question_id"] == sample.id}
    for arm, row in previous.items():
        if row["frame_indices"] != sets[arm] or row["pixel_manifest_sha256"] != stamp:
            raise ValueError("Resume frame selection or pixels changed")
    same_frames = {tuple(r["frame_indices"]): r for r in previous.values()}
    for arm in ARMS:
        if arm in previous:
            continue
        if stop():
            raise InterruptedError("Stopped between committed variants")
        frames = sets[arm]
        prior = same_frames.get(tuple(frames))
        started = time.perf_counter()
        if prior is None:
            decision = answer(frames)
            scores, tokens = decision.scores, decision.input_tokens
            if (len(scores) != len(sample.choices)
                    or any(not math.isfinite(x) or not 0 <= x <= 1 for x in scores)
                    or abs(sum(scores) - 1) > 1e-4):
                raise ValueError("Invalid answer probabilities")
            prediction = decision.prediction
        else:
            scores, tokens, prediction = prior["scores"], prior["input_tokens"], prior["prediction"]
        if type(prediction) is not int or not 0 <= prediction < len(sample.choices):
            raise ValueError("Invalid answer prediction")
        row = {"id": sample.id + "::" + arm, "question_id": sample.id, "variant": arm,
               "video": sample.video, "question_type": sample.question_type,
               "frame_indices": frames, "prediction": prediction, "answer": sample.answer,
               "correct": prediction == sample.answer, "scores": scores, "input_tokens": tokens,
               "answer_seconds": time.perf_counter() - started,
               "reused_from": None if prior is None else prior["id"],
               "pixel_manifest_sha256": stamp, "selection": details if arm == "protected_v2" else None}
        store.put(row)
        same_frames[tuple(frames)] = row
        print(f"COMMIT {sample.id} {arm} correct={row['correct']}", flush=True)


def run(args):
    samples, frozen, plan, inputs = prepare(args)
    target_questions = 1337
    shard = getattr(args, "question_ids", None)
    if shard is not None:
        ids = json.loads(shard.read_text())
        known = {s.id for s in samples}
        if not ids or len(set(ids)) != len(ids) or not set(ids) <= known:
            raise ValueError("Invalid shard question manifest")
        samples = [s for s in samples if s.id in set(ids)]
        target_questions = len(ids)
        inputs["question_ids_sha256"] = digest(sorted(ids))
    if args.preflight:
        print(json.dumps({"questions": len(samples), "arms": ARMS, "inputs": inputs,
                          "subtitle_matched_questions": sum(any(m["candidates"] for m in r["matches"])
                                                            for r in plan.values())}, indent=2))
        return 0
    import torch
    import decord
    from PIL import Image
    import frames_full
    from evidencelab.aks_adapter import aks_candidates
    from evidencelab.backends import HFBackend
    from evidencelab.config import source_fingerprint
    from evidencelab.focus_adapter import BlipITMScorer
    from evidencelab.hardware import execution_profile
    from evidencelab.longvideo import longvideo_runtime
    from evidencelab.longvideo_config import LongVideoConfig
    from evidencelab.longvideo_data import (MultipartReader, HTTPRangeSource, archive_index,
                                          fetch_video, check_space, TRANSFER_CHUNK_BYTES, SOURCE)

    args.ram.mkdir(parents=True, exist_ok=True)
    if subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "--target", str(args.ram)], text=True).strip() != "tmpfs":
        raise ValueError("Select a tmpfs RAM directory, e.g. /dev/shm")
    config = LongVideoConfig(method="ridge", frames=64, limit=0, dtype=args.dtype)
    execution = execution_profile(config)
    contract = {"schema": "subtitle-protected-molmo2-v2", "inputs": inputs, "policy": POLICY,
                "arms": ARMS, "config": config.contract(), "source_sha256": source_fingerprint(),
                "code": {p.name: file_hash(p) for p in (Path(__file__), HERE / "selection.py", Path(frames_full.__file__))},
                "runtime": longvideo_runtime(config),
                "execution": {k: v for k, v in execution.items() if k != "node"}, "dataset": SOURCE,
                "inference_policy": "fresh-controls; identical-frame reuse within this journal only",
                "auxiliary_scoring": "fresh question-only BLIP on eventcover base frames"}
    backend = scorer = reader = None
    with session_stop(args.session_minutes) as stop, run_lock(args.output):
        store = Store(args.output, contract, (args.output / "results.sqlite3").exists())
        try:
            for folder in ("pixels", "scores"):
                (args.output / folder).mkdir(exist_ok=True)
            done = store.done()
            # Enrich the pilot for subtitle changes, using public matches only.
            ordered = sorted(samples, key=lambda s: (
                not any(m["candidates"] for m in plan[s.id]["matches"]), digest(s.id)))
            pending = [s for s in ordered
                       if any(s.id + "::" + a not in done for a in ARMS)]
            if args.max_new_questions:
                pending = pending[:args.max_new_questions]
            by_video = defaultdict(list)
            for sample in pending:
                by_video[sample.video].append(sample)
            reader = MultipartReader(HTTPRangeSource(stop=stop))
            # The archive index has its own run_lock, separate from the journal lock.
            index_path = getattr(args, "archive_index", None) or args.output / "index" / "archive-index.json"
            members = archive_index(reader, index_path, set(by_video)) if by_video else {}
            reader.clear_cache()
            reader.block_size = TRANSFER_CHUNK_BYTES
            for video, batch in by_video.items():
                if stop():
                    raise InterruptedError("Session budget reached")
                with tempfile.TemporaryDirectory(dir=args.ram, prefix="ridge-protected-") as folder:
                    temporary = Path(folder)
                    path = temporary / "video.mp4"
                    begin = time.perf_counter()
                    sha = fetch_video(reader, members["videos/" + video], path, stop=stop)
                    reader.clear_cache()
                    native = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
                    count, fps = len(native), float(native.get_avg_fps())
                    del native
                    grid = aks_candidates(count, fps)
                    # Use ALL questions of the video for a stable manifest across
                    # pilot/full, partially committed questions and later resumes.
                    video_samples = [s for s in samples if s.video == video]
                    union, targets = set(), {}
                    for sample in video_samples:
                        base = frozen["eventcover_sigma2"][sample.id]["frames"]
                        target = subtitle_targets(count, fps, plan[sample.id])
                        old = select(base, count, fps, plan[sample.id], 3)
                        if old != frozen["multi_anchor_v1"][sample.id]["frames"]:
                            raise ValueError(f"Original subtitle selector replay differs: {sample.id}")
                        if any(len(frozen[a][sample.id]["frames"]) != len(base) for a in ARMS[:-1]):
                            raise ValueError("Frozen controls have unequal frame budgets")
                        if old != sorted((target + [i for i in base if i not in target])[:len(base)]):
                            raise ValueError("Subtitle target proposals differ from v1")
                        targets[sample.id] = target
                        union.update(target)
                        for arm in ARMS[:-1]:
                            union.update(frozen[arm][sample.id]["frames"])
                    pixels = frames_full.build(path, temporary / "pixels", count, fps, union, sha, check_space, stop)
                    stamp = digest(pixels.manifest)
                    saved = args.output / "pixels" / (digest(video) + ".json")
                    if saved.exists() and json.loads(saved.read_text()) != pixels.manifest:
                        raise ValueError("Resume source pixels changed")
                    atomic_json(saved, pixels.manifest)
                    store.event({"video": video, "download_decode_seconds": time.perf_counter() - begin})
                    for sample in batch:
                        if stop():
                            raise InterruptedError("Session budget reached")
                        base = frozen["eventcover_sigma2"][sample.id]["frames"]
                        target = targets[sample.id]
                        key = {"contract": digest(contract), "pixels": stamp, "question": sample.question,
                               "base": base, "question_id": sample.id}
                        cache_path = args.output / "scores" / (digest(sample.id) + ".json")
                        if cache_path.exists():
                            record = json.loads(cache_path.read_text())
                            if (record["key"] != key or record["sha256"] != digest(
                                    {k: v for k, v in record.items() if k != "sha256"})):
                                raise ValueError("Resume BLIP cache changed")
                        else:
                            begin = time.perf_counter()
                            values = [0.] * len(base)
                            if set(target) - set(base):
                                if scorer is None:
                                    scorer = BlipITMScorer(config)
                                scorer.reset()
                                values = scorer(pixels, sample.question, base)
                            record = {"key": key, "values": values,
                                      "scored_frames": len(base) if set(target) - set(base) else 0,
                                      "seconds": time.perf_counter() - begin}
                            record["sha256"] = digest(record)
                            atomic_json(cache_path, record)
                        begin = time.perf_counter()
                        chosen, details = protected_select(base, target, count, grid, dict(zip(base, record["values"])))
                        details.update(selector_seconds=time.perf_counter() - begin,
                                       blip_cache_sha256=record["sha256"],
                                       blip_seconds=record["seconds"], blip_scored_frames=record["scored_frames"])
                        sets = {a: frozen[a][sample.id]["frames"] for a in ARMS[:-1]}
                        sets["protected_v2"] = chosen
                        if backend is None:
                            backend = HFBackend(config)
                        backend.video_duration = count / fps

                        def answer(indices):
                            check_space(args.ram, 0)
                            return backend.answer(sample.public_question(),
                                                  [Image.fromarray(pixels[i].asnumpy()) for i in indices],
                                                  [i / fps for i in indices])

                        commit_arms(store, sample, sets, stamp, details, answer, stop)
                    del pixels
        except InterruptedError as error:
            store.event({"event": "controlled_stop", "reason": str(error)})
        finally:
            if reader is not None:
                reader.clear_cache()
            result = export(args.output, store, target_questions)
            store.close()
            del backend, scorer
            gc.collect()
            torch.cuda.empty_cache()
        print(json.dumps(result, indent=2), flush=True)
        return 0 if result["complete"] else 75


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--anchors", type=Path)
    source.add_argument("--subtitles", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ram", type=Path, default=Path("/dev/shm"))
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="auto")
    parser.add_argument("--max-new-questions", type=int, default=20)
    parser.add_argument("--session-minutes", type=int, default=120)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--question-ids", type=Path)
    parser.add_argument("--archive-index", type=Path)
    args = parser.parse_args()
    if args.max_new_questions < 0 or args.session_minutes < 1:
        parser.error("Invalid question/session limit")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
