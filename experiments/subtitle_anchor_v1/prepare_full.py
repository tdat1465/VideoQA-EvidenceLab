"""Build the frozen subtitle-anchor plan and supervise five inference workers."""
import json
import re
import subprocess
import sys
import tarfile
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path("/marimo/ridge-canonical")
OUT = ROOT / "subtitle-full-v1"
DATA = ROOT / "subtitle-pilot"


def seconds(value):
    if isinstance(value, str):
        return sum(float(x) * 60**i for i, x in enumerate(reversed(value.replace(",", ".").split(":"))))
    return float(value)


def tokens(text):
    return re.findall(r"\w+", text.casefold())


def windows(rows):
    cues = []
    for row in rows:
        start, end = row.get("timestamp") or (row.get("start"), row.get("end"))
        if start is not None and end is not None:
            cues.append((seconds(start), seconds(end), tokens(row.get("text") or row.get("line") or "")))
    cues.sort()
    for index, (start, _, _) in enumerate(cues):
        words = []
        for _, end, text in cues[index:index+4]:
            if end - start > 20:
                break
            words.extend(text)
            if words:
                yield start, end, words.copy()


def candidates(query, spans):
    query = tokens(query)
    if len(query) < 3:
        return []
    ranked = []
    for start, end, words in spans:
        matcher = SequenceMatcher(None, query, words, autojunk=False)
        coverage = sum(block.size for block in matcher.get_matching_blocks()) / len(query)
        if coverage >= .7:
            ranked.append((coverage, matcher.ratio(), start, end))
    keep = []
    for row in sorted(ranked, reverse=True):
        if all(abs(row[2] - old[2]) >= 5 for old in keep):
            keep.append(row)
            if len(keep) == 3:
                break
    return keep


def main():
    meta = {r["id"]: r for r in json.loads((DATA / "lvb_val.json").read_text())}
    with (ROOT / "results-relational-w5-v1/predictions.jsonl").open() as f:
        rows = [r for line in f if (r := json.loads(line))["variant"] == "eventcover_sigma2"]
    assert len(rows) == len({r["question_id"] for r in rows}) == 1337
    plan = []
    with tarfile.open(DATA / "subtitles.tar") as archive:
        for row in sorted(rows, key=lambda r: (r["video"], r["question_id"])):
            key, kind = row["question_id"], row["question_type"]
            m = meta[key.split(":", 2)[-1]]
            queries = m["candidates"] if kind == "TOS" else [
                a or b for a, b in re.findall(r"'([^']{8,})'|\"([^\"]{8,})\"", m["question"])
            ] if kind.startswith("T") else []
            matches = []
            if queries and m.get("subtitle_path"):
                with archive.extractfile("subtitles/" + m["subtitle_path"]) as stream:
                    spans = list(windows(json.load(stream)))
                matches = [{"candidates": candidates(q, spans)} for q in queries]
            plan.append(dict(id=key, kind=kind, offset=m["starting_timestamp_for_subtitles"], matches=matches))
    (OUT / "anchors.json").write_text(json.dumps(plan))
    print("PLAN_READY", len(plan), "eligible", sum(any(q["candidates"] for q in r["matches"]) for r in plan), flush=True)
    workers = []
    for rank in range(5):
        with (OUT / f"worker-{rank}.log").open("a") as log:
            worker = subprocess.Popen([sys.executable, "-u", str(OUT / "run_pilot.py"), "--full-rank", str(rank)],
                                      stdout=log, stderr=subprocess.STDOUT)
        workers.append(worker)
        print("WORKER", rank, worker.pid, flush=True)
    codes = [worker.wait() for worker in workers]
    (OUT / "worker-exit-codes.json").write_text(json.dumps(codes))
    if any(codes):
        raise SystemExit(f"Worker failures: {codes}")
    print("FULL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
