"""Detached five-worker experiment with durable progress and final export."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
import traceback
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parent))
from parallel_common_full import partitions, summarize, gpu_fraction, load_full_inputs

ROOT = Path('/marimo/ridge-canonical')
SOURCE = Path(__file__).resolve().parent
OUTPUT = ROOT / 'results-relational-w5-v1'
VARIANTS = ('sigma2', 'eventcover_sigma2', 'dedup_sigma2', 'relation_sigma2')


def atomic(path, data):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    tmp.replace(path)


def rows_at(path):
    if not path.exists():
        return []
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True, timeout=30) as db:
        return [json.loads(r[0]) for r in db.execute('SELECT value FROM results')]


def snapshot(processes, shards, ids, phase):
    rows = []
    for rank in range(5):
        own = rows_at(OUTPUT/f'shard-{rank}/results.sqlite3')
        summarize(own, shards[rank], VARIANTS)
        rows.extend(own)
    state = summarize(rows, ids, VARIANTS)
    state.update(phase=phase, workers=5, supervisor_pid=os.getpid(), updated_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                 worker_pids=[p.pid for p in processes], worker_exit_codes=[p.poll() for p in processes],
                 evaluation_note='Post-hoc development on previously inspected validation; no independent test claim.')
    atomic(OUTPUT/'summary.json', state)
    atomic(OUTPUT/'state.json', state)
    return state, rows


def package():
    inventory = []
    for path in sorted(OUTPUT.rglob('*')):
        if path.is_file() and not path.name.endswith(('.lock', '.tmp', '-wal', '-shm')):
            inventory.append({'path': 'results/'+path.relative_to(OUTPUT).as_posix(),
                              'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    for path in sorted(SOURCE.glob('*')):
        if path.is_file() and path.suffix in ('.py', '.json', '.md'):
            inventory.append({'path': 'source/'+path.name, 'bytes': path.stat().st_size,
                              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    archive = ROOT/'ridge-relational-w5-v1-complete.zip'
    with zipfile.ZipFile(str(archive)+'.tmp', 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('EXPORT_INVENTORY.json', json.dumps(inventory, indent=2))
        z.write(ROOT/'inputs-full1337.json', 'inputs-full1337.json')
        for entry in inventory:
            prefix, rel = entry['path'].split('/', 1)
            z.write((OUTPUT if prefix == 'results' else SOURCE)/rel, entry['path'])
    Path(str(archive)+'.tmp').replace(archive)
    return {'path': str(archive), 'bytes': archive.stat().st_size, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, choices=[5], default=5)
    global OUTPUT
    p.add_argument('--pilot', action='store_true')
    args=p.parse_args()
    if args.pilot: OUTPUT=Path(str(OUTPUT)+'-pilot')
    OUTPUT.mkdir(exist_ok=True)
    import fcntl
    lock = (OUTPUT/'driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    processes, stop = [], []
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda s, f: stop.append(s))
    shards = None
    success = False
    try:
        atomic(OUTPUT/'state.json', {'phase': 'loading_inputs', 'workers': 5, 'supervisor_pid': os.getpid()})
        inputs = load_full_inputs(ROOT)
        shards = partitions(inputs, 5)
        if args.pilot:shards=[s[:1] for s in shards]
        ids=sorted(q for s in shards for q in s)
        atomic(OUTPUT/'assignment.json', {'workers': 5, 'shards': shards, 'variants': VARIANTS,
               'target_questions': len(ids), 'target_variant_answers': len(ids)*len(VARIANTS)})
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        for rank in range(5):
            with (OUTPUT/f'worker-{rank}.log').open('a') as log:
                proc = subprocess.Popen([str(ROOT/'venv-active/bin/python'), '-u', str(SOURCE/'worker_sigma2.py'),
                                         '--rank', str(rank), '--workers', '5']+(['--pilot'] if args.pilot else []), env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append(proc)
            print('Started worker', rank, 'pid', proc.pid, 'questions', len(shards[rank]), 'VRAM fraction', gpu_fraction(5), flush=True)
        while True:
            state, rows = snapshot(processes, shards, ids, 'running')
            if stop or any(p.poll() not in (None, 0) for p in processes):
                break
            if all(p.poll() is not None for p in processes):
                success = state['complete']
                with (OUTPUT/'predictions.jsonl').open('w', encoding='utf-8') as stream:
                    for row in rows:
                        stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                break
            time.sleep(10)
    except Exception:
        atomic(OUTPUT/'failure.json', {'error': traceback.format_exc()})
        traceback.print_exc()
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            try:
                proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        phase = 'completed' if success else ('stopped' if stop else 'failed')
        if shards is not None:
            snapshot(processes, shards, ids, phase)
        else:
            atomic(OUTPUT/'state.json', {'phase': phase, 'supervisor_pid': os.getpid()})
        if success and not args.pilot:
            try:
                atomic(OUTPUT/'archive.json', package())
            except Exception:
                atomic(OUTPUT/'export_failure.json', {'error': traceback.format_exc()})
                traceback.print_exc()
                success=False
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
