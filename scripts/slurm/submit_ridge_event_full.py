"""Submit or resume the separate full1337 event-coverage run on one BF16 GPU."""
import argparse
from contextlib import closing
import getpass
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

from submit import ALLOWED, submission_args
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'scripts'), str(REPO / 'src')]
import ridge_event_full as full

ROOT = Path('/media/lnthanh03/DatHa')
BASELINE = ROOT / 'runs/videoqa/lvb-val-molmo2-ridge64-full-ridge-v1'
SOURCE = ROOT / 'runs/videoqa/lvb-ridge-smoothing-v1-dev200'
DEVELOPMENT = ROOT / 'runs/videoqa/lvb-ridge-event-coverage-v1-dev200'
OUTPUT = ROOT / 'runs/videoqa/lvb-ridge-event-coverage-full1337-v1'


def progress(path, ids):
    if not path.exists():
        return 0, 0
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        rows = [json.loads(x) for x, in db.execute('SELECT value FROM results')]
    groups = {}
    for r in rows:
        if (r['question_id'] not in ids or r['variant'] not in full.VARIANTS
                or r['id'] != r['question_id'] + '::' + r['variant']):
            raise ValueError('Foreign full-run result')
        groups.setdefault(r['question_id'], set()).add(r['variant'])
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate full-run results')
    return len(rows), sum(g == set(full.VARIANTS) for g in groups.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['full1337'], nargs='?', default='full1337')
    p.add_argument('--node', choices=['any'] + sorted(ALLOWED), default='any')
    p.add_argument('--session-minutes', type=int, default=2880)
    action = p.add_mutually_exclusive_group()
    action.add_argument('--submit', action='store_true')
    action.add_argument('--test-only', action='store_true')
    args = p.parse_args()
    if not 10 <= args.session_minutes <= 2880:
        p.error('Session must be 10-2880 minutes including setup')
    if getpass.getuser() != 'lnthanh03' or REPO != ROOT / 'code/VideoQA-EvidenceLab-ridge-v1':
        p.error('Use the lnthanh03 school checkout')
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', 'src', 'scripts', 'configs', 'pyproject.toml', 'requirements*.txt'], cwd=REPO, capture_output=True, check=True)
    data = full.validate_inputs(SOURCE, BASELINE, DEVELOPMENT)
    expected = {'schema': full.SCHEMA, 'source_sha256': full.source_fingerprint(),
                **full.code_contract(), **data['inputs'], 'manifest': data['manifest'],
                'variants': list(full.VARIANTS), 'prototype': full.events.PROTOTYPE,
                'selected_count': 6685, 'small_grid_policy': 'all-candidates-if-count-le64'}
    if (OUTPUT / 'contract.json').exists():
        old = full.load(OUTPUT / 'contract.json')
        if (any(old.get(k) != v for k, v in expected.items())
                or any(old.get(k) != data['base'][k] for k in ('config', 'runtime', 'resolved_dtype'))):
            p.error('Full-run code/source/config changed; use a new run version')
    elif (OUTPUT / 'results.sqlite3').exists():
        p.error('Existing journal has no contract; investigate without deleting it')
    records, questions = progress(OUTPUT / 'results.sqlite3', set(data['manifest']['ids']))
    if records == 6685 and questions == 1337:
        print('Full1337 is complete; no job submitted.')
        return 0
    queue = subprocess.check_output(['squeue', '-h', '-u', 'lnthanh03', '-o', '%A|%j'], text=True)
    for line in queue.splitlines():
        job, name = line.strip().split('|', 1)
        if name in ('ridge-sm', 'ridge-cmp', 'ridge-post', 'ridge-query', 'ridge-event', 'ridge-evtfull'):
            p.error(f'RIDGE job {job} is active; wait before submission')
    nodes = set(subprocess.check_output(['sinfo', '-N', '-h', '-p', 'batch', '-o', '%N'], text=True).split())
    placement = submission_args(nodes, node=None if args.node == 'any' else args.node, memory_gib=90)
    placement = [x for x in placement if not x.startswith('--time=')]
    days, rest = divmod(args.session_minutes, 1440)
    hours, minutes = divmod(rest, 60)
    placement.append('--time=' + (f'{days}-' if days else '') + f'{hours:02d}:{minutes:02d}:00')
    env = os.environ.copy()
    env.update(PERSIST_ROOT=str(ROOT), REPO_ROOT=str(REPO), RUN_DIR=str(OUTPUT), BASELINE_RUN=str(BASELINE),
               SCORE_CACHE_RUN=str(SOURCE), EVENT_DEV_RUN=str(DEVELOPMENT), EXPERIMENT_MODE='full1337',
               CONFIG_FILE='configs/lvb_molmo2_ridge64_full.json', MAX_NEW_SAMPLES='0',
               EVENT_COVERAGE_SESSION_SECONDS=str(args.session_minutes * 60),
               RESUME=str(int((OUTPUT / 'results.sqlite3').exists())))
    record = {'mode': 'full1337', 'run_dir': str(OUTPUT), 'target_questions': 1337, 'target_variant_answers': 6685,
              'completed_questions': questions, 'completed_variant_answers': records, 'variants': list(full.VARIANTS),
              'gpu_count': 1, 'memory_gib': 90, 'node_mode': args.node, 'session_minutes': args.session_minutes,
              'eligible_nodes': sorted(nodes & (ALLOWED if args.node == 'any' else {args.node})),
              'sbatch_placement_args': placement, 'development_run_read_only': str(DEVELOPMENT),
              'question_score_cache_read_only': str(SOURCE), 'new_qwen_calls': 0,
              'answer_policy': 'Fresh inference or same-frame reuse inside this new run only.',
              'input_contract': expected}
    print(json.dumps(record, indent=2))
    if not (args.submit or args.test_only):
        return 0
    script = str(REPO / 'scripts/slurm/ridge_event_full_job.sh')
    (REPO / 'logs').mkdir(exist_ok=True)
    subprocess.run(placement + ['--test-only', '--chdir=' + str(REPO), script], env=env, check=True)
    if args.test_only:
        return 0
    if not env.get('HF_TOKEN'):
        env['HF_TOKEN'] = getpass.getpass('Hugging Face read token (hidden): ')
    if not env['HF_TOKEN']:
        p.error('HF_TOKEN is empty')
    reply = subprocess.check_output(placement + ['--parsable', '--chdir=' + str(REPO), script], env=env, text=True).strip()
    record['job_id'] = reply.split(';', 1)[0]
    (REPO / 'logs' / f'ridge-event-full-{record["job_id"]}.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'Submitted one full1337 job {record["job_id"]}; resume target 6685 records.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
