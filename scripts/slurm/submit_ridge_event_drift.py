"""One isolated diagnostic job for o2F-N42Ufo4_2; leaves the full-run journal untouched."""
import argparse
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys

from submit import ALLOWED, submission_args
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'scripts'), str(REPO / 'src')]
import diagnose_ridge_event_drift as diagnostic

ROOT = Path('/media/lnthanh03/DatHa')
BASELINE = ROOT / 'runs/videoqa/lvb-val-molmo2-ridge64-full-ridge-v1'
SOURCE = ROOT / 'runs/videoqa/lvb-ridge-smoothing-v1-dev200'
DEVELOPMENT = ROOT / 'runs/videoqa/lvb-ridge-event-coverage-v1-dev200'
FULL_RUN = ROOT / 'runs/videoqa/lvb-ridge-event-coverage-full1337-v1'
OUTPUT = ROOT / 'runs/videoqa/lvb-ridge-event-drift-o2f-v1'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--node', choices=['any'] + sorted(ALLOWED), default='gpu03')
    p.add_argument('--session-minutes', type=int, default=240)
    action = p.add_mutually_exclusive_group()
    action.add_argument('--submit', action='store_true')
    action.add_argument('--test-only', action='store_true')
    args = p.parse_args()
    if not 10 <= args.session_minutes <= 480:
        p.error('Diagnostic session must be 10-480 minutes including setup')
    if getpass.getuser() != 'lnthanh03' or REPO != ROOT / 'code/VideoQA-EvidenceLab-ridge-v1':
        p.error('Use the lnthanh03 school checkout')
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', 'src', 'scripts', 'configs', 'pyproject.toml', 'requirements*.txt'], cwd=REPO, capture_output=True, check=True)
    context = diagnostic.validate_inputs(SOURCE, BASELINE, DEVELOPMENT, FULL_RUN)
    queue = subprocess.check_output(['squeue', '-h', '-u', 'lnthanh03', '-o', '%A|%j'], text=True)
    for line in queue.splitlines():
        job, name = line.strip().split('|', 1)
        if name in ('ridge-sm', 'ridge-cmp', 'ridge-post', 'ridge-query', 'ridge-event', 'ridge-evtfull', 'ridge-evtdiag'):
            p.error(f'RIDGE job {job} is active; wait before the diagnostic')
    nodes = set(subprocess.check_output(['sinfo', '-N', '-h', '-p', 'batch', '-o', '%N'], text=True).split())
    placement = submission_args(nodes, node=None if args.node == 'any' else args.node, memory_gib=90)
    placement = [x for x in placement if not x.startswith('--time=')]
    hours, minutes = divmod(args.session_minutes, 60)
    placement.append(f'--time={hours:02d}:{minutes:02d}:00')
    env = os.environ.copy()
    env.update(PERSIST_ROOT=str(ROOT), REPO_ROOT=str(REPO), RUN_DIR=str(OUTPUT), BASELINE_RUN=str(BASELINE),
               SCORE_CACHE_RUN=str(SOURCE), EVENT_DEV_RUN=str(DEVELOPMENT), FULL_RUN=str(FULL_RUN),
               EXPERIMENT_MODE='diagnose', CONFIG_FILE='configs/lvb_molmo2_ridge64_full.json', MAX_NEW_SAMPLES='0',
               EVENT_COVERAGE_SESSION_SECONDS=str(args.session_minutes * 60), RESUME='0')
    record = {'purpose': 'isolated-control-drift-diagnostic', 'target': diagnostic.TARGET,
              'cases': list(diagnostic.CASES), 'output_root': str(OUTPUT), 'full_run_read_only': str(FULL_RUN),
              'snapshot': context['snapshot'], 'gpu_count': 1, 'memory_gib': 90, 'node_mode': args.node,
              'session_minutes': args.session_minutes, 'sbatch_placement_args': placement,
              'worker_sha256': diagnostic.file_hash(Path(diagnostic.__file__))}
    print(json.dumps(record, indent=2))
    if not (args.submit or args.test_only):
        return 0
    script = str(REPO / 'scripts/slurm/ridge_event_drift_job.sh')
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
    (REPO / 'logs' / f'ridge-event-drift-{record["job_id"]}.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'Submitted diagnostic job {record["job_id"]}; target {diagnostic.TARGET}; no full-run evaluation writes.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
