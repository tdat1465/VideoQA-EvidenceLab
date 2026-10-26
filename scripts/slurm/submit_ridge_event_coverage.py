"""One BF16 GPU job; event-query scoring with frozen question weights and 64 frames."""
import argparse
from contextlib import closing
import getpass
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

from submit import ALLOWED,submission_args
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
from evidencelab.config import source_fingerprint
from evidencelab.data import file_hash
from ridge_event_coverage_ablation import BASE_SOURCE,VARIANTS,PROTOTYPE,validate_source,pilot_manifest,helper_hash,QUERY_HELPER_SHA

ROOT=Path('/media/lnthanh03/DatHa')
BASELINE=ROOT/'runs/videoqa/lvb-val-molmo2-ridge64-full-ridge-v1'
SOURCE=ROOT/'runs/videoqa/lvb-ridge-smoothing-v1-dev200'
OUTPUT=ROOT/'runs/videoqa/lvb-ridge-event-coverage-v1-dev200'

def progress(path):
    if not path.exists():return set(),False
    with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)) as db:
        rows=[json.loads(x) for x, in db.execute('SELECT value FROM results')]
    groups={}
    for row in rows:
        if row['variant'] not in VARIANTS:raise ValueError('Foreign result variant')
        groups.setdefault(row['question_id'],set()).add(row['variant'])
    complete_ids={q for q,v in groups.items() if v==set(VARIANTS)}
    return complete_ids,len(rows)==1000 and len(complete_ids)==200

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['pilot20','dev200'])
    p.add_argument('--node',choices=['any']+sorted(ALLOWED),default='any')
    p.add_argument('--accept-partial-pilot',action='store_true',
        help='For dev200 only: accept a reviewed pilot with at least 18 of 20 complete questions')
    p.add_argument('--session-minutes',type=int,default=2880,
        help='Total wall time including setup, 10-2880 minutes; independent of the GPU idle policy')
    action=p.add_mutually_exclusive_group();action.add_argument('--submit',action='store_true');action.add_argument('--test-only',action='store_true')
    args=p.parse_args()
    if args.accept_partial_pilot and args.mode!='dev200':p.error('--accept-partial-pilot is only valid with dev200')
    if not 10<=args.session_minutes<=2880:p.error('--session-minutes must be 10-2880 (up to 48 hours total wall time)')
    if getpass.getuser()!='lnthanh03' or REPO!=ROOT/'code/VideoQA-EvidenceLab-ridge-v1':p.error('Use lnthanh03 school checkout')
    if source_fingerprint()!=BASE_SOURCE:p.error('Frozen baseline source changed')
    subprocess.run(['git','diff','--exit-code','HEAD','--','src','scripts','configs','pyproject.toml','requirements*.txt'],cwd=REPO,check=True,capture_output=True)
    summary=json.loads((BASELINE/'summary.json').read_text())
    if not summary['complete'] or summary['completed_count']!=1337:p.error('Require completed RIDGE baseline')
    cached=validate_source(SOURCE,BASELINE)
    baseline_rows=[json.loads(line) for line in (BASELINE/'predictions.jsonl').read_text().splitlines() if line.strip()]
    pilot=pilot_manifest(baseline_rows,cached['contract']['manifest'])
    if (OUTPUT/'contract.json').exists():
        old=json.loads((OUTPUT/'contract.json').read_text())
        expected={'schema':'ridge-event-coverage-ablation-v1','experiment_script_sha256':file_hash(REPO/'scripts/ridge_event_coverage_ablation.py'),
            'event_helper_sha256':helper_hash(),'query_helper_sha256':QUERY_HELPER_SHA,
            'baseline_weights_sha256':file_hash(BASELINE/'ridge-weights.json'),'baseline_predictions_sha256':file_hash(BASELINE/'predictions.jsonl'),
            'score_source_contract_sha256':file_hash(SOURCE/'contract.json'),'score_source_predictions_sha256':file_hash(SOURCE/'predictions.jsonl'),
            'score_inventory':cached['inventory'],'pilot20_manifest':pilot,'prototype':PROTOTYPE}
        if any(old.get(k)!=v for k,v in expected.items()):p.error('Run/source/code changed; do not mix experiments')
    completed,complete=progress(OUTPUT/'results.sqlite3')
    pilot_completed=len(set(pilot['ids'])&completed)
    pilot_complete=set(pilot['ids'])<=completed
    partial_pilot_accepted=args.accept_partial_pilot and not pilot_complete and pilot_completed>=18
    if args.mode=='dev200' and not (pilot_complete or partial_pilot_accepted):
        p.error('Complete pilot20 before submitting dev200, or use --accept-partial-pilot after reviewing at least 18 complete pilot questions')
    scope=set(pilot['ids'] if args.mode=='pilot20' else cached['contract']['manifest']['ids'])
    if complete or scope<=completed:
        print('This mode is complete; no job submitted.');return 0
    queue=subprocess.check_output(['squeue','-h','-u','lnthanh03','-o','%A|%j'],text=True)
    for line in queue.splitlines():
        job,name=line.strip().split('|',1)
        if name in ('ridge-sm','ridge-cmp','ridge-post','ridge-query','ridge-event'):p.error(f'RIDGE job {job} is active; wait before submission')
    nodes=set(subprocess.check_output(['sinfo','-N','-h','-p','batch','-o','%N'],text=True).split())
    placement=submission_args(nodes,node=None if args.node=='any' else args.node,memory_gib=90)
    placement=[item for item in placement if not item.startswith('--time=')]
    days,remainder=divmod(args.session_minutes,1440)
    hours,minutes=divmod(remainder,60)
    wall_time=f'{hours:02d}:{minutes:02d}:00'
    placement.append('--time='+(f'{days}-' if days else '')+wall_time)
    eligible=sorted(nodes&(ALLOWED if args.node=='any' else {args.node}))
    env=os.environ.copy()
    env.update(PERSIST_ROOT=str(ROOT),REPO_ROOT=str(REPO),RUN_DIR=str(OUTPUT),BASELINE_RUN=str(BASELINE),SCORE_CACHE_RUN=str(SOURCE),
        CONFIG_FILE='configs/lvb_molmo2_ridge64_full.json',EXPERIMENT_MODE=args.mode,MAX_NEW_SAMPLES='0',
        EVENT_COVERAGE_SESSION_SECONDS=str(args.session_minutes*60),
        RESUME=str(int((OUTPUT/'results.sqlite3').exists())))
    record={'mode':args.mode,'run_dir':str(OUTPUT),'source_cache_read_only':str(SOURCE),'variants':list(VARIANTS),
        'expected_variant_answers':1000,'mode_target_variant_answers':len(scope)*len(VARIANTS),'pilot20_manifest':pilot,
        'completed_mode_questions':len(scope&completed),'gpu_count':1,'memory_gib':90,'node_mode':args.node,
        'pilot20_completed_questions':pilot_completed,'pilot20_complete':pilot_complete,
        'partial_pilot_accepted':partial_pilot_accepted,
        'eligible_nodes':eligible,'sbatch_placement_args':placement,'new_blip_scoring':'deduplicated event descriptions for public-text sequence questions','new_qwen_calls':0}
    record.update(session_minutes=args.session_minutes,checkpoint_cleanup_margin_seconds=300,
        scheduling_only_change=False,new_separate_worker=True)
    print(json.dumps(record,indent=2))
    if not (args.submit or args.test_only):return 0
    script=str(REPO/'scripts/slurm/ridge_event_coverage_job.sh');(REPO/'logs').mkdir(exist_ok=True)
    subprocess.run(placement+['--test-only','--chdir='+str(REPO),script],env=env,check=True)
    if args.test_only:return 0
    if not env.get('HF_TOKEN'):env['HF_TOKEN']=getpass.getpass('Hugging Face read token (hidden): ')
    if not env['HF_TOKEN']:p.error('HF_TOKEN is empty')
    reply=subprocess.check_output(placement+['--parsable','--chdir='+str(REPO),script],env=env,text=True).strip()
    job=reply.split(';',1)[0];record['job_id']=job
    (REPO/'logs'/f'ridge-event-coverage-{job}.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f'Submitted one GPU job {job}; expected mode records: {len(scope)*len(VARIANTS)}');return 0

if __name__=='__main__':raise SystemExit(main())
