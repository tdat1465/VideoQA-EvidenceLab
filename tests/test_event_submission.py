import importlib.util,json,os,sqlite3,sys,tempfile,unittest
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'scripts/slurm'),str(REPO/'scripts')]
spec=importlib.util.spec_from_file_location('post_submit',REPO/'scripts/slurm/submit_ridge_event_coverage.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class PostSubmissionTests(unittest.TestCase):
 def test_read_only_progress_requires_both_variants(self):
  with tempfile.TemporaryDirectory() as temporary:
   path=Path(temporary)/'results.sqlite3';self.assertEqual(mod.progress(path),(set(),False))
   with sqlite3.connect(path) as db:
    db.execute('CREATE TABLE results (value TEXT)')
    db.executemany('INSERT INTO results VALUES (?)',[(json.dumps({'question_id':'q','variant':v}),) for v in mod.VARIANTS[:1]])
   db.close();before=path.read_bytes();self.assertEqual(mod.progress(path),(set(),False));self.assertEqual(before,path.read_bytes())
   with sqlite3.connect(path) as db:db.executemany('INSERT INTO results VALUES (?)',[(json.dumps({'question_id':'q','variant':v}),) for v in mod.VARIANTS[1:]])
   db.close();self.assertEqual(mod.progress(path),({'q'},False))

 def test_pilot_submits_one_eligible_gpu_job_and_blocks_duplicate(self):
  with tempfile.TemporaryDirectory() as temporary:
   root=Path(temporary);repo=root/'code/VideoQA-EvidenceLab-ridge-v1';repo.mkdir(parents=True)
   base=root/'baseline';base.mkdir();(base/'summary.json').write_text(json.dumps({'complete':True,'completed_count':1337}))
   rows=[{'id':f'q{i:03d}','video':f'v{i//2:03d}','ridge_candidate_count':100} for i in range(200)]
   (base/'predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
   queue=[''];calls=[]
   def output(command,**kwargs):
    calls.append(command)
    if command[0]=='squeue':return queue[0]
    if command[0]=='sinfo':return 'gpu01\ngpu02\ngpu03\ngpu04\n'
    if command[0]=='sbatch':
     self.assertEqual(kwargs['env']['EVENT_COVERAGE_SESSION_SECONDS'],str(session_minutes[0]*60))
     self.assertEqual(kwargs['env']['EXPERIMENT_MODE'],'pilot20');return '54321\n'
    raise AssertionError(command)
   session_minutes=[2880]
   patches=[patch.object(mod,'ROOT',root),patch.object(mod,'REPO',repo),patch.object(mod,'BASELINE',base),
    patch.object(mod,'SOURCE',root/'source'),patch.object(mod,'OUTPUT',root/'output'),
    patch.object(mod,'source_fingerprint',return_value=mod.BASE_SOURCE),
    patch.object(mod,'validate_source',return_value={'contract':{'manifest':{'ids':[r['id'] for r in rows]}},'inventory':{}}),
    patch.object(mod.getpass,'getuser',return_value='lnthanh03'),patch.object(mod.subprocess,'run'),
    patch.object(mod.subprocess,'check_output',side_effect=output),patch.dict(os.environ,{'HF_TOKEN':'test-only-secret'}),
    patch.object(sys,'argv',['submit','pilot20','--node','any','--submit'])]
   with ExitStack() as stack:
    for p in patches:stack.enter_context(p)
    self.assertEqual(mod.main(),0)
    jobs=[c for c in calls if c[0]=='sbatch'];self.assertEqual(len(jobs),1)
    for flag in ['--nodes=1','--gres=gpu:1','--mem=90G','--exclude=gpu04','--time=2-00:00:00']:self.assertIn(flag,jobs[0])
    self.assertEqual(sum(x.startswith('--time=') for x in jobs[0]),1)
    record=json.loads((repo/'logs/ridge-event-coverage-54321.json').read_text())
    self.assertEqual(record['mode_target_variant_answers'],100);self.assertEqual(record['expected_variant_answers'],1000)
    self.assertEqual(record['eligible_nodes'],['gpu01','gpu02','gpu03'])
    self.assertEqual(record['session_minutes'],2880)
    for duration,expected_time in [(60,'01:00:00'),(120,'02:00:00'),(1500,'1-01:00:00')]:
     calls.clear();session_minutes[0]=duration
     with patch.object(sys,'argv',['submit','pilot20','--session-minutes',str(duration),'--submit']):
      self.assertEqual(mod.main(),0)
     jobs=[c for c in calls if c[0]=='sbatch'];self.assertEqual(len(jobs),1)
     self.assertIn('--time='+expected_time,jobs[0])
     self.assertEqual(sum(x.startswith('--time=') for x in jobs[0]),1)
    calls.clear()
    with patch.object(sys,'argv',['submit','dev200','--submit']):
     with self.assertRaises(SystemExit):mod.main()
    self.assertFalse(any(c[0]=='sbatch' for c in calls))
    queue[0]='54321|ridge-event\n'
    with self.assertRaises(SystemExit):mod.main()
    self.assertFalse(any(c[0]=='sbatch' for c in calls))

 def test_rejects_invalid_total_wall_time_before_submission(self):
  for minutes in ('0','9','-1','2881'):
   with patch.object(sys,'argv',['submit','dev200','--session-minutes',minutes]),patch.object(mod.subprocess,'check_output') as output:
    with self.assertRaises(SystemExit):mod.main()
    output.assert_not_called()

if __name__=='__main__':unittest.main()
