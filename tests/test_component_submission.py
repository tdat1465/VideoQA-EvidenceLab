import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'scripts/slurm'))
spec=importlib.util.spec_from_file_location('component_submit',REPO/'scripts/slurm/submit_ridge_components.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class ComponentSubmissionTests(unittest.TestCase):
 def test_progress_reads_400_answer_journal_without_mutation(self):
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'results.sqlite3'
   self.assertEqual(module.progress(path),(0,False))
   with sqlite3.connect(path) as db:
    db.execute('create table results (value text)')
    db.executemany('insert into results values (?)',[(json.dumps({'question_id':f'q{i}','variant':v}),) for i in range(200) for v in ['vote_only','rescue_only']])
   db.close()
   before=path.read_bytes()
   self.assertEqual(module.progress(path),(200,True))
   self.assertEqual(before,path.read_bytes())

 def test_submission_is_one_gpu_job_and_blocks_active_component_job(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);repo=root/'code/VideoQA-EvidenceLab-ridge-v1';repo.mkdir(parents=True)
   base=root/'baseline';base.mkdir();(base/'summary.json').write_text(json.dumps({'complete':True,'completed_count':1337}))
   calls=[];queue=['']
   def output(cmd,**kw):
    calls.append(cmd)
    if cmd[0]=='squeue':return queue[0]
    if cmd[0]=='sinfo':return 'gpu01\ngpu02\ngpu03\ngpu04\n'
    if cmd[0]=='sbatch':return '12345\n'
    raise AssertionError(cmd)
   patches=[patch.object(module,'ROOT',root),patch.object(module,'REPO',repo),patch.object(module,'BASELINE',base),
            patch.object(module,'SOURCE',root/'source'),patch.object(module,'OUTPUT',root/'output'),
            patch.object(module,'source_fingerprint',return_value=module.BASE_SOURCE),
            patch.object(module,'validate_source',return_value={'inventory':{}}),
            patch.object(module.getpass,'getuser',return_value='lnthanh03'),
            patch.object(module.subprocess,'check_output',side_effect=output),
            patch.object(module.subprocess,'run'),patch.dict(os.environ,{'HF_TOKEN':'fake-test-token'}),
            patch.object(sys,'argv',['submit','dev200','--node','any','--submit'])]
   with ExitStack() as stack:
    for p in patches:stack.enter_context(p)
    self.assertEqual(module.main(),0)
    submitted=[c for c in calls if c[0]=='sbatch']
    self.assertEqual(len(submitted),1)
    self.assertIn('--nodes=1',submitted[0]);self.assertIn('--gres=gpu:1',submitted[0]);self.assertIn('--exclude=gpu04',submitted[0])
    record=json.loads((repo/'logs/ridge-components-12345.json').read_text())
    self.assertEqual(record['expected_variant_answers'],400)
    self.assertEqual(record['eligible_nodes'],['gpu01','gpu02','gpu03'])
    calls.clear();queue[0]='12345|ridge-cmp\n'
    with self.assertRaises(SystemExit):module.main()
    self.assertFalse(any(c[0]=='sbatch' for c in calls))

if __name__=='__main__':unittest.main()
