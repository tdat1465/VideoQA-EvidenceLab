import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import ExitStack
import numpy as np
from evidencelab.data import digest,file_hash
from evidencelab.pipeline import Decision
from evidencelab.ridge_adapter import load_ridge
from evidencelab.store import atomic_json

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO.parents[1]
spec=importlib.util.spec_from_file_location('components',REPO/'scripts/ridge_component_ablation.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
SOURCE=ROOT/'outputs/ridge-smoothing-results-20261021-232214/lvb-ridge-smoothing-v1-dev200'
BASE=ROOT/'outputs/ridge-results-20261021-075851/lvb-val-molmo2-ridge64-full-ridge-v1'
WEIGHTS=dict(zip(('peak_similarity','slope_abs','rising_slope','falling_slope','boundary_change','context_density'),(8,6,5,7,4,3)))

class ComponentTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.author,cls.weight_author=load_ridge(Path(os.environ['RIDGE_UPSTREAM_DIR']))

 @unittest.skipUnless(SOURCE.is_dir() and BASE.is_dir(),'Requires downloaded dev200 and baseline exports')
 def test_real_cache_preflight_and_broken_cache_rejected(self):
  data=mod.validate_source(SOURCE,BASE)
  self.assertEqual(len(data['inventory']),200)
  with tempfile.TemporaryDirectory() as temp:
   copy=Path(temp)/'cache';copy.mkdir()
   for name in ['summary.json','contract.json','manifest.json','predictions.jsonl']:shutil.copy2(SOURCE/name,copy/name)
   shutil.copytree(SOURCE/'scores',copy/'scores')
   file=next((copy/'scores').glob('*.json'));record=mod.load(file)
   record['scores'][0]=.123456
   atomic_json(file,record)
   with self.assertRaises(ValueError):mod.validate_source(copy,BASE)
   file.unlink()
   with self.assertRaises(FileNotFoundError):mod.validate_source(copy,BASE)

 @unittest.skipUnless(SOURCE.is_dir() and BASE.is_dir(),'Requires downloaded dev200 and baseline exports')
 def test_components_isolate_changes_on_all_200_real_curves(self):
  weights=mod.load(BASE/'ridge-weights.json')['records'];counts={'questions':0,'no_rescue_controls':0}
  for path in (SOURCE/'curves').glob('*.json'):
   c=mod.load(path);qid=c['question_id'];s=mod.load(SOURCE/'scores'/path.name)
   vote,vd=mod.select_variant(self.author,s['scores'],weights[qid]['weights'],'vote_only')
   rescue,rd=mod.select_variant(self.author,s['scores'],weights[qid]['weights'],'rescue_only')
   counts['questions']+=1
   self.assertEqual(vd['rescued_peaks'],[])
   self.assertEqual(rd['rescued_peaks'],c['variants']['multiscale_v1']['rescued_peaks'])
   for chosen in [vote,rescue]:self.assertEqual(chosen,sorted(set(chosen)));self.assertEqual(len(chosen),64)
   if not rd['rescued_peaks']:
    counts['no_rescue_controls']+=1
    self.assertEqual(rescue,c['variants']['sigma2']['selected_positions'])
    self.assertEqual(vote,c['variants']['multiscale_v1']['selected_positions'])
   else:
    self.assertTrue(all(label==c['variants']['sigma2']['labels'][i] for i,label in enumerate(rd['labels'])
                        if all(abs(i-p)>1 for p in rd['rescued_peaks'])))
  self.assertEqual(counts,{'questions':200,'no_rescue_controls':97})

 def test_synthetic_supported_shoulder_vote_has_no_rescue(self):
  scores=.01+np.exp(-((np.arange(300)-150)/25)**2/2);scores[130:132]+=.4
  _,vote=mod.select_variant(self.author,scores,WEIGHTS,'vote_only')
  _,rescue=mod.select_variant(self.author,scores,WEIGHTS,'rescue_only')
  self.assertEqual(vote['rescued_peaks'],[]);self.assertIn(131,rescue['rescued_peaks'])
  with self.assertRaises(ValueError):mod.select_variant(self.author,[.1]*63,WEIGHTS,'vote_only')

 def test_mock_workflow_resume_reads_source_without_scorer_or_source_writes(self):
  from evidencelab.longvideo_config import LongVideoConfig
  from evidencelab.longvideo_data import parse_annotations
  from evidencelab.ridge_weights import question_hash,weight_contract
  runtime={'fake_runtime':'test'};execution={'resolved_dtype':'bfloat16','gpu':'fake','node':'fake'}
  counts={'models':0,'answers':0}
  class Video:
   def __len__(self):return 90
   def get_avg_fps(self):return 1.
   def __getitem__(self,i):return SimpleNamespace(asnumpy=lambda:np.zeros((16,16,3),dtype=np.uint8))
  class Backend:
   def __init__(self,c):counts['models']+=1
   def reset_peak(self):pass
   def measurements(self):return {}
   def answer(self,q,frames,times):counts['answers']+=1;return Decision([.2,.8],100)
  class Reader:
   def __init__(self,s):pass
   def clear_cache(self):pass
  def fetch(reader,entry,path,stop):path.write_bytes(b'fake');return 'a'*64
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);base=root/'base';source=root/'source';output=root/'output';ram=root/'ram'
   for p in [base,source,ram,source/'scores']:p.mkdir()
   annotations=[{'id':str(i),'video_path':f'v{i//2}.mp4','question':f'question {i}','candidates':['a','b'],
                 'correct_choice':1,'question_category':'TAA','duration_group':'600','duration':90.} for i in range(1337)]
   atomic_json(root/'annotations.json',annotations);samples=parse_annotations(annotations)
   config=LongVideoConfig.load(REPO/'configs/lvb_molmo2_ridge64_full.json')
   bc={'config':config.contract(),'source_sha256':mod.BASE_SOURCE,'resolved_dtype':'bfloat16','runtime':runtime,
       'selected_ids_hash':digest(sorted(q.id for q in samples)),'annotation_sha256':file_hash(root/'annotations.json')}
   atomic_json(base/'contract.json',bc);atomic_json(base/'summary.json',{'complete':True,'completed_count':1337})
   positions=self.author.gradselect([i/90 for i in range(90)],64,WEIGHTS,mod.args_for('sigma2'))
   records={};rows=[]
   for q in samples:
    rec={'question_id':q.id,'question_sha256':question_hash(q.public_question()),'weights':WEIGHTS,
         'input_tokens':32,'output_tokens':64,'generation_seconds':1.,'response_sha256':'f'*64}
    rec['record_sha256']=digest(rec);records[q.id]=rec
    rows.append({'id':q.id,'video':q.video,'ridge_candidate_count':90,'ridge_weight_record_sha256':rec['record_sha256'],
                 'video_sha256':'a'*64,'calls':[{'frame_indices':positions}],'prediction':1})
   (base/'predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
   atomic_json(base/'ridge-weights.json',{'contract':weight_contract(config,bc,self.weight_author),'records':records})
   manifest=mod.choose_development(rows);sc={'manifest':manifest,'schema':'fake-score-source'};inventory={}
   atomic_json(source/'contract.json',sc);(source/'predictions.jsonl').write_text('fake reference')
   for q in samples:
    if q.id not in manifest['ids']:continue
    key=mod.cache_key(sc,q,'a'*64,list(range(90)),1.,90)
    score=mod.score_record(key,[i/90 for i in range(90)],3.,23);inventory[q.id]=score['sha256']
    atomic_json(source/'scores'/(digest(q.id)+'.json'),score)
   cached={'contract':sc,'inventory':inventory,'summary':{'variants':{'sigma2':{'n':200,'correct':200}}}}
   before={str(p):file_hash(p) for folder in [base,source] for p in folder.rglob('*') if p.is_file()}
   args=SimpleNamespace(config=REPO/'configs/lvb_molmo2_ridge64_full.json',annotations=root/'annotations.json',
                        baseline=base,source_cache=source,output=output,ram_root=ram,index=root/'index.json',
                        max_new_questions=2,max_seconds=60)
   patches=[patch.dict('sys.modules',{'decord':SimpleNamespace(VideoReader=lambda *a,**k:Video(),cpu=lambda x:None)}),
            patch.object(mod,'validate_source',return_value=cached),patch('evidencelab.longvideo.validate_ram_root'),
            patch('evidencelab.hardware.execution_profile',return_value=execution),
            patch('evidencelab.longvideo.longvideo_runtime',return_value=runtime),patch('evidencelab.backends.HFBackend',Backend),
            patch('evidencelab.focus_adapter.BlipITMScorer',side_effect=AssertionError('Must not load scorer')),
            patch('evidencelab.ridge_adapter.fetch_ridge',return_value=Path(os.environ['RIDGE_UPSTREAM_DIR'])),
            patch('evidencelab.longvideo_data.HTTPRangeSource',return_value=SimpleNamespace()),
            patch('evidencelab.longvideo_data.MultipartReader',Reader),
            patch('evidencelab.longvideo_data.archive_index',return_value={'videos/'+q.video:[0,1] for q in samples}),
            patch('evidencelab.longvideo_data.fetch_video',side_effect=fetch),patch('evidencelab.longvideo_data.check_space')]
   with ExitStack() as stack:
    for p in patches:stack.enter_context(p)
    self.assertEqual(mod.run(args),75)
    first=(output/'predictions.jsonl').read_text().splitlines()
    self.assertEqual(len(first),4)
    self.assertEqual(mod.run(args),75)
   stats=mod.load(output/'summary.json')
   self.assertEqual(stats['completed_variant_answers'],8)
   self.assertEqual(stats['one_time_scored_questions'],0)
   self.assertEqual(stats['one_time_scoring_seconds'],0)
   self.assertEqual(counts,{'models':2,'answers':8})
   self.assertFalse((output/'scores').exists())
   self.assertTrue(set(first)<=set((output/'predictions.jsonl').read_text().splitlines()))
   self.assertEqual(before,{str(p):file_hash(p) for folder in [base,source] for p in folder.rglob('*') if p.is_file()})

if __name__=='__main__':unittest.main()
