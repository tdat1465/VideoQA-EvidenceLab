import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'scripts'))
import ridge_post_rescue_ablation as mod
from evidencelab.store import Store
from evidencelab.pipeline import Decision

WEIGHTS=dict(zip(('peak_similarity','slope_abs','rising_slope','falling_slope','boundary_change','context_density'),(8,6,5,7,4,3)))

class PostRescueTests(unittest.TestCase):
 def synthetic(self):
  base=list(range(28))+[48,54,56,58,60]+list(range(100,126))+[176,180,182,184,186]
  base=sorted(base);self.assertEqual(len(base),64)
  scores=[.1]*256
  for p in [50,178]:scores[p-1:p+2]=[.8,.9,.8]
  detail={'labels':['background']*256,'curve':{},'allocation':{'background':64},'rescued_peaks':[]}
  detector={**detail,'rescued_peaks':[50,178],'scale_analyses':[{'curve':{'s_smooth':scores}}]}
  def select(author,raw,weights,variant,k):return (base,detail) if variant=='sigma2' else (base,detector)
  return base,scores,select

 def test_caps_nested_swaps_fixed_bins_and_no_input_mutation(self):
  base,scores,select=self.synthetic();before=list(scores)
  with patch.object(mod.frozen,'select_variant',side_effect=select):
   first,a=mod.select_variant(None,scores,WEIGHTS,'post_rescue1')
   second,b=mod.select_variant(None,scores,WEIGHTS,'post_rescue2')
  self.assertEqual(a['swap_count'],1);self.assertEqual(b['swap_count'],2)
  self.assertEqual(a['swaps'],b['swaps'][:1]);self.assertEqual(before,scores)
  for selected,d in [(first,a),(second,b)]:
   self.assertEqual(len(selected),64);self.assertEqual(selected,sorted(set(selected)))
   self.assertEqual(len(set(selected)-set(base)),d['swap_count'])
   self.assertTrue({base[0],base[-1]}<=set(selected))
   for swap in d['swaps']:
    self.assertEqual(swap['added']*16//256,swap['removed']*16//256)
    self.assertLessEqual(abs(swap['added']-swap['removed']),12)

 def test_no_local_donor_means_exact_control(self):
  base,scores,select=self.synthetic()
  with patch.object(mod.frozen,'select_variant',side_effect=select),patch.dict(mod.PROTOTYPE,{'donor_radius_candidates':1}):
   positions,details=mod.select_variant(None,scores,WEIGHTS,'post_rescue2')
  self.assertEqual(positions,base);self.assertEqual(details['swap_count'],0)

 def test_pilot_is_label_blind_and_uses_whole_videos(self):
  rows=[{'id':f'q{i:03d}','video':f'v{i//2:03d}','ridge_candidate_count':100,'answer':i%4,'correct':True} for i in range(200)]
  manifest={'ids':[r['id'] for r in rows]}
  pilot=mod.pilot_manifest(rows,manifest)
  changed=[{**r,'answer':(r['answer']+1)%4,'correct':False,'prediction':999} for r in rows]
  self.assertEqual(pilot,mod.pilot_manifest(changed,manifest));self.assertEqual(len(pilot['ids']),20)
  for video in pilot['videos']:
   self.assertTrue({r['id'] for r in rows if r['video']==video}<=set(pilot['ids']))

 def test_interrupt_resume_and_no_source_writes(self):
  base,scores,select=self.synthetic()
  sample=type('Sample',(),dict(id='q',video='v.mp4',question='Demo?',choices=('a','b'),answer=1,question_type='TAA',duration_group='600'))()
  score={'scores':scores,'sha256':'source-score'};weight={'weights':WEIGHTS,'record_sha256':'weight','generation_seconds':1.}
  calls=[]
  def answer(indices,times):
   calls.append(indices);return Decision([.2,.8],100),.1,{}
  with tempfile.TemporaryDirectory() as temp,patch.object(mod.frozen,'select_variant',side_effect=select):
   root=Path(temp);source=root/'source';source.mkdir();(source/'predictions.jsonl').write_text('')
   contract={'manifest':{'ids':['q']},'pilot20_manifest':{'ids':['q']},'selected_count':3,
             'score_source':str(source),'reference_variants':{}}
   control_bytes=(source/'predictions.jsonl').read_bytes()
   output=root/'output';output.mkdir();(output/'curves').mkdir();store=Store(output,contract,False)
   with self.assertRaises(InterruptedError):
    mod.evaluate_question(sample,list(range(256)),1.,weight,score,None,store,output/'curves/q.json',answer,stop=lambda:len(calls)>=1)
   self.assertEqual(len(store.results()),1);store.close()
   store=Store(output,contract,True)
   mod.evaluate_question(sample,list(range(256)),1.,weight,score,None,store,output/'curves/q.json',answer)
   self.assertEqual(len(calls),3)
   result=mod.export(output,contract,store);store.close()
   self.assertTrue(result['complete']);self.assertTrue(result['pilot20']['complete'])
   self.assertEqual(result['one_time_scoring_seconds'],0);self.assertEqual(score['scores'],scores)
   self.assertEqual(control_bytes,(source/'predictions.jsonl').read_bytes())

if __name__=='__main__':unittest.main()
