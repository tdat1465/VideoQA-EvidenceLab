import importlib.util
from pathlib import Path
import json,os,tempfile,unittest,zipfile,gc
from unittest.mock import patch
import numpy as np
from evidencelab.data import digest
from evidencelab.pipeline import Decision
from evidencelab.store import Store
from evidencelab.ridge_adapter import load_ridge

REPO=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('kaggle_ridge',REPO/'scripts/kaggle/ridge_cached_ablation.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
WEIGHTS=dict(zip(('peak_similarity','slope_abs','rising_slope','falling_slope','boundary_change','context_density'),(8,6,5,7,4,3)))

class KaggleRidgeTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.author,_=load_ridge(Path(os.environ['RIDGE_UPSTREAM_DIR']))

 def test_sharding_requires_both_gpus_and_no_offload(self):
  mod.validate_device_map({'vision':0,'decoder':'cuda:1'})
  for invalid in [{'all':0},{'a':0,'b':1,'c':'cpu'},{'a':0,'b':'disk'}]:
   with self.assertRaises(RuntimeError):mod.validate_device_map(invalid)

 def test_disk_frames_preserve_pixels_order_and_release_decoder_before_processor(self):
  alive=[]
  class Decoder:
   def __init__(self,path):alive.append(True)
   def __getitem__(self,index):
    a=np.full((7,11,3),index,np.uint8)
    return type('Frame',(),{'asnumpy':lambda self:a})()
   def __del__(self):alive.pop()
  indices=list(range(0,128,2))
  with tempfile.TemporaryDirectory() as temp,patch.object(mod,'memory_phase',return_value={}):
   root=Path(temp);target=root/'frames.bin'
   with mod.mapped_rgb_frames(root/'video.mp4',indices,target,Decoder) as frames:
    self.assertEqual(alive,[])
    self.assertIsInstance(frames,np.memmap)
    self.assertEqual(frames.shape,(64,7,11,3))
    self.assertEqual(frames.dtype,np.uint8)
    np.testing.assert_array_equal(frames[:,0,0,0],indices)
    self.assertTrue(target.exists())
   self.assertFalse(target.exists())

 def test_disk_frames_cleanup_after_interruption(self):
  class Decoder:
   def __init__(self,path):pass
   def __getitem__(self,index):return type('Frame',(),{'asnumpy':lambda self:np.zeros((2,3,3),np.uint8)})()
  with tempfile.TemporaryDirectory() as temp,patch.object(mod,'memory_phase',return_value={}):
   root=Path(temp);target=root/'frames.bin'
   with self.assertRaises(InterruptedError):
    with mod.mapped_rgb_frames(root/'v.mp4',[0,1],target,Decoder,stop=lambda:True):pass
   self.assertFalse(target.exists())

 def test_memory_limits_use_cgroup_without_slurm_and_stop_before_low_headroom(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);group=root/'cgroup';group.mkdir();proc=root/'proc/self';proc.mkdir(parents=True)
   (group/'memory.current').write_text(str(12*1024**3));(group/'memory.max').write_text(str(13*1024**3))
   (proc/'cgroup').write_text('0::/\n');(proc/'status').write_text('VmRSS:\t4096 kB\n')
   info=mod.memory_snapshot(group,root/'proc')
   self.assertEqual(info['cgroup_headroom_bytes'],1024**3)
   self.assertEqual(info['worker_rss_bytes'],4096*1024)
   with patch.object(mod,'memory_snapshot',return_value=info):
    with self.assertRaises(MemoryError):mod.memory_phase('decode_frame')

 def test_mapped_processor_input_avoids_native_stack_and_preserves_metadata(self):
  from evidencelab.backends import prompt,molmo_metadata
  class Processor:
   def apply_chat_template(self,messages,**kwargs):
    self.messages=messages;return 'rendered'
   def __call__(self,**kwargs):self.kwargs=kwargs;return {'input_ids':[[1,2]]}
  backend=object.__new__(mod.KaggleBackend);backend.processor=Processor()
  question=type('Public',(),dict(question='Which action?',choices=['a','b']))()
  times=[i*1.25 for i in range(64)]
  with tempfile.TemporaryDirectory() as temp,patch.object(mod,'memory_phase',return_value={}):
   frames=np.memmap(Path(temp)/'frames',dtype=np.uint8,mode='w+',shape=(64,2,3,3))
   try:
    backend._inputs(question,frames,times)
    self.assertIs(backend.processor.kwargs['videos'][0],frames)
    self.assertEqual(backend.processor.kwargs['video_metadata'],[molmo_metadata(times)])
    self.assertFalse(backend.processor.kwargs['do_sample_frames'])
    self.assertEqual(backend.processor.messages[0]['content'][1]['text'],prompt(question))
   finally:frames._mmap.close()

 def test_platform_paths_are_separate_from_school_server(self):
  mod.validate_paths(Path('/kaggle/working/ridge/run'),Path('/kaggle/temp/ridge'))
  for output,scratch in [(Path('/media/lnthanh03/DatHa/runs/x'),Path('/kaggle/temp/x')),
                         (Path('/kaggle/working/x'),Path('/dev/shm/x'))]:
   with self.assertRaises(ValueError):mod.validate_paths(output,scratch)

 def test_fp16_contract_preserves_original_config_and_rejects_bf16_journal(self):
  old={'config':{'dtype':'auto','frames':64},'resolved_dtype':'bfloat16'}
  manifest={'ids':['q']}
  c=mod.make_contract(old,manifest,{'python':'3.12'},{'gpu_names':['T4','T4']},{'q':'hash'})
  self.assertEqual(old['config']['dtype'],'auto')
  self.assertEqual(c['config']['dtype'],'float16');self.assertEqual(c['selected_count'],5)
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);store=Store(root,old,False);store.close()
   with self.assertRaises(ValueError):Store(root,c,True)

 def test_mid_variant_checkpoint_restores_exactly_and_resumes_remaining(self):
  sample=type('Sample',(),dict(id='q',video='v.mp4',answer=1,choices=('a','b'),question_type='TAA',duration_group='600'))()
  score={'scores':np.random.default_rng(18).random(150).tolist(),'key':{'candidate_indices':list(range(150)),'fps':1.}}
  score['sha256']=digest(score);weight={'weights':WEIGHTS,'record_sha256':'w'};calls=[]
  def answer(indices,times):
   calls.append(indices);return Decision([.2,.8],100),.1,{'peak_vram_allocated_gib_per_gpu':[1.,1.]}
  contract={'schema':'kaggle-ridge-fp16-ablation-v1','manifest':{'ids':['q']},'selected_count':5}
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);output=root/'run';output.mkdir();(output/'curves').mkdir()
   store=Store(output,contract,False)
   with self.assertRaises(InterruptedError):
    mod.evaluate_one(sample,score,weight,self.author,store,output/'curves/q.json',answer,stop=lambda:len(calls)>=2)
   self.assertEqual(len(store.results()),2)
   self.assertFalse(mod.export(output,contract,store)['complete'])
   mod.checkpoint(output,store,root/'checkpoint.zip');store.close()
   restored=root/'restored';restored.mkdir()
   with zipfile.ZipFile(root/'checkpoint.zip') as z:
    self.assertFalse(any('hf' in n or 'token' in n or n.endswith('run.lock') for n in z.namelist()))
    z.extractall(restored)
   output=restored/'run';store=Store(output,contract,True)
   mod.evaluate_one(sample,score,weight,self.author,store,output/'curves/q.json',answer)
   self.assertEqual(len(calls),5)
   stats=mod.export(output,contract,store)
   self.assertTrue(stats['complete']);self.assertEqual(stats['completed_variant_answers'],5)
   self.assertEqual(stats['new_blip_calls'],0);self.assertEqual(stats['new_qwen_calls'],0)
   self.assertTrue(all(v['n']==1 and v['correct']==1 for v in stats['variants'].values()))
   self.assertTrue(all(r['final_frames']==64 and r['resolved_dtype']=='float16' for r in store.results()))
   store.close()

 def test_nonfinite_fp16_answer_is_not_committed(self):
  sample=type('Sample',(),dict(id='q',video='v',answer=0,choices=('a','b'),question_type='TAA',duration_group='600'))()
  score={'scores':[.5]*90,'key':{'candidate_indices':list(range(90)),'fps':1.},'sha256':'s'}
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);store=Store(root,{'schema':'test'},False)
   try:
    with self.assertRaises(ValueError):
     mod.evaluate_one(sample,score,{'weights':WEIGHTS,'record_sha256':'w'},self.author,store,root/'q.json',
                      lambda *a:(Decision([float('nan'),.5],100),.1,{}))
    self.assertEqual(store.results(),[])
   finally:store.close()

if __name__=='__main__':unittest.main()
