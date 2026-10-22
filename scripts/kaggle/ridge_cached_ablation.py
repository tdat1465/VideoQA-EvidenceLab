"""Separate Kaggle T4x2 FP16 experiment. Never resume or mutate Slurm BF16 runs."""
from __future__ import annotations
import argparse
from contextlib import closing
from dataclasses import replace
import gc
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import sys
import time
import zipfile

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import ridge_component_ablation as components
import ridge_smoothing_ablation as smoothing
from evidencelab.data import digest,file_hash
from evidencelab.config import source_fingerprint
from evidencelab.store import Store,atomic_json,run_lock
from evidencelab.backends import HFBackend

VARIANTS=('sigma2','sigma1','multiscale_v1','vote_only','rescue_only')
load=components.load

def validate_device_map(mapping):
    devices={str(v).replace('cuda:','') for v in mapping.values()}
    if devices != {'0','1'}:
        raise RuntimeError(f'Require model weights on both CUDA GPUs with no CPU/disk offload; got {sorted(devices)}')

def kaggle_profile():
    import torch
    if not Path('/kaggle/working').is_dir() or not Path('/kaggle/input').is_dir():
        raise RuntimeError('This entry point is only for Kaggle notebooks')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=2:
        raise RuntimeError('Select Kaggle GPU T4 x2. Keep both GPUs visible.')
    names=[torch.cuda.get_device_name(i) for i in range(2)]
    if any('T4' not in n for n in names):
        raise RuntimeError(f'This profile was prepared for two T4 GPUs, got {names}')
    for i in range(2):
        x=torch.ones((8,8),device=f'cuda:{i}',dtype=torch.float16)
        if not torch.isfinite(x@x).all().item():raise RuntimeError('FP16 CUDA smoke failed')
        torch.cuda.synchronize(i)
    return {'gpu_names':names,'dtype':'float16','torch_cuda':torch.version.cuda,
            'capabilities':[list(torch.cuda.get_device_capability(i)) for i in range(2)],
            'placement':'balanced-max6GiB-per-GPU-sdpa-v1'}

class KaggleBackend(HFBackend):
    """Reuse exact prompt/video processor/letter scoring, change dtype and placement."""
    def __init__(self,config):
        import torch
        from transformers import AutoProcessor,AutoModelForImageTextToText
        self.profile=kaggle_profile()
        self.torch,self.config,self.dtype=torch,config,torch.float16
        torch.manual_seed(config.seed);torch.cuda.manual_seed_all(config.seed)
        self.processor=AutoProcessor.from_pretrained(config.model_id,revision=config.revision,
                                                    use_fast=False,trust_remote_code=True)
        self.model=AutoModelForImageTextToText.from_pretrained(config.model_id,revision=config.revision,
            trust_remote_code=True,torch_dtype=torch.float16,attn_implementation='sdpa',
            device_map='balanced',max_memory={0:'6GiB',1:'6GiB','cpu':'0GiB'},low_cpu_mem_usage=True)
        validate_device_map(self.model.hf_device_map)
        self.model.eval().requires_grad_(False)
        self.letter_ids=[]
        for label in 'ABCDE':
            ids=self.processor.tokenizer.encode(label,add_special_tokens=False)
            if len(ids)!=1:raise ValueError('Answer letter tokenizer changed')
            self.letter_ids.append(ids[0])
        self.profile['device_map']={k:str(v) for k,v in self.model.hf_device_map.items()}
    def reset_peak(self):
        for i in range(2):self.torch.cuda.synchronize(i);self.torch.cuda.reset_peak_memory_stats(i)
    def measurements(self):
        for i in range(2):self.torch.cuda.synchronize(i)
        return {'peak_vram_allocated_gib_per_gpu':[self.torch.cuda.max_memory_allocated(i)/1024**3 for i in range(2)],
                'peak_vram_reserved_gib_per_gpu':[self.torch.cuda.max_memory_reserved(i)/1024**3 for i in range(2)]}

def select(author,scores,weights,variant):
    fn=components.select_variant if variant in components.VARIANTS else smoothing.select_variant
    return fn(author,scores,weights,variant)

def make_contract(original,manifest,runtime,profile,inventory):
    return {'schema':'kaggle-ridge-fp16-ablation-v1','platform':'kaggle-T4x2',
        'config':{**original['config'],'dtype':'float16'},'resolved_dtype':'float16','hardware':profile,
        'runtime':runtime,'source_sha256':source_fingerprint(),'runner_sha256':file_hash(Path(__file__)),
        'component_script_sha256':file_hash(REPO/'scripts/ridge_component_ablation.py'),
        'smoothing_script_sha256':file_hash(REPO/'scripts/ridge_smoothing_ablation.py'),
        'source_contract_digest':digest(original),'score_inventory':inventory,
        'manifest':manifest,'variants':list(VARIANTS),'selected_count':len(manifest['ids'])*len(VARIANTS),
        'historical_score_and_qwen_reuse':True,'server_predictions_are_not_resumed':True}

def evaluate_one(sample,score,weight,author,store,curves,answer,stop=lambda:False):
    done=store.done();qid=sample.id;diagnostics={'question_id':qid,'score_sha256':score['sha256'],'variants':{}}
    if curves.exists():
        diagnostics=load(curves)
        if diagnostics['score_sha256']!=score['sha256']:raise ValueError('Curve/cache mismatch')
    for variant in VARIANTS:
        key=qid+'::'+variant
        if key in done:continue
        if stop():raise InterruptedError('Stop between committed variants')
        start=time.perf_counter();positions,details=select(author,score['scores'],weight['weights'],variant)
        selection=time.perf_counter()-start
        indices=[score['key']['candidate_indices'][i] for i in positions]
        times=[i/score['key']['fps'] for i in indices]
        if len(indices)!=64 or indices!=sorted(set(indices)):raise ValueError('Invalid 64-frame selection')
        diagnostics['variants'][variant]={**details,'selected_positions':positions,'frame_indices':indices,'timestamps':times}
        atomic_json(curves,diagnostics)
        decision,seconds,measurements=answer(indices,times)
        probabilities=decision.scores
        if (len(probabilities)!=len(sample.choices) or any(not math.isfinite(v) or not 0<=v<=1 for v in probabilities)
                or abs(sum(probabilities)-1)>1e-4):raise ValueError('Nonfinite or invalid FP16 answer; no precision fallback')
        store.put({'id':key,'question_id':qid,'variant':variant,'video':sample.video,'answer':sample.answer,
            'prediction':decision.prediction,'correct':decision.prediction==sample.answer,
            'question_type':sample.question_type,'duration_group':sample.duration_group,
            'frame_indices':indices,'timestamps':times,'final_frames':64,'scores':probabilities,
            'input_tokens':decision.input_tokens,'score_sha256':score['sha256'],'weight_sha256':weight['record_sha256'],
            'selector_only_seconds':selection,'answer_decode_seconds':seconds,'resolved_dtype':'float16',**measurements})
        print(f'Committed {qid}: {variant}, correct={decision.prediction==sample.answer}',flush=True)

def export(output,contract,store):
    records=store.results();stats={}
    for variant in VARIANTS:
        values=[r for r in records if r['variant']==variant];n=len(values);correct=sum(r['correct'] for r in values)
        stats[variant]={'n':n,'correct':correct,'accuracy':correct/n if n else None}
    result={'complete':len(records)==contract['selected_count'],'completed_variant_answers':len(records),
        'target_variant_answers':contract['selected_count'],'variants':stats,'new_blip_calls':0,'new_qwen_calls':0,
        'interpretation':'Separate FP16 T4x2 development experiment. Compare variants within this run; do not merge with BF16 server results.'}
    atomic_json(output/'summary.json',result)
    with (output/'predictions.jsonl').open('w',encoding='utf-8',newline='\n') as f:
        for record in records:f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
    atomic_json(output/'manifest.json',contract['manifest'])
    return result

def checkpoint(output,store,target):
    """SQLite backup is transaction-consistent; archive contains no model/token/cache."""
    backup=output/'checkpoint-snapshot.sqlite3'
    try:
        with closing(sqlite3.connect(backup)) as db:store.db.backup(db)
        temporary=target.with_suffix('.tmp.zip')
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(output.rglob('*')):
                if p.is_file() and p.name not in {'results.sqlite3','checkpoint-snapshot.sqlite3','run.lock'} and not p.name.endswith(('.tmp','-journal')):
                    z.write(p,'run/'+p.relative_to(output).as_posix())
            z.write(backup,'run/results.sqlite3')
            index=output.parent/'metadata'/'index.json'
            if index.is_file():z.write(index,'metadata/index.json')
        os.replace(temporary,target)
    finally:backup.unlink(missing_ok=True)

def validate_paths(output,scratch):
    for path,root in [(output,Path('/kaggle/working')),(scratch,Path('/kaggle/temp'))]:
        if root.resolve() not in path.resolve().parents:raise ValueError(f'Use a dedicated directory inside {root}')

def run(args):
    from evidencelab.longvideo_config import LongVideoConfig
    from evidencelab.longvideo import longvideo_runtime
    from evidencelab.longvideo_data import fetch_annotations,parse_annotations,HTTPRangeSource,MultipartReader,archive_index,fetch_video,TRANSFER_CHUNK_BYTES
    from evidencelab.ridge_adapter import load_ridge
    from evidencelab.ridge_weights import RidgeWeightCache,weight_contract
    from evidencelab.aks_adapter import aks_candidates
    from PIL import Image
    import decord
    output,scratch=args.output.resolve(),args.scratch.resolve();validate_paths(output,scratch)
    output.mkdir(parents=True,exist_ok=True);scratch.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(scratch).free<15*1024**3:raise RuntimeError('Need at least 15 GiB free scratch disk before model download')
    bundle=args.bundle.resolve();base,source=bundle/'baseline',bundle/'source'
    cached=components.validate_source(source,base);original=cached['contract']
    config=replace(LongVideoConfig(**original['config']),dtype='float16')
    annotations=fetch_annotations(scratch)
    baseline_contract=load(base/'contract.json')
    if file_hash(annotations)!=baseline_contract['annotation_sha256']:raise ValueError('Annotation revision/hash changed')
    samples=parse_annotations(load(annotations));selected=sorted((q for q in samples if q.id in original['manifest']['ids']),key=lambda q:(q.video,q.id))
    if len(selected)!=200:raise ValueError('Manifest must contain the frozen 200 questions')
    author,weight_author=load_ridge(bundle/'upstream')
    # Validate historical preferences with their original BF16 provenance, not FP16 answer config.
    weight_cache=RidgeWeightCache(base/'ridge-weights.json',weight_contract(LongVideoConfig(**original['config']),baseline_contract,weight_author),[q.public_question() for q in samples])
    profile=kaggle_profile();started=time.monotonic();stopped=[];handlers={}
    for name in ('SIGINT','SIGTERM','SIGUSR1'):
        n=getattr(signal,name,None)
        if n is not None:handlers[n]=signal.signal(n,lambda number,frame:stopped.append(number))
    stop=lambda:bool(stopped) or time.monotonic()-started>=args.max_seconds
    video=None;reader=None;active=scratch/'active-video.mp4';target=output.parent/'ridge-kaggle-checkpoint.zip'
    try:
        backend=KaggleBackend(config)
        profile=backend.profile
        contract=make_contract(original,original['manifest'],longvideo_runtime(config),profile,cached['inventory'])
        with run_lock(output):
            store=Store(output,contract,(output/'results.sqlite3').exists())
            try:
                (output/'curves').mkdir(exist_ok=True)
                store.event({'event':'session_start','profile':profile,'mode':args.mode})
                pending=[q for q in selected if any(q.id+'::'+v not in store.done() for v in VARIANTS)]
                complete_questions=sum(all(q.id+'::'+v in store.done() for v in VARIANTS) for q in selected)
                quota=max(0,2-complete_questions) if args.mode=='pilot' else len(pending)
                if quota:
                    source_reader=HTTPRangeSource(stop=stop);reader=MultipartReader(source_reader)
                    members=archive_index(reader,output.parent/'metadata'/'index.json',{q.video for q in pending})
                    reader.clear_cache();reader.block_size=TRANSFER_CHUNK_BYTES
                current=None;processed=0
                for sample in pending:
                    if stop() or processed>=quota:break
                    if sample.video!=current:
                        video=None;gc.collect();active.unlink(missing_ok=True)
                        size=members['videos/'+sample.video][1]
                        if size>8*1024**3 or shutil.disk_usage(scratch).free<size+4*1024**3:
                            raise MemoryError('Insufficient Kaggle scratch disk for bounded active video')
                        sha=fetch_video(reader,members['videos/'+sample.video],active,stop=stop,guard=False)
                        reader.clear_cache();video=decord.VideoReader(str(active),ctx=decord.cpu(0),num_threads=2);current=sample.video
                    fps=float(video.get_avg_fps())
                    if len(video)==0 or not math.isfinite(fps) or fps<=0:raise ValueError('Invalid video FPS/length')
                    key=components.cache_key(original,sample,sha,aks_candidates(len(video),fps),fps,len(video))
                    score=components.read_score_cache(source/'scores'/(digest(sample.id)+'.json'),key)
                    if score['sha256']!=cached['inventory'][sample.id]:raise ValueError('Score changed after preflight')
                    def answer(indices,times):
                        backend.reset_peak();begin=time.perf_counter()
                        images=[Image.fromarray(video[i].asnumpy()).convert('RGB') for i in indices]
                        decision=backend.answer(sample.public_question(),images,times)
                        return decision,time.perf_counter()-begin,backend.measurements()
                    evaluate_one(sample,score,weight_cache.records[sample.id],author,store,output/'curves'/(digest(sample.id)+'.json'),answer,stop)
                    processed+=1;summary=export(output,contract,store)
                    print(f"Progress {summary['completed_variant_answers']}/{summary['target_variant_answers']}",flush=True)
                    if processed%5==0:checkpoint(output,store,target)
            except InterruptedError:
                store.event({'event':'controlled_stop','signals':stopped})
            finally:
                video=None;gc.collect();active.unlink(missing_ok=True)
                if reader is not None:reader.clear_cache()
                try:
                    summary=export(output,contract,store)
                    checkpoint(output,store,target)
                    print(json.dumps(summary,ensure_ascii=False),flush=True)
                finally:store.close()
            return 0 if summary['complete'] else 75
    finally:
        for n,handler in handlers.items():signal.signal(n,handler)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scratch',type=Path,default=Path('/kaggle/temp/ridge-v1'))
    p.add_argument('--mode',choices=['pilot','full'],default='pilot');p.add_argument('--max-seconds',type=int,default=21600)
    a=p.parse_args()
    if a.max_seconds<60:p.error('Allow at least 60 seconds per session')
    raise SystemExit(run(a))
