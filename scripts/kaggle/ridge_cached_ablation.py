"""Separate Kaggle T4x2 FP16 experiment. Never resume or mutate Slurm BF16 runs."""
from __future__ import annotations
import argparse
from contextlib import closing,contextmanager
from dataclasses import replace
import gc
import json
import math
import mmap
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
from evidencelab.backends import HFBackend,prompt,molmo_metadata

VARIANTS=('sigma2','sigma1','multiscale_v1','vote_only','rescue_only')
load=components.load
VIDEO_IO_PROFILE='rgb64-disk-map-original-pixels-decoder-released-v2'

def memory_snapshot(cgroup=Path('/sys/fs/cgroup'),proc=Path('/proc')):
    """Read container limits independently of Slurm; never use host RAM as a job limit."""
    result={};groups={cgroup,cgroup/'memory'}
    try:
        for line in (proc/'self/cgroup').read_text().splitlines():
            _,controllers,relative=line.split(':',2)
            if not controllers or 'memory' in controllers.split(','):
                root=cgroup if not controllers else cgroup/'memory'
                leaf=root/relative.lstrip('/')
                if root==leaf or root in leaf.parents:
                    groups.add(leaf)
                    groups.update(p for p in leaf.parents if p==root or root in p.parents)
    except (OSError,ValueError):pass
    limits=[]
    for group in groups:
        for current,maximum in [('memory.current','memory.max'),('memory.usage_in_bytes','memory.limit_in_bytes')]:
            try:
                used=int((group/current).read_text());limit=int((group/maximum).read_text())
                if 0<limit<2**60:limits.append((limit-used,used,limit,str(group)))
            except (OSError,ValueError):pass
    if limits:
        free,used,limit,group=min(limits)
        result.update(cgroup_current_bytes=used,cgroup_limit_bytes=limit,cgroup_headroom_bytes=free,cgroup_path=group)
    try:
        for line in (proc/'self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):result['worker_rss_bytes']=int(line.split()[1])*1024
    except (OSError,ValueError):pass
    return result

def memory_phase(phase,**details):
    info=memory_snapshot()
    print(json.dumps({'event':'memory_phase','phase':phase,**info,**details}),flush=True)
    if info.get('cgroup_headroom_bytes',2**63)<1280*1024**2:
        raise MemoryError(f'Container RAM headroom below 1.25 GiB before {phase}; stopping to save results')
    return info

def release_cpu():
    gc.collect()
    if sys.platform.startswith('linux'):
        import ctypes
        try:ctypes.CDLL(None).malloc_trim(0)
        except (AttributeError,OSError):pass

def scratch_filesystem(path,mountinfo=Path('/proc/self/mountinfo')):
    matches=[]
    try:
        for line in mountinfo.read_text().splitlines():
            left,right=line.split(' - ',1);mount=Path(left.split()[4])
            if path==mount or mount in path.parents:matches.append((len(mount.parts),right.split()[0]))
    except (OSError,ValueError,IndexError):pass
    return max(matches)[1] if matches else None

@contextmanager
def mapped_rgb_frames(path,indices,destination,reader_factory,stop=lambda:False):
    """Decode one RGB frame at a time; native pixels live on disk, not a PIL list."""
    import numpy as np
    decoder=None;frames=None;frame=None
    destination.unlink(missing_ok=True)
    try:
        memory_phase('decode_start',frames=len(indices))
        decoder=reader_factory(path)
        shape=None
        with destination.open('xb',buffering=0) as output:
            for position,index in enumerate(indices):
                if stop():raise InterruptedError('Stop during frame staging')
                memory_phase('decode_frame',position=position,index=index)
                frame=decoder[index].asnumpy()
                if frame.dtype!=np.uint8 or frame.ndim!=3 or frame.shape[-1]!=3 or not frame.flags.c_contiguous:
                    raise ValueError('Decoder must supply contiguous original RGB uint8 frames')
                if shape is None:
                    shape=(len(indices),*frame.shape)
                    needed=math.prod(shape)
                    if shutil.disk_usage(destination.parent).free<needed+4*1024**3:
                        raise MemoryError('Not enough scratch disk for original-resolution selected frames')
                    print(json.dumps({'event':'rgb_disk_staging','shape':shape,'bytes':needed}),flush=True)
                elif frame.shape!=shape[1:]:raise ValueError('Video frame shape changed')
                view=memoryview(frame).cast('B');offset=0
                while offset<len(view):
                    written=output.write(view[offset:])
                    if not written:raise OSError('Short RGB write')
                    offset+=written
                del view;frame=None
                if hasattr(os,'posix_fadvise'):
                    try:
                        os.fdatasync(output.fileno())
                        os.posix_fadvise(output.fileno(),0,0,os.POSIX_FADV_DONTNEED)
                    except OSError:
                        if position==0:print('Scratch filesystem does not support cache advice',flush=True)
        if shape is None:raise ValueError('No selected frames')
        decoder=None;release_cpu()  # Release native decoder buffers before processor allocations.
        memory_phase('decode_done',shape=shape)
        frames=np.memmap(destination,mode='r+',dtype=np.uint8,shape=shape)
        yield frames
    finally:
        frame=None;decoder=None
        if frames is not None:frames._mmap.close()
        destination.unlink(missing_ok=True)
        release_cpu()

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
        memory_phase('model_load_start')
        self.model=AutoModelForImageTextToText.from_pretrained(config.model_id,revision=config.revision,
            trust_remote_code=True,torch_dtype=torch.float16,attn_implementation='sdpa',
            device_map='balanced',max_memory={0:'6GiB',1:'6GiB','cpu':'0GiB'},low_cpu_mem_usage=True)
        validate_device_map(self.model.hf_device_map)
        self.model.eval().requires_grad_(False)
        release_cpu();memory_phase('model_load_done')
        self.letter_ids=[]
        for label in 'ABCDE':
            ids=self.processor.tokenizer.encode(label,add_special_tokens=False)
            if len(ids)!=1:raise ValueError('Answer letter tokenizer changed')
            self.letter_ids.append(ids[0])
        self.profile['device_map']={k:str(v) for k,v in self.model.hf_device_map.items()}
        self.profile['video_io_profile']=VIDEO_IO_PROFILE
    def _inputs(self,question,frames,times):
        import numpy as np
        if not isinstance(frames,np.memmap) or frames.ndim!=4 or frames.shape[0]!=64:
            raise ValueError('Kaggle v2 requires 64 original RGB frames in a disk mapping')
        messages=[{'role':'user','content':[{'type':'video','video':'predecoded'},
                                           {'type':'text','text':prompt(question)}]}]
        rendered=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        memory_phase('processor_start',rgb_shape=list(frames.shape))
        inputs=self.processor(text=[rendered],videos=[frames],video_metadata=[molmo_metadata(times)],
                              do_sample_frames=False,return_tensors='pt')
        # Upstream resizes each original frame identically; this path avoids np.stack/PIL copies.
        if hasattr(frames._mmap,'madvise'):
            try:frames._mmap.madvise(mmap.MADV_DONTNEED)
            except OSError:pass
        memory_phase('processor_done')
        return inputs
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
    return {'schema':'kaggle-ridge-fp16-ablation-v2','platform':'kaggle-T4x2',
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
    import decord
    output,scratch=args.output.resolve(),args.scratch.resolve();validate_paths(output,scratch)
    output.mkdir(parents=True,exist_ok=True);scratch.mkdir(parents=True,exist_ok=True)
    filesystem=scratch_filesystem(scratch)
    if filesystem in {'tmpfs','ramfs'}:raise MemoryError('Kaggle v2 RGB staging requires disk scratch, not tmpfs/RAM')
    memory_phase('worker_start',scratch_filesystem=filesystem,video_io_profile=VIDEO_IO_PROFILE)
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
                export(output,contract,store);checkpoint(output,store,target)
                pending=[q for q in selected if any(q.id+'::'+v not in store.done() for v in VARIANTS)]
                complete_questions=sum(all(q.id+'::'+v in store.done() for v in VARIANTS) for q in selected)
                quota=max(0,2-complete_questions) if args.mode=='pilot' else len(pending)
                if quota:
                    memory_phase('archive_index_start')
                    source_reader=HTTPRangeSource(stop=stop);reader=MultipartReader(source_reader)
                    members=archive_index(reader,output.parent/'metadata'/'index.json',{q.video for q in pending})
                    reader.clear_cache();reader.block_size=TRANSFER_CHUNK_BYTES
                    memory_phase('archive_index_done')
                current=None;processed=0
                for sample in pending:
                    if stop() or processed>=quota:break
                    if sample.video!=current:
                        video=None;release_cpu();active.unlink(missing_ok=True)
                        size=members['videos/'+sample.video][1]
                        if size>8*1024**3 or shutil.disk_usage(scratch).free<size+4*1024**3:
                            raise MemoryError('Insufficient Kaggle scratch disk for bounded active video')
                        memory_phase('video_download_start',video=sample.video,compressed_bytes=size)
                        sha=fetch_video(reader,members['videos/'+sample.video],active,stop=stop,guard=False)
                        reader.clear_cache();memory_phase('video_metadata_start')
                        video=decord.VideoReader(str(active),ctx=decord.cpu(0),num_threads=1)
                        fps=float(video.get_avg_fps());frame_count=len(video)
                        video=None;release_cpu();current=sample.video
                        memory_phase('video_metadata_done',fps=fps,total_frames=frame_count)
                    if frame_count==0 or not math.isfinite(fps) or fps<=0:raise ValueError('Invalid video FPS/length')
                    key=components.cache_key(original,sample,sha,aks_candidates(frame_count,fps),fps,frame_count)
                    score=components.read_score_cache(source/'scores'/(digest(sample.id)+'.json'),key)
                    if score['sha256']!=cached['inventory'][sample.id]:raise ValueError('Score changed after preflight')
                    def answer(indices,times):
                        backend.reset_peak();begin=time.perf_counter()
                        factory=lambda p:decord.VideoReader(str(p),ctx=decord.cpu(0),num_threads=1)
                        with mapped_rgb_frames(active,indices,scratch/'selected-rgb.bin',factory,stop) as frames:
                            memory_phase('answer_start')
                            decision=backend.answer(sample.public_question(),frames,times)
                        memory_phase('answer_done')
                        return decision,time.perf_counter()-begin,backend.measurements()
                    evaluate_one(sample,score,weight_cache.records[sample.id],author,store,output/'curves'/(digest(sample.id)+'.json'),answer,stop)
                    processed+=1;summary=export(output,contract,store)
                    print(f"Progress {summary['completed_variant_answers']}/{summary['target_variant_answers']}",flush=True)
                    checkpoint(output,store,target)
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
