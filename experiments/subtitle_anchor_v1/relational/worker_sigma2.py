"""Independent Molab pilot; frozen selectors and text weights, fresh inference."""
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import argparse
sys.path.insert(0,str(Path(__file__).resolve().parent))
from parallel_common_full import partitions, export, gpu_fraction, load_full_inputs
import frames_full

ROOT = Path('/marimo/ridge-canonical')
REPO = ROOT / 'repo-fresh'
sys.path[:0] = [str(REPO / 'scripts'), str(REPO / 'src')]
import ridge_event_canonical as w
from evidencelab.data import Question, digest, file_hash
from evidencelab.store import Store, atomic_json, run_lock
from evidencelab.longvideo_config import LongVideoConfig
from evidencelab.hardware import execution_profile
from evidencelab.longvideo import longvideo_runtime
from evidencelab.longvideo_data import HTTPRangeSource, MultipartReader, archive_index, fetch_video, check_space, TRANSFER_CHUNK_BYTES
from evidencelab.ridge_adapter import fetch_ridge, load_ridge
from evidencelab.aks_adapter import aks_candidates
from evidencelab.backends import HFBackend
from evidencelab.focus_adapter import BlipITMScorer
from PIL import Image
import decord
import sigma2_adapter
sigma2_adapter.configure(w)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--rank',type=int,required=True)
    p.add_argument('--workers',type=int,default=5,choices=[5])
    p.add_argument('--pilot',action='store_true')
    args=p.parse_args()
    inputs = load_full_inputs(ROOT)
    shard_ids=partitions(inputs,args.workers)[args.rank]
    if args.pilot:shard_ids=shard_ids[:1]
    suffix='-pilot' if args.pilot else ''
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(gpu_fraction(args.workers))
    torch.cuda.empty_cache()
    actual = subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    if actual != inputs['source_commit']:
        raise ValueError('Molab source revision changed')
    config = LongVideoConfig(**inputs['config'])
    execution = execution_profile(config)
    contract = {'schema':'ridge-canonical-relational-5worker-v1', 'phase':'pilot' if args.pilot else 'full', 'inputs_sha256':file_hash(ROOT/'inputs-full1337.json'),
                'source_commit':actual, 'worker_sha256':file_hash(Path(__file__)),
                'parallel_helper_sha256':file_hash(Path(__file__).with_name('parallel_common_full.py')), 'code':{**w.code_contract(), **sigma2_adapter.code_contract()}, 'config':config.contract(), 'runtime':longvideo_runtime(config),
                'resolved_dtype':execution['resolved_dtype'], 'variants':list(w.VARIANTS), 'selected_count':len(shard_ids)*len(w.VARIANTS), 'parallel':{'rank':args.rank,'workers':args.workers,'memory_fraction':gpu_fraction(args.workers),'png_cache_budget_gib':12,'assignment':'whole-video-greedy-candidate-cost-v1'},
                'manifest':{'ids':shard_ids,'pilot_ids':shard_ids,
                            'engineering_videos':inputs['engineering_videos'],
                            'evaluation_note':'Post-hoc sigma2 ablation on previously inspected validation; report engineering videos separately. New Python/CUDA/GPU runtime; fresh BLIP curves and answers. No prior answers copied.'},
                'pixel_policy':frames_full.POLICY, 'frame_helper_sha256':file_hash(Path(frames_full.__file__))}
    output=ROOT/('results-relational-w5-v1'+suffix)/f'shard-{args.rank}'
    ram=Path('/dev/shm')/f'ridge-canonical-relational-w5-v1-{args.rank}{suffix}'
    ram.mkdir(exist_ok=True)
    if subprocess.check_output(['findmnt','-n','-o','FSTYPE','--target',str(ram)],text=True).strip() != 'tmpfs':
        raise RuntimeError('Video and pixel cache require tmpfs')
    stopped=[]
    for name in ('SIGINT','SIGTERM'):
        signal.signal(getattr(signal,name),lambda s,f:stopped.append(s))
    stop=lambda:bool(stopped)
    samples=[]
    for r in inputs['samples']:
        if r['question_id'] not in shard_ids:continue
        q=Question(r['question_id'],r['question'],tuple(r['choices']))
        samples.append(SimpleNamespace(**{k:v for k,v in r.items() if k!='question_id'},id=q.id,public_question=lambda q=q:q))
    samples.sort(key=lambda q:(q.video,q.id))
    with run_lock(output):
        store=Store(output,contract,(output/'results.sqlite3').exists())
        for name in ('scores','curves','pixels'):
            (output/name).mkdir(exist_ok=True)
        video=None
        reader=None
        cache_root=ram/'active-pixels'
        video_path=ram/'active-video.mp4'
        start=time.monotonic()
        try:
            done=store.done()
            pending=[q for q in samples if any(q.id+'::'+v not in done for v in w.VARIANTS)]
            if not pending:
                return
            print('PHASE archive index',flush=True)
            reader=MultipartReader(HTTPRangeSource(stop=stop))
            members=archive_index(reader,ROOT/'archive-index.json',{q.video for q in pending})
            reader.clear_cache();reader.block_size=TRANSFER_CHUNK_BYTES
            print('ASSIGNMENT',args.rank,len(shard_ids),'questions',flush=True)
            print('PHASE model loading',flush=True)
            author,_=load_ridge(fetch_ridge(ram/'ridge-upstream'))
            if args.rank:time.sleep(10)
            backend=HFBackend(config)
            scorer=BlipITMScorer(config)
            current=None
            for sample in pending:
                if stop():raise InterruptedError('Stop requested')
                baseline=inputs['baselines'][sample.id]
                if current!=sample.video:
                    video=None;gc.collect()
                    if cache_root.exists():shutil.rmtree(cache_root)
                    video_path.unlink(missing_ok=True)
                    print('PHASE fetch video',sample.video,flush=True)
                    sha=fetch_video(reader,members['videos/'+sample.video],video_path,stop=stop)
                    reader.clear_cache()
                    if sha!=baseline['video_sha256']:raise ValueError('Source video changed')
                    native=decord.VideoReader(str(video_path),ctx=decord.cpu(0),num_threads=1)
                    count,fps=len(native),float(native.get_avg_fps());del native;gc.collect()
                    candidates=aks_candidates(count,fps)
                    print('PHASE canonical double decode',sample.video,flush=True)
                    video=frames_full.build(video_path,cache_root,count,fps,sorted(set(candidates)|set(w.uniform(count,64))),sha,check_space,stop)
                    path=output/'pixels'/(digest(sample.video)+'.json')
                    if path.exists() and w.load(path)!=video.manifest:raise ValueError('Resume pixels changed')
                    references=list((ROOT/'reference-full1337-v1').glob('shard-*/pixels/'+path.name))
                    if len(references)!=1 or w.load(references[0])!=video.manifest:raise ValueError('Pixels differ from completed canonical full')
                    atomic_json(path,video.manifest);current=sample.video
                if len(candidates)!=baseline['ridge_candidate_count']:raise ValueError('Candidate grid changed')
                path=output/'scores'/(digest(sample.id)+'.json')
                key=w.score_key(contract,sample,video,candidates)
                print('PHASE score',sample.id,flush=True)
                score=w.full.validate_score(w.load(path),key) if path.exists() else w.make_score(key,sample,video,scorer,config.blip_batch_size,stop)
                atomic_json(path,score)
                def answer(indices,times):
                    backend.reset_peak();begin=time.perf_counter();check_space(ram,0)
                    frames=[Image.fromarray(video[i].asnumpy()) for i in indices]
                    backend.video_duration=len(video)/fps
                    decision=backend.answer(sample.public_question(),frames,times)
                    return decision,time.perf_counter()-begin,{**backend.measurements(),'execution':execution}
                w.evaluate(sample,score,inputs['weights'][sample.id],author,store,output/'curves'/(digest(sample.id)+'.json'),answer,baseline,stop)
                export(output,contract,store)
        except InterruptedError:
            store.event({'event':'controlled_stop','signals':stopped})
        finally:
            video=None;gc.collect()
            if reader is not None:reader.clear_cache()
            if cache_root.exists():shutil.rmtree(cache_root)
            video_path.unlink(missing_ok=True)
            store.event({'event':'session_end','wall_seconds':time.monotonic()-start})
            print(json.dumps(export(output,contract,store)),flush=True)
            store.close()


if __name__=='__main__':
    main()
