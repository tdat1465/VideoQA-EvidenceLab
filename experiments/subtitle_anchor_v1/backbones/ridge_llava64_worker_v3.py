"""Paired RIDGE/multi-anchor evaluation; resume only completed JSONL records."""
import argparse, hashlib, json, os, sys, tempfile, time
from pathlib import Path
import torch
from PIL import Image
from types import SimpleNamespace

ROOT = Path('/marimo/ridge-canonical')
RUN = ROOT / 'backbone-64-v1'
sys.path.insert(0, str(ROOT / 'relational-w5-v1'))
from worker_sigma2 import frames_full, load_full_inputs, MultipartReader, HTTPRangeSource, archive_index, fetch_video, check_space, TRANSFER_CHUNK_BYTES, decord, digest

parser = argparse.ArgumentParser()
parser.add_argument('--rank', type=int, required=True)
parser.add_argument('--pilot', action='store_true')
args = parser.parse_args()
assert 0 <= args.rank < 5
torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.197)
rows = json.loads((RUN / 'paired-manifest.json').read_text())
rows = [r for r in rows if int(hashlib.sha256(r['video'].encode()).hexdigest(),16) % 5 == args.rank]
if args.pilot: rows = rows[:1]
inputs = load_full_inputs(ROOT)
reader = MultipartReader(HTTPRangeSource())
members = archive_index(reader, ROOT / 'archive-index.json', {r['video'] for r in rows})
reader.clear_cache(); reader.block_size = TRANSFER_CHUNK_BYTES
sys.path.insert(0, str(ROOT / 'repo-fresh/src'))
from evidencelab.llava_backend import LlavaVideoBackend
from evidencelab.data import Question
model_path = '/dev/shm/ridge-backbone-hf-cache/models--lmms-lab--llava-onevision-qwen2-7b-ov/snapshots/0b07bf7565e244cf4f39982249eafe8cd799d6dd'
config=SimpleNamespace(backend='llava_video', dtype='bfloat16', seed=18, model_id=model_path, revision='0b07bf7565e244cf4f39982249eafe8cd799d6dd', vision_id='google/siglip-so400m-patch14-384', vision_revision='9fdffc58afc957d1a03a25b10dba0329ab15c2a3', max_input_tokens=20000)
backend=LlavaVideoBackend(config)
# Batch-independent vision encoding keeps all frames while limiting peak VRAM.
original_encode_images = backend.model.encode_images
def chunked_encode_images(images):
    return torch.cat([original_encode_images(chunk) for chunk in images.split(4)], dim=0)
backend.model.encode_images = chunked_encode_images
outdir = RUN / ('llava-pilot-v3' if args.pilot else 'llava-full')
outdir.mkdir(exist_ok=True)
output = outdir / f'worker-{args.rank}.jsonl'
done = {}
if output.exists():
    for line in output.read_text().splitlines():
        r = json.loads(line); key=(r['id'],r['arm']); assert key not in done; done[key]=r

def answer(row, imgs, times):
    result=backend.answer(Question(row['question_id'], row['question'], tuple(row['choices'])), imgs, times)
    return result.prediction, result.input_tokens

for row in rows:
    ident=row['question_id']; arms={'ridge_sigma2':row['frame_indices'],'multi_anchor':row['multi_frames']}
    if all((ident,a) in done for a in arms): continue
    with tempfile.TemporaryDirectory(dir='/dev/shm',prefix='ridge-qwen-') as tmp:
        path=Path(tmp)/'video.mp4'; sha=fetch_video(reader,members['videos/'+row['video']],path); reader.clear_cache()
        assert sha==inputs['baselines'][ident]['video_sha256']
        native=decord.VideoReader(str(path),ctx=decord.cpu(0),num_threads=1); count=len(native); fps=float(native.get_avg_fps()); del native
        backend.video_duration=count/fps
        cache=frames_full.build(path,Path(tmp)/'pixels',count,fps,set().union(*arms.values()),sha,check_space)
        refs=list((ROOT/'reference-full1337-v1').glob('shard-*/pixels/'+digest(row['video'])+'.json')); assert len(refs)==1
        ref=json.loads(refs[0].read_text()); assert all(ref['frames'][str(i)]==cache.manifest['frames'][str(i)] for i in row['frame_indices'])
        predictions={tuple(r['frames']):(r['prediction'],r['tokens']) for (qid,_),r in done.items() if qid==ident}
        for arm,frames in arms.items():
            if (ident,arm) in done: continue
            start=time.monotonic(); key=tuple(frames); reused=key in predictions
            if not reused: predictions[key]=answer(row,[Image.fromarray(cache[i].asnumpy()) for i in frames],[i/fps for i in frames])
            pred,tokens=predictions[key]; result={'id':ident,'arm':arm,'frames':frames,'prediction':pred,'correct':pred==row['answer'],'kind':row['question_type'],'tokens':tokens,'reused':reused,'seconds':time.monotonic()-start}
            with output.open('a',encoding='utf8') as f: f.write(json.dumps(result)+'\n'); f.flush(); os.fsync(f.fileno())
            done[(ident,arm)]=result
    print('DONE',ident,flush=True)
print('WORKER_COMPLETE',args.rank,flush=True)
