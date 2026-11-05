"""Paired RIDGE/multi-anchor evaluation; resume only completed JSONL records."""
import argparse, hashlib, json, os, sys, tempfile, time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

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
torch.cuda.set_per_process_memory_fraction(0.19)
rows = json.loads((RUN / 'paired-manifest.json').read_text())
rows = [r for r in rows if int(hashlib.sha256(r['video'].encode()).hexdigest(),16) % 5 == args.rank]
if args.pilot: rows = rows[:1]
inputs = load_full_inputs(ROOT)
reader = MultipartReader(HTTPRangeSource())
members = archive_index(reader, ROOT / 'archive-index.json', {r['video'] for r in rows})
reader.clear_cache(); reader.block_size = TRANSFER_CHUNK_BYTES
model_path = json.loads((RUN / 'models.jsonl').read_text().splitlines()[0])['path']
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'':0}, local_files_only=True).eval()
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False)
tokenizer = processor.tokenizer
letters = [tokenizer.encode(c, add_special_tokens=False) for c in 'ABCDE']
assert all(len(x)==1 for x in letters)
letter_ids = [x[0] for x in letters]
outdir = RUN / ('qwen-pilot' if args.pilot else 'qwen-full')
outdir.mkdir(exist_ok=True)
output = outdir / f'worker-{args.rank}.jsonl'
done = {}
if output.exists():
    for line in output.read_text().splitlines():
        r = json.loads(line); key=(r['id'],r['arm']); assert key not in done; done[key]=r

def answer(row, imgs, times):
    content=[]
    for img,t in zip(imgs,times):
        content += [{'type':'text','text':f'Frame at {t:.3f}s:'},{'type':'image','image':img}]
    choices='\n'.join(f'{chr(65+i)}. {x}' for i,x in enumerate(row['choices']))
    text='Answer the multiple-choice question using the video evidence. Timestamps are seconds in the source video. Select exactly one option letter.\nQuestion: '+row['question']+'\n'+choices+'\nAnswer:'
    content.append({'type':'text','text':text})
    prompt=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
    data=processor(text=[prompt],images=imgs,min_pixels=224*224,max_pixels=224*224,return_tensors='pt').to('cuda')
    with torch.inference_mode():
        result=model.generate(**data,max_new_tokens=1,do_sample=False,return_dict_in_generate=True,output_scores=True)
    logits=result.scores[0][0,letter_ids[:len(row['choices'])]].float()
    return int(logits.argmax().item()),int(data.input_ids.shape[-1])

for row in rows:
    ident=row['question_id']; arms={'ridge_sigma2':row['frame_indices'],'multi_anchor':row['multi_frames']}
    if all((ident,a) in done for a in arms): continue
    with tempfile.TemporaryDirectory(dir='/dev/shm',prefix='ridge-qwen-') as tmp:
        path=Path(tmp)/'video.mp4'; sha=fetch_video(reader,members['videos/'+row['video']],path); reader.clear_cache()
        assert sha==inputs['baselines'][ident]['video_sha256']
        native=decord.VideoReader(str(path),ctx=decord.cpu(0),num_threads=1); count=len(native); fps=float(native.get_avg_fps()); del native
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
