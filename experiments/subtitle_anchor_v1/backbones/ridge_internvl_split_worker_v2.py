"""Paired RIDGE/multi-anchor evaluation; resume only completed JSONL records."""
import argparse, hashlib, json, os, sys, tempfile, time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import inspect

ROOT = Path('/marimo/ridge-canonical')
RUN = ROOT / 'backbone-64-v1'
sys.path.insert(0, str(ROOT / 'relational-w5-v1'))
from worker_sigma2 import frames_full, load_full_inputs, MultipartReader, HTTPRangeSource, archive_index, fetch_video, check_space, TRANSFER_CHUNK_BYTES, decord, digest

parser = argparse.ArgumentParser()
parser.add_argument('--rank', type=int, required=True)
parser.add_argument('--pilot', action='store_true')
parser.add_argument('--part', type=int, choices=[0,1], required=True)
args = parser.parse_args()
assert 0 <= args.rank < 5
torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.19)
rows = json.loads((RUN / 'paired-manifest.json').read_text())
rows = [r for r in rows if (int(hashlib.sha256(r['video'].encode()).hexdigest(),16) % 2 == args.part and (int(hashlib.sha256(r['video'].encode()).hexdigest(),16) // 2) % 5 == args.rank)]
if args.pilot:
    rows = [r for r in rows if len(r['frame_indices']) >= 60][:1]
    assert rows, 'No >=60 frame pilot in shard'
inputs = load_full_inputs(ROOT)
reader = MultipartReader(HTTPRangeSource())
members = archive_index(reader, ROOT / 'archive-index.json', {r['video'] for r in rows})
reader.clear_cache(); reader.block_size = TRANSFER_CHUNK_BYTES
model_path = '/dev/shm/ridge-backbone-hf-cache/models--OpenGVLab--InternVL3-8B/snapshots/853e3a797a661694b1b8ece0cb72dc2b23e3dac9'
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True, use_flash_attn=False, device_map={'':0}, local_files_only=True).eval()
model.language_model.config._attn_implementation = 'sdpa'
assert model.language_model.model.layers[0].self_attn.config._attn_implementation == 'sdpa'
print('LANGUAGE_ATTENTION_SDPA', flush=True)
# Qwen2 MLP is position-wise: chunk its sequence axis to bound activation VRAM.
for _layer in model.language_model.model.layers:
    _forward = _layer.mlp.forward
    def _chunked_mlp(x, _forward=_forward):
        return torch.cat([_forward(part) for part in x.split(1024, dim=1)], dim=1)
    _layer.mlp.forward = _chunked_mlp
print('MLP_CHUNK_SIZE', 1024, flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, local_files_only=True)
letters = [tokenizer.encode(c, add_special_tokens=False) for c in 'ABCDE']
assert all(len(x)==1 for x in letters)
letter_ids = [x[0] for x in letters]
model.img_context_token_id=tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
get_conv_template=inspect.getmodule(model).get_conv_template
transform=T.Compose([T.Lambda(lambda img: img.convert('RGB')), T.Resize((448,448), interpolation=InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))])
outdir = RUN / (f'internvl-split-p{args.part}-pilot-v2' if args.pilot else f'internvl-split-p{args.part}-full')
outdir.mkdir(exist_ok=True)
output = outdir / f'worker-{args.rank}.jsonl'
done = {}
if output.exists():
    for line in output.read_text().splitlines():
        r = json.loads(line); key=(r['id'],r['arm']); assert key not in done; done[key]=r

def answer(row, imgs, times):
    pixel_values=torch.stack([transform(x) for x in imgs]).to('cuda',torch.bfloat16)
    choices='\n'.join(f'{chr(65+i)}. {x}' for i,x in enumerate(row['choices']))
    prefix=''.join(f'Frame at {t:.3f}s: <image>\n' for t in times)
    question=prefix+'Answer the multiple-choice question using the video evidence. Timestamps are seconds in the source video. Select exactly one option letter.\nQuestion: '+row['question']+'\n'+choices+'\nAnswer:'
    template=get_conv_template(model.template); template.system_message=model.system_message; template.append_message(template.roles[0],question); template.append_message(template.roles[1],None); query=template.get_prompt()
    for _ in imgs: query=query.replace('<image>', '<img>'+'<IMG_CONTEXT>'*model.num_image_token+'</img>', 1)
    data=tokenizer(query,return_tensors='pt').to('cuda')
    with torch.inference_mode():
        visual=torch.cat([model.extract_feature(chunk) for chunk in pixel_values.split(2)],dim=0)
        out=model.generate(pixel_values=pixel_values,visual_features=visual,input_ids=data.input_ids,attention_mask=data.attention_mask,max_new_tokens=1,do_sample=False,return_dict_in_generate=True,output_scores=True)
    logits=out.scores[0][0,letter_ids[:len(row['choices'])]].float()
    return int(logits.argmax().item()), int(data.input_ids.shape[-1])

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
