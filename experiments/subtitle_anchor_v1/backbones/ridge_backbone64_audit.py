import collections, hashlib, json, zipfile
from pathlib import Path
import numpy as np

root=Path(__file__).parent
manifest_path=root/'paired-manifest-64.json'
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()=='7e658e00c695c21ec5253bb725d8d126a7135f4920f7458e9c5174460fd411c5'
manifest={r['question_id']:r for r in json.loads(manifest_path.read_text(encoding='utf8'))}
assert len(manifest)==1337
report={}
for model in ('qwen','llava'):
    path=root/f'ridge-{model}64-checkpoint'/'LATEST.zip'
    records={}; ranks=collections.Counter()
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None and json.loads(z.read('checkpoint.json'))['records']==2674
        for rank in range(5):
            name=f'{model}-full/worker-{rank}.jsonl'
            for line in z.read(name).splitlines():
                r=json.loads(line); sample=manifest[r['id']]; key=(r['id'],r['arm'])
                assert key not in records and int(hashlib.sha256(sample['video'].encode()).hexdigest(),16)%5==rank
                assert r['frames']==sample['frame_indices' if r['arm']=='ridge_sigma2' else 'multi_frames']
                assert 0<=r['prediction']<len(sample['choices']) and r['correct']==(r['prediction']==sample['answer'])
                records[key]=r;ranks[rank]+=1
    assert set(records)=={(q,a) for q in manifest for a in ('ridge_sigma2','multi_anchor')}
    arms=('ridge_sigma2','multi_anchor'); results={}
    for arm in arms:
        results[arm]={'correct':sum(records[(q,arm)]['correct'] for q in manifest),'total':len(manifest)}
    gains=losses=frame_changes=prediction_changes=0; byvideo=collections.defaultdict(lambda:[0,0]); groups=collections.defaultdict(lambda:[0,0,0])
    for q,m in manifest.items():
        a,b=(records[(q,x)] for x in arms); gains+=not a['correct'] and b['correct'];losses+=a['correct'] and not b['correct']
        frame_changes+=a['frames']!=b['frames'];prediction_changes+=a['prediction']!=b['prediction']
        byvideo[m['video']][0]+=int(b['correct'])-int(a['correct']);byvideo[m['video']][1]+=1
        group=groups[m['question_type']];group[0]+=int(a['correct']);group[1]+=int(b['correct']);group[2]+=1
    videos=sorted(byvideo); values=np.array([byvideo[v] for v in videos]); rng=np.random.default_rng(20261102); samples=[]
    for _ in range(100):
        ix=rng.integers(0,len(videos),size=(100,len(videos))); totals=values[ix].sum(axis=1);samples.extend((100*totals[:,0]/totals[:,1]).tolist())
    report[model]={'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'records':len(records),'videos':len(videos),'rank_counts':dict(ranks),'accuracy':results,'gains':gains,'losses':losses,'frame_changes':frame_changes,'prediction_changes':prediction_changes,'delta_pp':100*(gains-losses)/len(manifest),'bootstrap_video_95ci_pp':np.quantile(samples,[.025,.975]).tolist(),'question_types':dict(groups)}
out=root/'ridge-backbone64-audit.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(report,ensure_ascii=False,indent=2))
