import hashlib,json,os,subprocess,sys,time
from pathlib import Path
part=int(sys.argv[1]); assert part in (0,1)
py=sys.argv[2]; root=Path('/marimo/ridge-canonical/backbone-64-v1'); stem=f'internvl-split-p{part}'
manifest={r['question_id']:r for r in json.loads((root/'paired-manifest.json').read_text())}
allowed={q for q,r in manifest.items() if int(hashlib.sha256(r['video'].encode()).hexdigest(),16)%2==part}
assert len(allowed)==(666 if part==0 else 671)
deadline=time.monotonic()+1800
while time.monotonic()<deadline:
 logs=[(root/f'{stem}-pilot-v2-worker-{i}.log').read_text() for i in range(5)]
 if all('WORKER_COMPLETE' in x for x in logs): break
 if any('Traceback (most recent call last)' in x for x in logs): print('PILOT_FAILED',flush=True);raise SystemExit(1)
 time.sleep(10)
else: print('PILOT_TIMEOUT',flush=True);raise SystemExit(1)
keys=set()
for rank in range(5):
 records=[json.loads(x) for x in (root/f'{stem}-pilot-v2'/f'worker-{rank}.jsonl').read_text().splitlines()]
 assert len(records)==2
 for r in records:
  assert r['id'] in allowed and r['tokens']>0
  m=manifest[r['id']]; assert len(r['frames'])>=60 and r['frames']==m['frame_indices' if r['arm']=='ridge_sigma2' else 'multi_frames']
  key=(r['id'],r['arm']);assert key not in keys;keys.add(key)
assert len(keys)==10
print('PILOT_VERIFIED',len(keys),flush=True)
procs=[]
for rank in range(5):
 log=(root/f'{stem}-full-worker-{rank}.log').open('wb')
 procs.append(subprocess.Popen([py,str(root/'internvl_split_worker_v2.py'),'--rank',str(rank),'--part',str(part)],stdout=log,stderr=subprocess.STDOUT))
print('FULL_STARTED',[p.pid for p in procs],flush=True)
codes=[p.wait() for p in procs];(root/f'{stem}-full-exit-codes.json').write_text(json.dumps(codes));print('FULL_EXIT_CODES',codes,flush=True)
if codes!=[0]*5:raise SystemExit(1)
records=[]
for rank in range(5):
 for line in (root/f'{stem}-full'/f'worker-{rank}.jsonl').read_text().splitlines():
  r=json.loads(line);m=manifest[r['id']];h=int(hashlib.sha256(m['video'].encode()).hexdigest(),16)
  assert h%2==part and h//2%5==rank and r['frames']==m['frame_indices' if r['arm']=='ridge_sigma2' else 'multi_frames'] and r['correct']==(r['prediction']==m['answer'])
  records.append(r)
assert len(records)==2*len(allowed) and {(r['id'],r['arm']) for r in records}=={(q,a) for q in allowed for a in ('ridge_sigma2','multi_anchor')}
print('FULL_COMPLETE',len(records),flush=True)
