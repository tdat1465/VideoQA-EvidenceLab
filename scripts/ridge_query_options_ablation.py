"""Question plus options score ablation. New run; all frozen baseline files stay read-only."""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import json
import math
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import ridge_component_ablation as frozen
from evidencelab.data import digest,file_hash
from evidencelab.store import Store,atomic_json,run_lock
from evidencelab.ridge_adapter import RIDGE_FILES,validate_weights


VARIANTS=('sigma2','qopts_mean_sigma2')
BASE_SOURCE=frozen.BASE_SOURCE
COMPONENT_SHA='ee89d02c2cf37c8c130db877665792ab52302e41f80b1301140769e1399eb083'
PROTOTYPE={'schema':'question-options-mean-v1','scales':[2.0],
    'question_score_weight':0.5,'option_aggregation':'mean-of-per-option-mean-chunks',
    'option_order':'sorted-normalized-text','text_content_tokens_per_chunk':33,
    'text_total_tokens_max':35,'keep_all_option_tokens':True,
    'reuse_question_only_scores':True,'reuse_question_only_qwen_weights':True,
    'pilot_questions':20,'pilot_seed':20261022}
load=frozen.load
read_score_cache=frozen.read_score_cache
cache_key=frozen.cache_key

def validate_source(source,baseline):
    if file_hash(Path(frozen.__file__))!=COMPONENT_SHA:
        raise ValueError('Frozen component helper changed')
    return frozen.validate_source(source,baseline)

def pilot_manifest(baseline_rows,manifest):
    return frozen.choose_development([r for r in baseline_rows if r['id'] in set(manifest['ids'])],
        count=PROTOTYPE['pilot_questions'],seed=PROTOTYPE['pilot_seed'])

def option_plan(question,tokenizer):
    # Only public question data; no correct choice, category or outcomes.
    import re
    if not 2<=len(question.choices)<=5:raise ValueError('Require 2-5 options')
    texts=[]
    for choice in question.choices:
        text=re.sub(r'([.!"()*#:;~])',' ',choice.lower())
        text=re.sub(r'\s+',' ',text).strip()
        if not text:raise ValueError('Empty normalized option')
        texts.append(text)
    result=[]
    for text in sorted(texts):
        ids=tokenizer.encode(text,add_special_tokens=False,truncation=False)
        if not ids:raise ValueError('Empty option token sequence')
        chunks=[ids[i:i+33] for i in range(0,len(ids),33)]
        if [v for chunk in chunks for v in chunk]!=ids:raise AssertionError('Option tokens lost')
        result.append({'text':text,'content_ids':chunks,'token_count':len(ids)})
    return result

def combine_scores(question_scores,per_option):
    import math
    if not 2<=len(per_option)<=5 or not question_scores:raise ValueError('Invalid option scores')
    n=len(question_scores)
    if any(len(x)!=n for x in per_option):raise ValueError('Score shape mismatch')
    if any(not math.isfinite(x) or not 0<=x<=1 for v in [question_scores,*per_option] for x in v):
        raise ValueError('Invalid BLIP probability')
    mean=[sum(v[i] for v in per_option)/len(per_option) for i in range(n)]
    return [(q+o)/2 for q,o in zip(question_scores,mean)]

def score_options(scorer,video,candidates,plan,batch_size,stop=lambda:False):
    # Reuse transformed pixels within a small frame batch. Each ITM forward stays
    # the frozen scorer's original HF model forward, with a different text input.
    from PIL import Image
    torch=scorer.torch
    scorer.reset()
    values=[[] for _ in plan]
    for offset in range(0,len(candidates),batch_size):
        if stop():raise InterruptedError('Stop during option scoring')
        batch=candidates[offset:offset+batch_size]
        pixels=torch.stack([scorer.transform(Image.fromarray(video[i].asnumpy()).convert('RGB')) for i in batch]).to('cuda:0')
        for j,option in enumerate(plan):
            sums=[0.0]*len(batch)
            for content in option['content_ids']:
                if stop():raise InterruptedError('Stop between option chunks')
                ids=scorer.processor.tokenizer.build_inputs_with_special_tokens(content)
                if len(ids)>35:raise ValueError('BLIP text exceeds frozen 35-token limit')
                ids[0]=scorer.enc_id
                inputs={'input_ids':torch.tensor([ids]*len(batch),device='cuda:0'),
                        'attention_mask':torch.ones((len(batch),len(ids)),dtype=torch.long,device='cuda:0')}
                with torch.inference_mode():
                    out=scorer.model(pixel_values=pixels,**inputs,use_itm_head=True).itm_score
                    scores=torch.softmax(out.float(),dim=-1)[:,1].cpu().tolist()
                sums=[a+b for a,b in zip(sums,scores)]
                scorer.calls+=1
                scorer.evaluated.extend(batch)
            values[j].extend(v/len(option['content_ids']) for v in sums)
    return values

def query_score(key,question_scores,per_option,plan,seconds,batches):
    if len(per_option)!=len(plan):raise ValueError('Missing option score or query plan')
    record=frozen.score_record(key,combine_scores(question_scores,per_option),seconds,batches)
    record.update(question_scores=question_scores,per_option_scores=per_option,query_plan=plan)
    record['sha256']=digest({k:v for k,v in record.items() if k!='sha256'})
    return record

def read_query_score(path,key,question_scores,plan):
    record=read_score_cache(path,key)
    if record['question_scores']!=question_scores or record['query_plan']!=plan:
        raise ValueError('Query/old question score changed')
    if len(record['per_option_scores'])!=len(plan):raise ValueError('Incomplete option score cache')
    if record['scores']!=combine_scores(question_scores,record['per_option_scores']):
        raise ValueError('Incorrect query aggregation')
    return record

def select_variant(author,score,weights,variant,k=64):
    if variant not in VARIANTS or k!=64:raise ValueError('Frozen 64-frame query experiment')
    values=score['question_scores'] if variant=='sigma2' else score['scores']
    return frozen.select_variant(author,values,weights,'sigma2',k)

def evaluate_question(sample,candidates,fps,weights,score,author,store,diagnostic_path,
                      answer,stop=lambda:False,reference=None):
    diagnostics=load(diagnostic_path) if diagnostic_path.exists() else {'variants':{}}
    if diagnostics.get('score_sha256',score['sha256'])!=score['sha256']:raise ValueError('Curve score changed')
    diagnostics.update(question_id=sample.id,score_sha256=score['sha256'])
    for variant in VARIANTS:
        key=sample.id+'::'+variant
        if key in store.done():continue
        if stop():raise InterruptedError('Stop between query variants')
        start=time.perf_counter()
        positions,details=select_variant(author,score,weights['weights'],variant)
        seconds=time.perf_counter()-start
        indices=[candidates[i] for i in positions];times=[i/fps for i in indices]
        diagnostics['variants'][variant]={**details,'selected_positions':positions,'frame_indices':indices,'timestamps':times}
        atomic_json(diagnostic_path,diagnostics)
        decision,answer_seconds,measurements=answer(indices,times)
        if (len(decision.scores)!=len(sample.choices) or any(not math.isfinite(x) or not 0<=x<=1 for x in decision.scores)
                or abs(sum(decision.scores)-1)>1e-4):raise ValueError('Invalid answer scores')
        fm=indices==reference['calls'][0]['frame_indices'];pm=decision.prediction==reference['prediction']
        if (variant=='sigma2' and not fm) or (fm and not pm):
            raise ValueError('Same-frame baseline control mismatch; investigate before continuing')
        row={'id':key,'question_id':sample.id,'variant':variant,'question':sample.question,'choices':list(sample.choices),
             'answer':sample.answer,'prediction':decision.prediction,'correct':sample.answer==decision.prediction,
             'video':sample.video,'question_type':sample.question_type,'duration_group':sample.duration_group,
             'frame_indices':indices,'timestamps':times,'final_frames':len(indices),'scores':decision.scores,
             'input_tokens':decision.input_tokens,'selector_only_seconds':seconds,'answer_decode_seconds':answer_seconds,
             'cached_question_seconds':seconds+answer_seconds,'score_sha256':score['sha256'],
             'weight_sha256':weights['record_sha256'],'baseline_frame_match':fm,'baseline_prediction_match':pm,**measurements}
        store.put(row)
        print(f'Committed {sample.id}: {variant}, correct={row["correct"]}',flush=True)

def export(output,contract,store):
    rows=store.results();groups={}
    stats={}
    for r in rows:groups.setdefault(r['question_id'],{})[r['variant']]=r
    for v in VARIANTS:
        rr=[r for r in rows if r['variant']==v]
        stats[v]={'n':len(rr),'correct':sum(r['correct'] for r in rr),
                  'accuracy':sum(r['correct'] for r in rr)/len(rr) if rr else None}
    completed={q for q,g in groups.items() if set(g)==set(VARIANTS)}
    pilot=set(contract['pilot20_manifest']['ids'])
    scores=[load(p) for p in sorted((output/'scores').glob('*.json'))]
    events=[json.loads(x) for x, in store.db.execute('SELECT value FROM events')]
    summary={'complete':len(rows)==contract['selected_count'],'selected_questions':200,
             'completed_variant_answers':len(rows),'variants':stats,
             'pilot20':{'target_questions':20,'completed_questions':len(completed&pilot),'complete':pilot<=completed},
             'new_blip_batches':sum(x['scorer_batches'] for x in scores),'new_qwen_calls':0,
             'one_time_scored_questions':len(scores),'one_time_scoring_seconds':sum(x['original_scoring_seconds'] for x in scores),
             'baseline_reproduction':{'n':stats['sigma2']['n'],
                'same_frames':sum(r['variant']=='sigma2' and r['baseline_frame_match'] for r in rows),
                'same_prediction':sum(r['variant']=='sigma2' and r['baseline_prediction_match'] for r in rows)},
             'paired_vs_sigma2':{'n':len(completed),
                'gains':sum(groups[q]['qopts_mean_sigma2']['correct'] and not groups[q]['sigma2']['correct'] for q in completed),
                'losses':sum(groups[q]['sigma2']['correct'] and not groups[q]['qopts_mean_sigma2']['correct'] for q in completed)},
             'recorded_sessions':{name:sum(e[field] for e in events if e['event']==kind)
                for name,kind,field in [('worker_wall_seconds','session_end','wall_seconds'),
                    ('model_setup_seconds','models_ready','seconds'),('video_transfer_hash_seconds','video_loaded','transfer_hash_seconds')]},
             'timing_scope':'New option BLIP scoring charged separately; cached answer timing excludes scoring, old Qwen/BLIP and setup.',
             'interpretation':'Development ablation. Options are public inference inputs; no answer labels enter scoring.'}
    atomic_json(output/'summary.json',summary)
    with (output/'predictions.jsonl').open('w',encoding='utf-8',newline='\n') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    return summary

def run(args):
    from evidencelab.aks_adapter import aks_candidates
    from evidencelab.backends import HFBackend
    from evidencelab.focus_adapter import BlipITMScorer
    from evidencelab.config import source_fingerprint
    from evidencelab.hardware import execution_profile
    from evidencelab.longvideo import longvideo_runtime, validate_ram_root
    from evidencelab.longvideo_config import LongVideoConfig
    from evidencelab.longvideo_data import (parse_annotations, archive_index, HTTPRangeSource,
        MultipartReader, fetch_video, TRANSFER_CHUNK_BYTES, check_space)
    from evidencelab.ridge_adapter import fetch_ridge, load_ridge
    from evidencelab.ridge_weights import RidgeWeightCache, weight_contract
    import decord
    from PIL import Image
    baseline, output, ram = args.baseline.resolve(), args.output.resolve(), args.ram_root.resolve()
    cached = validate_source(args.source_cache, baseline)
    source_cache = args.source_cache.resolve()
    if output == source_cache or source_cache in output.parents or output in source_cache.parents:
        raise ValueError("Component output must be separate from read-only score source")
    if output == baseline or baseline in output.parents or output in baseline.parents:
        raise ValueError("Ablation output must be separate from the frozen baseline")
    validate_ram_root(ram)
    config = LongVideoConfig.load(args.config)
    old, baseline_rows, old_summary = load(baseline/"contract.json"), [], load(baseline/"summary.json")
    with (baseline/"predictions.jsonl").open(encoding="utf-8") as handle:
        baseline_rows = [json.loads(line) for line in handle if line.strip()]
    if (source_fingerprint() != BASE_SOURCE or old["source_sha256"] != BASE_SOURCE
            or config.contract() != old["config"] or not old_summary["complete"]
            or len(baseline_rows) != 1337 or digest(sorted(v["id"] for v in baseline_rows)) != old["selected_ids_hash"]
            or file_hash(args.annotations) != old["annotation_sha256"]):
        raise ValueError("Frozen baseline/config/source/annotation mismatch")
    execution = execution_profile(config)
    runtime = longvideo_runtime(config)
    if execution["resolved_dtype"] != old["resolved_dtype"] or runtime != old["runtime"]:
        raise ValueError("Runtime/precision changed; do not silently reuse baseline preferences")
    author, weight_author = load_ridge(fetch_ridge(ram/"ridge-upstream"))
    samples = parse_annotations(load(args.annotations))
    weights = RidgeWeightCache(baseline/"ridge-weights.json", weight_contract(config, old, weight_author),
                               [q.public_question() for q in samples])
    weights.check_completed(baseline_rows)
    manifest = cached["contract"]["manifest"]
    pilot = pilot_manifest(baseline_rows, manifest)
    selected = sorted((q for q in samples if q.id in set(manifest["ids"])), key=lambda q:(q.video,q.id))
    if len(selected) != 200:
        raise ValueError("Development annotation coverage changed")
    contract = {"schema": "ridge-query-options-ablation-v1", "source_sha256": BASE_SOURCE,
                "experiment_script_sha256": file_hash(Path(__file__)), "config": config.contract(),
                "runtime": runtime, "resolved_dtype": execution["resolved_dtype"],
                "baseline_contract_sha256": digest(old), "baseline_weights_sha256": file_hash(baseline/"ridge-weights.json"),
                "baseline_predictions_sha256": file_hash(baseline/"predictions.jsonl"),
                "manifest": manifest, "pilot20_manifest":pilot, "component_helper_sha256":COMPONENT_SHA, "variants": list(VARIANTS), "prototype": PROTOTYPE,
                "selected_count": 200 * len(VARIANTS), "author_source": RIDGE_FILES,
                "score_source": str(source_cache), "score_source_contract_sha256": file_hash(source_cache/"contract.json"),
                "score_source_predictions_sha256": file_hash(source_cache/"predictions.jsonl"),
                "score_inventory": cached["inventory"], "reference_variants": cached["summary"]["variants"]}
    stopped, old_handlers = [], {}
    for name in ("SIGINT", "SIGTERM", "SIGUSR1"):
        number = getattr(signal, name, None)
        if number is not None:
            old_handlers[number] = signal.signal(number, lambda signum, frame: stopped.append(signum))
    started = time.monotonic()
    stop = lambda: bool(stopped) or time.monotonic()-started >= args.max_seconds
    video, reader, backend, scorer = None, None, None, None
    video_path = ram/"ablation-active-video.mp4"
    baseline_by_id = {v["id"]:v for v in baseline_rows}
    try:
        with run_lock(output):
            store = Store(output, contract, (output/"results.sqlite3").exists())
            try:
                (output/"curves").mkdir(exist_ok=True)
                (output/"scores").mkdir(exist_ok=True)
                atomic_json(output/"manifest.json", manifest)
                done = store.done()
                scope = set(pilot["ids"]) if args.mode == "pilot20" else set(manifest["ids"])
                selected = [q for q in selected if q.id in scope]
                pending = [q for q in selected if any(q.id+"::"+v not in done for v in VARIANTS)]
                if not pending:
                    mode_summary=export(output,contract,store)
                    return 0 if mode_summary['complete'] or (args.mode=='pilot20' and mode_summary['pilot20']['complete']) else 75
                if stop():
                    raise InterruptedError("Stop before setup")
                source = HTTPRangeSource(stop=stop)
                reader = MultipartReader(source)
                members = archive_index(reader, args.index, {q.video for q in selected})
                reader.clear_cache()
                reader.block_size = TRANSFER_CHUNK_BYTES
                setup_start = time.perf_counter()
                backend, scorer = HFBackend(config), BlipITMScorer(config)
                store.event({"event":"models_ready", "seconds":time.perf_counter()-setup_start, "execution":execution})
                current_video, completed_questions = None, 0
                for sample in pending:
                    if stop() or (args.max_new_questions and completed_questions >= args.max_new_questions):
                        break
                    if sample.video != current_video:
                        video = None
                        gc.collect()
                        video_path.unlink(missing_ok=True)
                        transfer_start = time.perf_counter()
                        video_sha = fetch_video(reader, members["videos/"+sample.video], video_path, stop=stop)
                        reader.clear_cache()
                        store.event({"event":"video_loaded", "video":sample.video,
                                     "transfer_hash_seconds":time.perf_counter()-transfer_start})
                        video = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=2)
                        current_video = sample.video
                    prior = baseline_by_id[sample.id]
                    if video_sha != prior["video_sha256"]:
                        raise ValueError("Source video bytes changed")
                    fps = float(video.get_avg_fps())
                    if len(video) == 0 or not math.isfinite(fps) or fps <= 0:
                        raise ValueError("Invalid source video FPS/length")
                    candidates = aks_candidates(len(video), fps)
                    if len(candidates) != prior["ridge_candidate_count"] or len(candidates) < 64:
                        raise ValueError("Candidate grid differs from the frozen baseline")
                    key = cache_key(cached["contract"], sample, video_sha, candidates, fps, len(video))
                    name = digest(sample.id)+".json"
                    score = read_score_cache(source_cache/"scores"/name, key)
                    if score["sha256"] != cached["inventory"][sample.id]:
                        raise ValueError("Read-only score cache changed during this job")
                    plan=option_plan(sample.public_question(),scorer.processor.tokenizer)
                    new_key={**cache_key(contract,sample,video_sha,candidates,fps,len(video)),
                             'public_query_sha256':digest({'question':sample.question,'choices':sorted(sample.choices)}),
                             'query_plan_sha256':digest(plan),'source_question_score_sha256':score['sha256']}
                    query_path=output/'scores'/name
                    if query_path.exists():
                        score=read_query_score(query_path,new_key,score['scores'],plan)
                    else:
                        begin=time.perf_counter()
                        option_scores=score_options(scorer,video,candidates,plan,config.blip_batch_size,stop)
                        backend.torch.cuda.synchronize()
                        score=query_score(new_key,score['scores'],option_scores,plan,time.perf_counter()-begin,scorer.calls)
                        atomic_json(query_path,score)
                    def answer(indices, times):
                        backend.reset_peak()
                        begin = time.perf_counter()
                        check_space(ram, 0)
                        frames = [Image.fromarray(video[i].asnumpy()).convert("RGB") for i in indices]
                        backend.video_duration = len(video)/fps
                        decision = backend.answer(sample.public_question(), frames, times)
                        measurements = backend.measurements()
                        measurements["execution"] = execution
                        return decision, time.perf_counter()-begin, measurements
                    evaluate_question(sample, candidates, fps, weights.records[sample.id], score, author,
                                      store, output/"curves"/name, answer, stop, reference=prior)
                    completed_questions += 1
                    store.event({"event":"question_done", "question_id":sample.id})
            except InterruptedError:
                store.event({"event":"controlled_stop", "signals":stopped})
            finally:
                video = None
                gc.collect()
                if reader is not None:
                    reader.clear_cache()
                video_path.unlink(missing_ok=True)
                try:
                    store.event({"event":"session_end", "wall_seconds":time.monotonic()-started, "signals":stopped})
                    summary = export(output, contract, store)
                    print(json.dumps(summary,ensure_ascii=False),flush=True)
                finally:
                    store.close()
            return 0 if summary['complete'] or (args.mode=='pilot20' and summary['pilot20']['complete']) else 75
    finally:
        for number, handler in old_handlers.items():
            signal.signal(number, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    if "--preflight" in sys.argv:
        parser.add_argument("--preflight", action="store_true")
        parser.add_argument("--baseline", type=Path, required=True)
        parser.add_argument("--source-cache", type=Path, required=True)
        args = parser.parse_args()
        data = validate_source(args.source_cache, args.baseline)
        print(json.dumps({"preflight":"passed", "cached_questions":len(data["inventory"]),
                          "variants":list(VARIANTS), "new_answers":400, "new_blip_scoring":"options only", "new_qwen_calls":0}))
        return 0
    parser.add_argument("--mode", choices=["pilot20","dev200"], default="pilot20")
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ram-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--max-new-questions", type=int, default=0)
    parser.add_argument("--max-seconds", type=int, default=165600)
    args = parser.parse_args()
    if args.max_new_questions < 0 or args.max_seconds <= 0:
        parser.error("Invalid session limit")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
