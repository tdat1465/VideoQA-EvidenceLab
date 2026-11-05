"""Label-independent whole-video partitioning and verified merged exports."""
import json
from pathlib import Path


def partitions(inputs, workers):
    if not 1 <= workers <= 5:
        raise ValueError('Support one through five GPU workers')
    groups = {}
    for sample in inputs['samples']:
        groups.setdefault(sample['video'], []).append(sample['question_id'])
    assignments, loads = [[] for _ in range(workers)], [0] * workers
    costs = {v: sum(inputs['baselines'][q]['ridge_candidate_count'] for q in qs) for v, qs in groups.items()}
    for video in sorted(groups, key=lambda v: (-costs[v], v)):
        rank = min(range(workers), key=lambda i: (loads[i], i))
        assignments[rank].extend(groups[video])
        loads[rank] += costs[video]
    return [sorted(ids) for ids in assignments]


def summarize(rows, ids, variants):
    expected = {q + '::' + v for q in ids for v in variants}
    byid = {r['id']: r for r in rows}
    if len(byid) != len(rows) or not set(byid) <= expected:
        raise ValueError('Duplicate or foreign parallel results')
    groups = {}
    for r in rows:
        if r['id'] != r['question_id'] + '::' + r['variant']:
            raise ValueError('Result ID mismatch')
        groups.setdefault(r['question_id'], set()).add(r['variant'])
    complete = sum(g == set(variants) for g in groups.values())
    return {'complete': set(byid) == expected, 'completed_variant_answers': len(rows),
            'target_variant_answers': len(expected), 'completed_questions': complete,
            'target_questions': len(ids), 'variants': {v: {'n': sum(r['variant'] == v for r in rows),
            'correct': sum(r['variant'] == v and r['correct'] for r in rows)} for v in variants},
            'actual_answer_forwards': sum(r['answer_inference_performed'] for r in rows),
            'same_run_answer_reuses': sum(not r['answer_inference_performed'] for r in rows)}


def export(output, contract, store):
    from evidencelab.store import atomic_json
    rows = store.results()
    summary = summarize(rows, contract['manifest']['ids'], contract['variants'])
    atomic_json(output / 'summary.json', summary)
    temp = output / 'predictions.jsonl.tmp'
    with temp.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(output / 'predictions.jsonl')
    return summary


def gpu_fraction(workers):
    if not 1 <= workers <= 5:raise ValueError("Invalid worker count")
    return min(0.9, 0.90/workers)


def load_full_inputs(root):
    import hashlib
    import sys
    root=Path(root)
    sys.path[:0]=[str(root/'repo-fresh/src'),str(root/'repo-fresh/scripts')]
    from huggingface_hub import hf_hub_download
    from evidencelab.longvideo_data import parse_annotations, DATASET_ID
    from evidencelab.ridge_weights import validate_record
    raw=json.loads((root/'inputs-full1337.json').read_text())
    if raw['schema']!='ridge-molab-full-inputs-v1':raise ValueError('Wrong full input schema')
    path=Path(hf_hub_download(DATASET_ID,'lvb_val.json',repo_type='dataset',revision=raw['config']['dataset_revision']))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=raw['annotation_sha256']:raise ValueError('Frozen annotations changed')
    samples=parse_annotations(json.loads(path.read_text()))
    if len(samples)!=1337 or sorted(s.id for s in samples)!=raw['pilot_ids']:raise ValueError('Full question coverage changed')
    if set(raw['weights'])!=set(raw['pilot_ids']):raise ValueError('Incomplete text weights')
    for sample in samples:validate_record(raw['weights'][sample.id],sample.public_question())
    raw['samples']=[{'question_id':s.id,'question':s.question,'choices':list(s.choices),'answer':s.answer,
                     'video':s.video,'question_type':s.question_type,'duration_group':s.duration_group} for s in samples]
    return raw
