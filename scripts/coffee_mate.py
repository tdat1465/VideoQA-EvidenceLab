#!/usr/bin/env python3
"""Pinned Coffee-Mate prepare/train/evaluate/report, isolated from Molmo experiments."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import statistics
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evidencelab.coffeemate import grade_answers, prepare, verify_prepared, write_json
from evidencelab.data import file_hash


def report(output):
    output = Path(output)
    contract = json.loads((output / 'eval_contract.json').read_text())
    rows = [json.loads(p.read_text()) for p in sorted((output / 'predictions').glob('*.json'))]
    groups, strict_groups, seen, invalid = {}, {}, set(), 0
    for row in rows:
        if row['index'] in seen or not 0 <= row['index'] < contract['expected_questions']:
            raise ValueError('Duplicate/invalid prediction index')
        seen.add(row['index'])
        legacy, strict, letter = grade_answers(row['response'], row['ground_truth'])
        row['correct'], row['strict_correct'], row['parsed_option'] = legacy, strict, letter
        invalid += letter is None
        broad = 'C' if row['type'].startswith('C') else 'T' if row['type'].startswith('T') else 'D'
        for key in (row['type'], broad, 'overall'):
            count = groups.setdefault(key, [0, 0])
            count[0] += bool(row['correct'])
            count[1] += 1
            strict_count = strict_groups.setdefault(key, [0, 0])
            strict_count[0] += strict
            strict_count[1] += 1
    result = {'expected_questions': contract['expected_questions'], 'completed_questions': len(rows),
              'complete': len(rows) == contract['expected_questions'], 'protocol': contract['protocol'],
              'accuracy': {k: {'correct': v[0], 'count': v[1], 'accuracy': v[0]/v[1]} for k, v in groups.items()},
              'strict_option_accuracy': {k: {'correct': v[0], 'count': v[1], 'accuracy': v[0]/v[1]} for k, v in strict_groups.items()},
              'invalid_option_outputs': invalid,
              'generation_seconds': sum(r['generation_seconds'] for r in rows),
              'note': 'Accuracy is over completed questions only. Incomplete run is not full validation accuracy.'}
    write_json(output / 'summary.json', result)
    (output / 'predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


def aggregate(outputs, destination):
    if len(outputs) != 3:
        raise ValueError('Paper reports three independent seeds; supply exactly three evaluation directories')
    contracts = [json.loads((Path(p) / 'eval_contract.json').read_text()) for p in outputs]
    keys = ('val_sha256', 'method', 'profile', 'protocol', 'runtime_hashes', 'versions')
    if any(any(c[k] != contracts[0][k] for k in keys) for c in contracts[1:]):
        raise ValueError('Seed runs differ in validation/protocol/runtime')
    if len({c['seed'] for c in contracts}) != 3:
        raise ValueError('Require three distinct training seeds')
    reports = [report(p) for p in outputs]
    if not all(r['complete'] for r in reports):
        raise ValueError('All seeds must have complete validation results')
    groups = set(reports[0]['accuracy'])
    if any(set(r['accuracy']) != groups for r in reports):
        raise ValueError('Metric groups differ')
    result = {'seeds': [c['seed'] for c in contracts], 'profile': contracts[0]['profile'],
              'method': contracts[0]['method'], 'standard_deviation': 'sample, ddof=1',
              'metrics': {g: {'mean_accuracy': statistics.mean(r['accuracy'][g]['accuracy'] for r in reports),
                              'std_accuracy': statistics.stdev(r['accuracy'][g]['accuracy'] for r in reports)} for g in sorted(groups)}}
    result['strict_option_metrics'] = {
        g: {'mean_accuracy': statistics.mean(r['strict_option_accuracy'][g]['accuracy'] for r in reports),
            'std_accuracy': statistics.stdev(r['strict_option_accuracy'][g]['accuracy'] for r in reports)} for g in sorted(groups)}
    write_json(destination, result)
    return result


def train(args):
    prepared = Path(args.prepared).resolve()
    config, contract = verify_prepared(prepared)
    target = Path(config['output_dir'])
    checkpoints = list(target.glob('*.pth'))
    if args.resume and not checkpoints:
        raise FileNotFoundError('No completed-epoch checkpoint to resume')
    if not args.resume and target.exists() and any(target.iterdir()):
        raise FileExistsError('Use --resume or a new preparation directory')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != contract['world_size']:
        raise RuntimeError('Visible GPU count must match prepared --world-size')
    environment = {'versions': {x: importlib.metadata.version(x) for x in ('torch', 'transformers', 'peft', 'numpy', 'timm')},
                   'cuda': torch.version.cuda, 'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                   'runner_sha256': file_hash(Path(__file__))}
    runtime_path = prepared / 'training_environment.json'
    if args.resume and json.loads(runtime_path.read_text()) != environment:
        raise ValueError('Training environment changed')
    write_json(runtime_path, environment)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node',
               str(contract['world_size']), '-m', 'tasks.train_coffee_mate', str(prepared / 'config.json')]
    if args.resume: command += ['auto_resume', 'True']
    env = dict(os.environ, PYTHONPATH=str(prepared / 'runtime'), PYTHONHASHSEED=str(contract['seed']))
    subprocess.run(command, cwd=prepared / 'runtime', env=env, check=True)


def evaluate(args):
    prepared = Path(args.prepared).resolve()
    config, preparation = verify_prepared(prepared)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError('Trained CoMa checkpoint required; VideoChat2 stage3 alone is not CoMa')
    import torch
    from filelock import FileLock
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Evaluation requires exactly one visible GPU')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / '.lock'), timeout=0):
        annotations = json.loads((prepared / 'val.json').read_text())
        contract = {'prepared_sha256': file_hash(prepared / 'contract.json'), 'checkpoint_sha256': file_hash(checkpoint),
                    'expected_questions': len(annotations), 'method': preparation['method'], 'profile': preparation['profile'],
                    'val_sha256': file_hash(prepared / 'val.json'), 'seed': preparation['seed'],
                    'runtime_hashes': preparation['runtime_hashes'],
                    'protocol': 'upstream-check_ans-middle-sampling-full-nextqa-val', 'runner_sha256': file_hash(Path(__file__)),
                    'versions': {x: importlib.metadata.version(x) for x in ('torch', 'transformers', 'peft', 'numpy', 'timm', 'decord')},
                    'gpu': torch.cuda.get_device_name(0)}
        path = output / 'eval_contract.json'
        if path.exists():
            if not args.resume or json.loads(path.read_text()) != contract:
                raise ValueError('Resume inputs/environment/code mismatch')
        elif args.resume: raise ValueError('No evaluation to resume')
        else: write_json(path, contract)
        sys.path.insert(0, str(prepared / 'runtime'))
        from utils.easydict import EasyDict
        from utils.basic_utils import setup_seed
        from dataset import create_dataset
        from models.coffee_mate import Coffee_Mate
        import models.coffee_mate as cm
        from tasks.shared_utils import check_ans
        setup_seed(preparation['seed'])
        cfg = EasyDict(config)
        data = create_dataset('it_test', cfg)[0]
        if len(data) != len(annotations): raise ValueError('Annotation length changed')
        model = Coffee_Mate(cfg.model).cuda().eval()
        state = torch.load(checkpoint, map_location='cpu')['model']
        required = {n for n, p in model.named_parameters() if p.requires_grad}
        if required - state.keys(): raise ValueError('Checkpoint lacks trained parameters')
        msg = model.load_state_dict(state, strict=False)
        if msg.unexpected_keys: raise ValueError('Unexpected checkpoint keys')
        trace = {}
        original_topk = cm.HardtopK_segment
        def traced_topk(x, m=8, k=1):
            matrix = original_topk(x, m, k)
            trace['candidate_scores'] = x.detach().cpu().tolist()[0]
            trace['selected_positions'] = matrix.argmax(dim=1).cpu().tolist()[0]
            return matrix
        cm.HardtopK_segment = traced_topk
        for ds in data.datasets:
            original_reader = ds.video_reader
            def traced_reader(*values, _reader=original_reader, **kwargs):
                frames, indices, fps = _reader(*values, **kwargs)
                trace.update(decoded_frame_ids=[int(i) for i in indices], fps=float(fps))
                return frames, indices, fps
            ds.video_reader = traced_reader
        for index, row in enumerate(annotations):
            saved = output / 'predictions' / f'{index:06d}.json'
            if saved.exists():
                old = json.loads(saved.read_text())
                if old['index'] != index or old['question_id'] != row['question_id']:
                    raise ValueError('Saved prediction identity mismatch')
                continue
            trace.clear()
            video, text, instruction, question, actual_index, generation, answer, qtype = data[index]
            if actual_index != index: raise RuntimeError('Dataset replaced requested question')
            setup_seed(preparation['seed'] + index)
            video = video.unsqueeze(0).cuda()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.inference_mode():
                # Ground truth never passed to the model or frame selection.
                response = model.generate(video, [''], [instruction], [question], [generation], [''])[0]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            selected = trace.get('selected_positions', list(range(len(trace['decoded_frame_ids']))))
            write_json(saved, {'index': index, 'question_id': row['question_id'], 'video': row['video'], 'type': qtype,
                       'response': response, 'ground_truth': answer, 'correct': bool(check_ans(response, answer)),
                       'generation_seconds': elapsed, 'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(),
                       **trace, 'selected_frame_ids': [trace['decoded_frame_ids'][p] for p in selected]})
            print(f'{index+1}/{len(data)} {row["question_id"]}: {response}', flush=True)
        report(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare')
    for flag in ('train-csv', 'val-csv', 'video-root', 'umt', 'vicuna', 'stage3', 'output'):
        a.add_argument('--'+flag, required=True, type=Path)
    a.add_argument('--profile', choices=('source', 'paper'), default='source')
    a.add_argument('--method', choices=('coma', 'm2l-sd'), default='coma')
    a.add_argument('--seed', type=int, default=42)
    a.add_argument('--world-size', type=int, choices=(1, 2), default=2)
    a = sub.add_parser('train')
    a.add_argument('--prepared', required=True)
    a.add_argument('--resume', action='store_true')
    a = sub.add_parser('evaluate')
    a.add_argument('--prepared', required=True)
    a.add_argument('--checkpoint', required=True)
    a.add_argument('--output', required=True)
    a.add_argument('--resume', action='store_true')
    a = sub.add_parser('report')
    a.add_argument('--output', required=True)
    a = sub.add_parser('aggregate')
    a.add_argument('--evaluations', nargs=3, required=True)
    a.add_argument('--output', required=True)
    args = p.parse_args()
    if args.command == 'prepare': print(prepare(**{k: v for k, v in vars(args).items() if k != 'command'}))
    elif args.command == 'train':
        from filelock import FileLock
        with FileLock(str(Path(args.prepared) / '.training.lock'), timeout=0):
            train(args)
    elif args.command == 'evaluate': evaluate(args)
    elif args.command == 'aggregate': print(json.dumps(aggregate(args.evaluations, args.output), indent=2))
    else: report(args.output)


if __name__ == '__main__': main()
