"""Pinned Coffee-Mate integration. CPU preparation never imports model code."""
from __future__ import annotations
import copy
import ast
import csv
import json
import re
import shutil
from pathlib import Path

from .data import file_hash, video_index, relative_video

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / 'vendor' / 'coffee_mate'


def grade_answers(response, ground_truth):
    """Expose the exact legacy grader and a strict option-letter audit metric."""
    tree = ast.parse((VENDOR / 'tasks/shared_utils.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'check_ans')
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<pinned-grader>', 'exec'), namespace)
    legacy = bool(namespace['check_ans'](response, ground_truth))
    pattern = r'^\s*[\(\[]?([A-E])(?:[\)\].\s]|$)'
    predicted = re.match(pattern, response, re.I)
    target = re.match(pattern, ground_truth, re.I)
    if not target:
        raise ValueError('Invalid evaluation gold option')
    letter = predicted.group(1).upper() if predicted else None
    return legacy, letter == target.group(1).upper(), letter


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def verify_vendor():
    provenance = json.loads((ROOT / 'docs/COFFEE_MATE_PROVENANCE.json').read_text())
    actual = {p.relative_to(VENDOR).as_posix(): file_hash(p) for p in VENDOR.rglob('*') if p.is_file()
              and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    if actual != provenance['files']:
        raise ValueError('Vendored Coffee-Mate differs from pinned provenance')
    return provenance


def convert_nextqa(annotations, video_root, instructions, training=False, instruction_count=5):
    """One row per QA; preserve CSV order, IDs, types and video-relative paths."""
    index = video_index(Path(video_root))
    with Path(annotations).open(encoding='utf-8-sig', newline='') as f:
        original = list(csv.DictReader(f))
    if not original or not 1 <= instruction_count <= len(instructions):
        raise ValueError('Empty annotations or invalid instruction count')
    rows, seen = [], set()
    for i, row in enumerate(original):
        qid = f"{row['video']}:{row['qid']}"
        if qid in seen:
            raise ValueError('Duplicate question ID')
        seen.add(qid)
        video = index.get(str(row['video']))
        if video is None:
            raise FileNotFoundError(f"Missing video {row['video']}")
        answer = int(row['answer'])
        if not 0 <= answer < 5 or not row['question'].strip():
            raise ValueError('Invalid answer/question')
        options = [row[f'a{k}'].strip() for k in range(5)]
        if any(not x for x in options):
            raise ValueError('Empty option')
        q = 'Question: ' + row['question'].strip() + '\nOptions:\n' + '\n'.join(
            f'({chr(65+k)}) {option.rstrip(".")}.' for k, option in enumerate(options))
        instruction = instructions[i % instruction_count] if training else instructions[0]
        a = f'Answer: ({chr(65+answer)}) {options[answer].rstrip(".")}.'
        if row['type'] not in ('CH', 'CW', 'TN', 'TC', 'TP', 'DL', 'DC', 'DO'):
            raise ValueError(f"Unknown NExT-QA type: {row['type']}")
        rows.append({'video': relative_video(video), 'QA': [{'i': instruction, 'q': q, 'a': a}],
                     'type': row['type'], 'question_id': qid})
    if training:
        # Author's sampler requires nine distinct triples from other videos of same type.
        grouped, by_video = {}, {}
        for row in rows:
            kind = 'TN' if row['type'] == 'TP' else row['type']
            triple = row['QA'][0]['q'] + row['QA'][0]['a']
            grouped.setdefault(kind, set()).add(triple)
            by_video.setdefault(row['video'], set()).add(triple)
        for row in rows:
            kind = 'TN' if row['type'] == 'TP' else row['type']
            if len(grouped[kind] - by_video[row['video']]) < 9:
                raise ValueError(f"Too few cross-video negatives for {row['question_id']}; use full train split")
    return rows


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Pinned upstream patch anchor mismatch: ' + old[:80])
    return text.replace(old, new, 1)


def materialize_runtime(destination):
    """Compatibility/audit fixes only; all modifications recorded in recipe docs."""
    verify_vendor()
    destination = Path(destination)
    shutil.copytree(VENDOR, destination, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    path = destination / 'dataset/__init__.py'
    text = path.read_text()
    old = 'sample_type=config.inputs.video_input.sample_type,\n        num_frames=config.inputs.video_input.num_frames_test,'
    text = replace_once(text, old, 'sample_type=config.inputs.video_input.sample_type_test,\n        num_frames=config.inputs.video_input.num_frames_test,')
    path.write_text(text, encoding='utf-8')
    path = destination / 'dataset/it_dataset.py'
    text = path.read_text()
    text = replace_once(text, 'neg_list = question_type_list - set(currnet_sample) - video_list',
                        'neg_list = question_type_list - {currnet_sample} - video_list')
    text = replace_once(text, 'random.sample(list(neg_list), 9)', 'random.sample(sorted(neg_list), 9)')
    text = text.replace('index = np.random.randint(0, len(self))\n            return self.__getitem__(index)',
                        'raise RuntimeError(f"Dataset decode failed at query {index}") from e')
    path.write_text(text, encoding='utf-8')
    path = destination / 'tasks/train_coffee_mate.py'
    text = path.read_text()
    text = replace_once(text, '        with torch.cuda.amp.autocast(enabled=config.fp16):',
                        '        qsn_id = qsn_id.to(device, non_blocking=True)\n        with torch.cuda.amp.autocast(enabled=config.fp16):')
    text = replace_once(text, 'def val(model, test_loaders, epoch, device, config):',
                        '@torch.no_grad()\ndef val(model, test_loaders, epoch, device, config):')
    path.write_text(text, encoding='utf-8')
    return {p.relative_to(destination).as_posix(): file_hash(p) for p in destination.rglob('*') if p.is_file()}


def resolved(value, context):
    if isinstance(value, dict):
        return {k: resolved(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [resolved(v, context) for v in value]
    if isinstance(value, str) and '${' in value:
        match = re.fullmatch(r'\$\{([\w.]+)\}', value)
        if not match:
            raise ValueError('Unresolved config interpolation: ' + value)
        target = context
        for key in match.group(1).split('.'):
            target = target[key]
        return resolved(target, context)
    return value


def recipe(train, val, video_root, umt, vicuna, stage3, output, profile='source', method='coma', seed=42, world_size=2):
    if profile not in ('source', 'paper') or method not in ('coma', 'm2l-sd') or world_size not in (1, 2):
        raise ValueError('Unsupported profile/method/world size')
    # Trusted pinned config contains only literals and assignments after removing its corpus import.
    template = (VENDOR / 'scripts/config_coffee_mate.py').read_text()
    template = replace_once(template, 'from configs.instruction_data import *', 'available_corpus = {}')
    namespace = {}
    exec(compile(template, '<pinned-coffee-config>', 'exec'), namespace)
    config = {k: copy.deepcopy(v) for k, v in namespace.items() if not k.startswith('__')}
    config.update(dataset='nextqa', method=method, train_file=[str(train), str(video_root), 'video'],
                  test_file=[str(val), str(video_root), 'video'], output_dir=str(output), seed=seed,
                  num_frames_test=16 if method == 'coma' else 8, auto_resume=False, save_latest=False)
    config['model'].update(method=method, vit_blip_model_path=str(umt), llama_model_path=str(vicuna),
                           videochat2_model_path=str(stage3))
    # Source globals remain bz1/accu2; paper profile chooses effective global bz4.
    if profile == 'paper':
        config['batch_size'] = 1
        config['accumulation_steps'] = 4 // world_size
        config['optimizer']['weight_decay'] = .2
    config['inputs']['batch_size_test'] = {'image': 1, 'video': 1}
    config['recipe_profile'] = profile
    config['expected_world_size'] = world_size
    return resolved(config, config)


def prepare(train_csv, val_csv, video_root, umt, vicuna, stage3, output, profile='source', method='coma', seed=42, world_size=2):
    output, video_root = Path(output).resolve(), Path(video_root).resolve()
    if output.exists():
        raise FileExistsError('Use a new preparation directory')
    provenance = verify_vendor()
    for path in (Path(umt), Path(stage3)):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f'Required VideoChat2 initialization weights: {path}')
    if not (Path(vicuna) / 'config.json').is_file():
        raise FileNotFoundError('Vicuna-v0 Hugging Face model directory is required')
    llm = json.loads((Path(vicuna) / 'config.json').read_text())
    if llm.get('hidden_size') != 4096 or llm.get('num_hidden_layers') != 32:
        raise ValueError('Expected 7B Vicuna architecture; v0 identity must also be checked against checkpoint provenance')
    instructions = json.loads((VENDOR / 'data/instructions_nextqa.json').read_text())
    train = convert_nextqa(train_csv, video_root, instructions, True)
    val = convert_nextqa(val_csv, video_root, instructions)
    if {r['question_id'] for r in train} & {r['question_id'] for r in val}:
        raise ValueError('Train/validation question overlap')
    if {r['video'] for r in train} & {r['video'] for r in val}:
        raise ValueError('Train/validation video overlap; expected official disjoint splits')
    runtime_hashes = materialize_runtime(output / 'runtime')
    write_json(output / 'train.json', train)
    write_json(output / 'val.json', val)
    config = recipe(output / 'train.json', output / 'val.json', video_root, Path(umt).resolve(),
                    Path(vicuna).resolve(), Path(stage3).resolve(), output / 'training', profile, method, seed, world_size)
    write_json(output / 'config.json', config)
    videos = sorted({r['video'] for r in train + val})
    write_json(output / 'contract.json', {'upstream_commit': provenance['commit'], 'profile': profile, 'method': method,
        'seed': seed, 'world_size': world_size, 'train_count': len(train), 'val_count': len(val),
        'runtime_hashes': runtime_hashes, 'config_sha256': file_hash(output / 'config.json'),
        'input_hashes': {str(Path(p).resolve()): file_hash(Path(p)) for p in (train_csv, val_csv, umt, stage3)},
        'annotation_hashes': {name: file_hash(output / name) for name in ('train.json', 'val.json')},
        'videos': {v: file_hash(video_root / v) for v in videos},
        'vicuna_files': {p.relative_to(Path(vicuna)).as_posix(): file_hash(p) for p in Path(vicuna).rglob('*')
                          if p.is_file() and '.cache' not in p.parts},
        'instruction_policy': 'First five author prompts assigned cyclically to train rows; first prompt for validation.'})
    return output


def verify_prepared(output):
    output = Path(output)
    contract = json.loads((output / 'contract.json').read_text())
    config = json.loads((output / 'config.json').read_text())
    if file_hash(output / 'config.json') != contract['config_sha256']:
        raise ValueError('Prepared config changed')
    for name, h in contract['runtime_hashes'].items():
        if file_hash(output / 'runtime' / name) != h:
            raise ValueError('Prepared runtime changed: ' + name)
    for name, h in contract['annotation_hashes'].items():
        if file_hash(output / name) != h:
            raise ValueError('Prepared annotations changed')
    for name, h in contract['input_hashes'].items():
        if file_hash(Path(name)) != h:
            raise ValueError('Prepared input changed: ' + name)
    for name, h in contract['videos'].items():
        if file_hash(Path(config['train_file'][1]) / name) != h:
            raise ValueError('Prepared video changed: ' + name)
    for name, h in contract['vicuna_files'].items():
        if file_hash(Path(config['model']['llama_model_path']) / name) != h:
            raise ValueError('Vicuna checkpoint changed: ' + name)
    return config, contract
