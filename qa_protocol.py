"""CPU-only paired QA validation, prompts and deterministic MCQ grading."""
import hashlib
import json
import math
import re
from pathlib import Path

SELECTORS = ('original', 'watershed')


def decode_selected(decoder, ids):
    if not ids or ids[-1] >= len(decoder):
        raise ValueError('Selected frame outside video length; check cache/video version')
    fps = float(decoder.get_avg_fps())
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid decoded FPS')
    return decoder.get_batch(ids).asnumpy(), fps


def prepare_inputs(processor, frames, text):
    # Feed the already decoded array, never a file path or an autosampling utility.
    inputs = processor(text=[text], videos=[frames], fps=1.0, return_tensors='pt')
    grid = inputs['video_grid_thw'].tolist()
    patch = processor.image_processor.temporal_patch_size
    if len(grid) != 1 or grid[0][0] != (len(frames) + patch - 1) // patch:
        raise RuntimeError('Processor unexpectedly changed temporal frame count')
    return inputs, grid


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def identity(row, dataset):
    if dataset == 'videomme':
        return str(row['question_id']), str(row['videoID']), row['answer'].strip().upper(), row['options']
    choices = row['candidates']
    index = row['correct_choice']
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(choices):
        raise ValueError('Invalid LongVideoBench ground truth')
    return str(row['id']), str(row['video_id']), chr(65 + index), choices


def parse_answer(text, nchoices):
    # Refuse explanations/ambiguous answers rather than randomly guessing.
    match = re.fullmatch(r'\s*(?:(?:the best answer is|answer|option)\s*[:：]?\s*)?[\(\[]?([A-Z])[\)\]]?[.。]?\s*', text, re.I)
    letter = match.group(1).upper() if match else None
    return letter if letter and 0 <= ord(letter) - 65 < nchoices else None


def prompt(row, dataset):
    if dataset == 'videomme':
        return ('Select the best answer to the following multiple-choice question based on the video and the subtitles. '
                'Respond with only the letter (A, B, C, or D) of the correct option.\n'
                + row['question'] + '\n' + '\n'.join(row['options'])
                + "\n\nAnswer with the option's letter from the given choices directly.")
    return (row['question'] + '\n' + '\n'.join(f'{chr(65+i)}. {choice}' for i, choice in enumerate(row['candidates']))
            + "\nAnswer with the option's letter from the given choices directly.\n")


def video_path(row, dataset, root):
    root = Path(root).resolve()
    if dataset == 'videomme':
        relative = Path('data') / (str(row['videoID']) + '.mp4')
    else:
        relative = Path('videos') / row['video_path']
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Video path escapes dataset root')
    alternatives = [path] if dataset != 'videomme' else [path, path.with_suffix('.MP4'), path.with_suffix('.mkv')]
    for candidate in alternatives:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'Missing video: {path}')


def load_pair(experiment):
    root = Path(experiment)
    manifest = read(root / 'manifest.json')
    dataset = manifest['arguments']['dataset_name']
    rows = {s: read(root / s / 'annotations.json') for s in SELECTORS}
    expected = manifest['query_count']
    if expected < 1 or any(len(rows[s]) != expected for s in SELECTORS):
        raise ValueError('Annotation counts disagree with manifest')
    seen = set()
    for index, (a, b) in enumerate(zip(rows['original'], rows['watershed'])):
        clean = lambda row: {k: v for k, v in row.items() if k != 'frame_idx'}
        if clean(a) != clean(b):
            raise ValueError(f'Paired question/labels/config differ at query {index}')
        qid, _, gold, choices = identity(a, dataset)
        if qid in seen or not choices or gold not in [chr(65+i) for i in range(len(choices))]:
            raise ValueError(f'Invalid or duplicate question {qid}')
        seen.add(qid)
        for row in (a, b):
            ids = row['frame_idx']
            if (not ids or len(ids) > manifest['arguments']['max_num_frames']
                    or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids)
                    or ids != sorted(set(ids))):
                raise ValueError(f'Invalid strict frame selection for {qid}')
        if len(a['frame_idx']) != len(b['frame_idx']):
            raise ValueError(f'Actual frame counts differ for {qid}')
    return manifest, rows
