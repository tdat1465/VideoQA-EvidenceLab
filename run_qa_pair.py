"""Strict-frame Qwen2.5-VL ablation, one GPU, atomic per-question resume."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time

from qa_protocol import SELECTORS, atomic_json, decode_selected, digest, identity, load_pair, parse_answer, prepare_inputs, prompt, read, video_path


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', required=True)
    p.add_argument('--video-root', required=True)
    p.add_argument('--model', default='Qwen/Qwen2.5-VL-3B-Instruct')
    p.add_argument('--revision', default='main', help='resolved and pinned to Hub commit before loading')
    p.add_argument('--max-pixels', type=int, default=256 * 28 * 28)
    p.add_argument('--max-new-tokens', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--offline', action='store_true', help='cached checkpoint only; requires pinned --revision SHA (or resume)')
    p.add_argument('--preflight-only', action='store_true', help='validate annotations/videos without downloading weights')
    return p.parse_args()


def main(args):
    manifest, rows = load_pair(args.experiment)
    dataset = manifest['arguments']['dataset_name']
    if args.max_pixels < 4 * 28 * 28 or args.max_new_tokens < 1:
        raise ValueError('Invalid pixel/token budget')
    paths = [video_path(row, dataset, args.video_root) for row in rows['original']]
    if args.preflight_only:
        print(f'OK: {len(paths)} paired questions, {len(set(paths))} local videos. No weights loaded.')
        return
    import torch
    from decord import VideoReader, cpu
    from filelock import FileLock
    from huggingface_hub import HfApi
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: run inside an allocated GPU job, not the login node')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Allocate exactly one visible GPU for this runner')
    output = Path(args.experiment).resolve() / 'qa_pair'
    output.mkdir(exist_ok=True)
    with FileLock(str(output / '.lock'), timeout=0):
        config_path = output / 'config.json'
        revision = args.revision
        if args.resume and revision == 'main' and config_path.exists():
            revision = read(config_path)['model_revision']
        if args.offline and not re.fullmatch(r'[0-9a-fA-F]{40}', revision):
            raise ValueError('--offline requires a 40-character checkpoint commit SHA in --revision')
        resolved_revision = revision if args.offline else HfApi().model_info(args.model, revision=revision).sha
        config = {k: v for k, v in vars(args).items() if k not in ('resume', 'preflight_only', 'revision', 'offline')}
        config.update(model_revision=resolved_revision,
                      selection_manifest_sha256=digest(Path(args.experiment) / 'manifest.json'),
                      annotation_hashes={s: digest(Path(args.experiment) / s / 'annotations.json') for s in SELECTORS},
                      code={name: digest(Path(__file__).parent / name) for name in ('run_qa_pair.py', 'qa_protocol.py')},
                      versions={name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'numpy', 'decord', 'pillow')},
                      video_files=[{'path': str(p), 'size': p.stat().st_size, 'sha256': digest(p)} for p in dict.fromkeys(paths)],
                      dtype='bfloat16' if torch.cuda.is_bf16_supported() else 'float16',
                      gpu=torch.cuda.get_device_name(0), model_fps=1.0, protocol='video-only-strict-mcq-v1')
        if config_path.exists():
            if not args.resume or read(config_path) != config:
                raise ValueError('Existing QA run: use --resume with identical inputs/model/code/environment')
        else:
            if args.resume or list(output.glob('predictions/*.json')):
                raise ValueError('No valid run configuration for resume')
            atomic_json(config_path, config)
        try:
            driver = subprocess.check_output(['nvidia-smi'], text=True)
        except (OSError, subprocess.CalledProcessError):
            driver = 'nvidia-smi unavailable'
        atomic_json(output / 'environment.json', {'python': platform.python_version(), 'host': platform.node(),
                    'cuda': torch.version.cuda, 'nvidia_smi': driver})
        for saved in (output / 'predictions').glob('*.json'):
            pair = read(saved)
            i = pair['index']
            if (isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(paths)
                    or saved.name != f'{i:06d}.json'):
                raise ValueError('Invalid saved prediction index')
            qid, vid, gold, _ = identity(rows['original'][i], dataset)
            if (pair['question_id'], pair['video_id'], pair['gold']) != (qid, vid, gold) or any(
                    pair[s]['status'] != 'ok' or pair[s]['frame_ids'] != rows[s][i]['frame_idx'] for s in SELECTORS):
                raise ValueError('Saved prediction does not match paired inputs')
        pending = [i for i in range(len(paths)) if not (output / 'predictions' / f'{i:06d}.json').exists()]
        if not pending:
            print('All pairs already complete. Run analyze_qa_pair.py.')
            return
        set_seed(args.seed)
        dtype = torch.bfloat16 if config['dtype'] == 'bfloat16' else torch.float16
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, revision=config['model_revision'],
                    torch_dtype=dtype, attn_implementation='sdpa', local_files_only=args.offline).to('cuda').eval()
        processor = AutoProcessor.from_pretrained(args.model, revision=config['model_revision'],
                    min_pixels=4 * 28 * 28, max_pixels=args.max_pixels, local_files_only=args.offline)

        def infer(row, path, index):
            start = time.perf_counter()
            decoder = VideoReader(str(path), ctx=cpu(0), num_threads=1)
            ids = row['frame_idx']
            frames, fps = decode_selected(decoder, ids)
            decode_seconds = time.perf_counter() - start
            messages = [{'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text': prompt(row, dataset)}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs, grid = prepare_inputs(processor, frames, text)
            inputs = inputs.to('cuda')
            # The processor may pad an odd count by repeating the final frame;
            # it must never resample or introduce another decoded frame ID.
            set_seed(args.seed + index)
            torch.cuda.synchronize()
            preprocessing_seconds = time.perf_counter() - start - decode_seconds
            torch.cuda.reset_peak_memory_stats()
            generation_start = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens)
            torch.cuda.synchronize()
            generation_seconds = time.perf_counter() - generation_start
            response_ids = generated[:, inputs.input_ids.shape[1]:]
            response = processor.batch_decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            _, _, gold, choices = identity(row, dataset)
            letter = parse_answer(response, len(choices))
            return {'response': response, 'parsed_answer': letter, 'correct': letter == gold,
                    'frame_ids': ids, 'timestamps_seconds': [i / fps for i in ids], 'video_fps': fps,
                    'video_grid_thw': grid, 'input_tokens': inputs.input_ids.shape[1],
                    'output_tokens': response_ids.shape[1], 'decode_seconds': decode_seconds,
                    'preprocessing_seconds': preprocessing_seconds, 'generation_seconds': generation_seconds,
                    'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(), 'status': 'ok'}

        for index in pending:
            row = rows['original'][index]
            qid, vid, gold, choices = identity(row, dataset)
            pair = {'index': index, 'question_id': qid, 'video_id': vid, 'gold': gold, 'question': row['question'],
                    'choices': choices, 'duration_group': row.get('duration', 'unknown'),
                    'task_type': row.get('task_type', row.get('question_category', 'unknown'))}
            # Alternate order; the first pair still includes kernel warmup costs.
            order = SELECTORS if index % 2 == 0 else SELECTORS[::-1]
            try:
                for selector in order:
                    pair[selector] = infer(rows[selector][index], paths[index], index)
                if pair['original']['video_grid_thw'] != pair['watershed']['video_grid_thw']:
                    raise RuntimeError('Paired visual token grids differ')
            except Exception as error:
                atomic_json(output / 'last_error.json', {'index': index, 'type': type(error).__name__, 'message': str(error)})
                raise  # No silent skipping or invented answer on OOM/decode failure.
            atomic_json(output / 'predictions' / f'{index:06d}.json', pair)
            print(f'{index+1}/{len(paths)} {qid}: original={pair["original"]["parsed_answer"]} watershed={pair["watershed"]["parsed_answer"]}', flush=True)
        print('QA complete. Run python analyze_qa_pair.py --experiment ' + args.experiment)


if __name__ == '__main__':
    main(arguments())
