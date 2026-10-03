"""Plan/download pinned VideoMME chunks and extract required local videos only."""
import argparse
from pathlib import Path
import shutil
import zipfile

from qa_protocol import atomic_json, read, video_path


def extract_required(archive, destination, wanted):
    """Flatten only known media basenames; never extract ZIP paths/symlinks."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            name = Path(member.filename.replace('\\', '/')).name
            stem = Path(name).stem
            if member.is_dir() or stem not in wanted or Path(name).suffix.lower() not in ('.mp4', '.mkv'):
                continue
            if shutil.disk_usage(destination).free < member.file_size + 1024**3:
                raise RuntimeError('Insufficient free space for extraction (quota may impose a lower limit)')
            target = destination / name
            if target.exists():
                raise ValueError(f'Conflicting media filename: {target}')
            temporary = target.with_suffix(target.suffix + '.part')
            with z.open(member) as source, temporary.open('wb') as output:
                shutil.copyfileobj(source, output, length=1024*1024)
            temporary.replace(target)  # ZIP CRC verified by reading through EOF.
            extracted.append(name)
            wanted.remove(stem)
    return extracted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, help='writable dataset root; videos stored in data/')
    p.add_argument('--annotations', default='datasets/videomme/include_frame_idx.json')
    p.add_argument('--limit', type=int, help='first N questions, same as compare_selectors --limit')
    p.add_argument('--revision', default='main')
    p.add_argument('--download', action='store_true', help='without this flag only metadata/size plan is fetched')
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error('--limit must be positive')
    rows = read(args.annotations)
    rows = rows[:args.limit] if args.limit else rows
    wanted = set()
    for row in rows:
        try:
            video_path(row, 'videomme', args.root)
        except FileNotFoundError:
            wanted.add(str(row['videoID']))
    if not wanted:
        print('All required videos already present.')
        return
    from huggingface_hub import HfApi, hf_hub_download
    repo = 'lmms-eval/Video-MME'
    root = Path(args.root).resolve()
    previous = root / 'download_plan.json'
    revision = args.revision
    if previous.exists() and args.revision == 'main':
        revision = read(previous)['revision']  # Resume the same dataset version.
    info = HfApi().dataset_info(repo, revision=revision, files_metadata=True)
    chunks = sorted((f for f in info.siblings if f.rfilename.startswith('videos_chunked_') and f.rfilename.endswith('.zip')),
                    key=lambda f: f.rfilename)
    if not chunks or any(f.size is None for f in chunks):
        raise RuntimeError('Unexpected dataset archive layout/size metadata')
    plan = {'repo': repo, 'revision': info.sha, 'questions': len(rows), 'missing_video_ids': sorted(wanted),
            'archives': [{'name': f.rfilename, 'bytes': f.size} for f in chunks],
            'worst_case_download_bytes': sum(f.size for f in chunks),
            'note': 'Subset media may be in any chunk; finding them can require downloading all archives. Archives retained for resume.'}
    atomic_json(previous, plan)
    print(f"Missing {len(wanted)} videos; worst-case archives {plan['worst_case_download_bytes']/1024**3:.1f} GiB. "
          'See download_plan.json; filesystem free space is not your account quota.')
    if not args.download:
        print('Plan only. Add --download to download/extract; no model downloaded here.')
        return
    downloads = root / 'archives'
    downloads.mkdir(exist_ok=True)
    for chunk in chunks:
        local = downloads / chunk.rfilename
        if not local.is_file() and shutil.disk_usage(downloads).free < chunk.size + 1024**3:
            raise RuntimeError('Insufficient filesystem free space; use approved storage with sufficient quota')
        archive = hf_hub_download(repo, chunk.rfilename, repo_type='dataset', revision=info.sha, local_dir=downloads)
        extract_required(archive, root / 'data', wanted)
        if not wanted:
            break
    if wanted:
        raise RuntimeError(f'Media not found in pinned archives: {sorted(wanted)}')
    atomic_json(root / 'download_complete.json', {'repo': repo, 'revision': info.sha, 'questions': len(rows)})
    print('Required video subset ready.')


if __name__ == '__main__':
    main()
