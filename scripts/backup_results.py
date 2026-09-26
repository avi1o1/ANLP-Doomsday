"""Build one size-capped results archive, excluding reconstructible vector caches.

No transfer or deletion is performed here. The full run stays on scratch. Included
manifests describe original artifacts; omitted vector files must be regenerated
before resuming encoding/indexing from an extracted archive.
"""
import argparse
import gzip
import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FULL = {'evaluations', 'baseline_evaluations', 'bases', 'predictor', 'analysis', 'report', 'jobs', 'external', 'transfer'}
METADATA = {'manifest.json', 'statistics.json', 'diagnostics.json', 'shards.json', 'encoding_metrics.json'}


def include(path):
    if path.name == '.lock' or path.name.startswith('.') or path.suffix == '.tmp':
        return False
    top = path.parts[0]
    if top in FULL:
        return True
    if top in {'indexes', 'encoded'}:
        return path.name in METADATA
    if top == 'datasets':
        return path.suffix == '.json' or path.name in {'queries.jsonl', 'qrels.jsonl'}
    if top == 'sample':
        return path.name == 'manifest.json' or path.name.endswith('_ids.jsonl')
    return len(path.parts) == 1 and path.suffix in {'.json', '.yaml', '.yml'}


def create_archive(root, project, target, max_bytes, workers=8):
    root, project, target = root.resolve(), project.resolve(), target.resolve()
    files = sorted(p for p in root.rglob('*') if p.is_file() and not p.is_symlink() and include(p.relative_to(root)))
    if not files:
        raise ValueError('No research results selected')
    code = sorted(p for directory in ('src', 'scripts', 'configs') for p in (project / directory).rglob('*')
                  if p.is_file() and p.suffix in {'.py', '.yaml', '.yml', '.toml'} and '__pycache__' not in p.parts)
    code += [p for p in (project / 'pyproject.toml', project / 'uv.lock', project / 'README.md') if p.is_file()]
    code += sorted(project.glob('*.sbatch'))
    counts = {}
    for p in files:
        key = p.relative_to(root).parts[0]
        counts[key] = counts.get(key, 0) + p.stat().st_size
    print(json.dumps({'selected_files': len(files), 'uncompressed_bytes_by_group': counts, 'max_archive_bytes': max_bytes}), flush=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.building')
    total, checksum, tick = 0, hashlib.sha256(), time.monotonic()
    process = None
    try:
        with tempfile.TemporaryDirectory(dir=target.parent) as work, ThreadPoolExecutor(max_workers=workers) as pool:
            work = Path(work)
            run_list, code_list = work / 'run-files', work / 'code-files'
            run_list.write_bytes(b''.join(os.fsencode(str(p.relative_to(root.parent))) + b'\0' for p in files))
            code_list.write_bytes(b''.join(os.fsencode(str(p.relative_to(project.parent))) + b'\0' for p in code))
            process = subprocess.Popen(['tar', '-cf', '-', '-C', str(root.parent), '--null', '-T', str(run_list),
                                        '-C', str(project.parent), '--null', '-T', str(code_list)], stdout=subprocess.PIPE)
            pending = deque()
            with temporary.open('wb') as output:
                def emit():
                    nonlocal total
                    data = pending.popleft().result()
                    total += len(data)
                    if total > max_bytes:
                        raise ValueError(f'Archive exceeds {max_bytes} bytes; no backup was transferred')
                    output.write(data)
                    checksum.update(data)
                while block := process.stdout.read(8 * 2**20):
                    pending.append(pool.submit(gzip.compress, block, compresslevel=1, mtime=0))
                    if len(pending) >= workers * 2:
                        emit()
                while pending:
                    emit()
                output.flush()
                os.fsync(output.fileno())
            if process.wait() != 0:
                raise RuntimeError('tar failed; keeping original results unchanged')
            os.replace(temporary, target)
        result = {'archive': str(target), 'bytes': total, 'sha256': checksum.hexdigest(),
                  'seconds': time.monotonic()-tick, 'selected_files': len(files),
                  'uncompressed_bytes_by_group': counts, 'max_archive_bytes': max_bytes}
        target.with_name(target.name + '.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        return result
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
            process.wait()
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_root', type=Path)
    parser.add_argument('--project', type=Path, default=Path.cwd())
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--max-bytes', type=int, default=40_000_000_000)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1 or args.max_bytes < 1:
        parser.error('workers and max-bytes must be positive')
    create_archive(args.output_root, args.project, args.archive, args.max_bytes, args.workers)


if __name__ == '__main__':
    main()
