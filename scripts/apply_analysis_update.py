"""Install a reviewed analysis-only bundle after its GPU prerequisite completes."""
import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ALLOWED = {'analysis.py', 'retrieval/pipeline.py'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def gpu_completed(job):
    states = subprocess.check_output(['sacct', '-X', '-j', str(job), '--format=State',
                                      '--noheader', '--parsable2'], text=True).splitlines()
    return bool(states) and all(state.strip() == 'COMPLETED' for state in states)


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def apply_bundle(project, bundle, output):
    with tarfile.open(bundle, 'r:gz') as archive:
        manifest = json.load(archive.extractfile('manifest.json'))
        if set(manifest['payload']) != ALLOWED:
            raise ValueError('Only the reviewed analysis files may change')
        payload = {name: archive.extractfile('src/' + name).read() for name in ALLOWED}
    if any(sha(payload[name]) != checksum for name, checksum in manifest['payload'].items()):
        raise ValueError('Bundle payload checksum mismatch')
    if not gpu_completed(manifest['required_gpu_job']):
        raise RuntimeError('GPU prerequisite has not completed successfully; source left untouched')
    source = project / 'src'
    current = {str(p.relative_to(source)): sha(p.read_bytes()) for p in source.rglob('*.py')}
    before = manifest['before']
    after = {**before, **manifest['payload']}
    if set(current) != set(before) or any(current[p] not in {before[p], after[p]} for p in current):
        raise ValueError('Source changed since staging; refusing to overwrite it')
    record_dir = output / 'analysis' / 'deployment' / sha(bundle.read_bytes())
    for name in sorted(ALLOWED):
        path = source / name
        if current[name] == before[name]:
            atomic_write(record_dir / 'before' / name, path.read_bytes())
        atomic_write(path, payload[name])
    record = {**manifest, 'installed_utc': datetime.now(timezone.utc).isoformat(),
              'bundle_sha256': sha(bundle.read_bytes()), 'after': after}
    atomic_write(record_dir / 'update.json', (json.dumps(record, sort_keys=True, indent=2) + '\n').encode())
    print(f'Installed validated analysis update; provenance: {record_dir / "update.json"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--project', type=Path, default=Path.cwd())
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    apply_bundle(args.project.resolve(), args.bundle.resolve(), args.output_root.resolve())


if __name__ == '__main__':
    main()
