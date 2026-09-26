import tarfile

import pytest

from scripts.backup_results import create_archive, include


def test_archive_selection_roundtrip_and_hard_cap(tmp_path):
    root = tmp_path / 'research-v5'
    files = {'evaluations/a/outcomes.npy': b'outcomes', 'evaluations/a/000/00000000.json': b'rankings',
             'predictor/predictor.json': b'predictor', 'evaluations/a/large.json': b'x' * (9 * 2**20), 'encoded/x/vectors.npz': b'large vectors',
             'indexes/x/statistics.json': b'stats', 'datasets/corpora/docs.jsonl': b'corpus',
             'sample/train_ids.jsonl': b'sample IDs', '.env': b'NEVER INCLUDE'}
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    project = tmp_path / 'project'
    (project / 'src').mkdir(parents=True)
    (project / 'src/a.py').write_text('code')
    (project / '.env').write_text('SECRET')
    target = tmp_path / 'backup.tar.gz'
    result = create_archive(root, project, target, 1_000_000, 2)
    assert result['bytes'] == target.stat().st_size
    with tarfile.open(target) as archive:
        names = archive.getnames()
        for name, data in files.items():
            if include(__import__('pathlib').Path(name)):
                assert archive.extractfile('research-v5/' + name).read() == data
            else:
                assert 'research-v5/' + name not in names
        assert 'project/.env' not in names
    with pytest.raises(ValueError, match='exceeds'):
        create_archive(root, project, tmp_path / 'too-small.tar.gz', 1, 2)
    assert not (tmp_path / 'too-small.tar.gz').exists()
    assert not (tmp_path / 'too-small.tar.gz.building').exists()
