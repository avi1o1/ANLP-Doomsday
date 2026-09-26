from pathlib import Path

import pytest

from scripts.resume_scheduler import migrate
from src.artifacts import atomic_json, file_hash, provenance, read_json, write_jsonl
from src.config import digest


def test_launcher_only_migration_preserves_batches_and_rejects_scientific_changes(tmp_path):
    source = Path('src').resolve()
    previous = {'source': {str(p.relative_to(source)): file_hash(p) for p in source.rglob('*.py')},
                'versions': provenance()['versions']}
    previous['source']['launch.py'] = 'previous-launcher-hash'
    parents = []
    for name in ('index', 'queries', 'collection'):
        folder = tmp_path / name
        atomic_json(folder / 'manifest.json', {'status': 'complete', 'fingerprint': name, 'outputs': []})
        parents.append({'path': str(folder), 'fingerprint': name, 'outputs_hash': digest([])})
    atomic_json(tmp_path / 'queries/shards.json', [{'path': str(tmp_path / 'queries')}])
    atomic_json(tmp_path / 'queries/ids.json', ['q1', 'q2'])
    write_jsonl(tmp_path / 'collection/qrels.jsonl', [{'query_id': 'q1'}, {'query_id': 'q2'}])
    directory = tmp_path / 'evaluations/test'
    original = {'status': 'interrupted', 'config': {'budget': 32}, 'parents': parents,
                'provenance': {**provenance(), 'source_hash': digest(previous['source'])}}
    atomic_json(directory / 'manifest.json', original)
    atomic_json(directory / '000/00000000.json', [{'query_id': 'q1'}, {'query_id': 'q2'}])
    checksum = file_hash(directory / '000/00000000.json')
    bad = {**previous, 'source': {**previous['source'], 'scoring.py': 'changed'}}
    with pytest.raises(ValueError, match='scientific source'):
        migrate(tmp_path, bad)
    assert migrate(tmp_path, previous) == [str(directory)]
    assert file_hash(directory / '000/00000000.json') == checksum
    assert read_json(directory / 'scheduler_migration.json')['original_manifest'] == original
    assert read_json(directory / 'manifest.json')['provenance'] == provenance()
