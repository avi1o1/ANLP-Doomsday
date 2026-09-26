import io
import json
import tarfile
from pathlib import Path

import pytest

from scripts import apply_analysis_update as update


def test_deferred_install_requires_completed_gpu_and_exact_source(tmp_path, monkeypatch):
    project = tmp_path / 'project'
    old = {'analysis.py': b'old analysis\n', 'retrieval/pipeline.py': b'old pipeline\n'}
    new = {p: b'new ' + data for p, data in old.items()}
    for name, data in old.items():
        path = project / 'src' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = {'before': {p: update.sha(v) for p, v in old.items()},
                'payload': {p: update.sha(v) for p, v in new.items()}, 'required_gpu_job': '123'}
    bundle = tmp_path / 'bundle.tar.gz'
    with tarfile.open(bundle, 'w:gz') as archive:
        for name, data in {'manifest.json': json.dumps(manifest).encode(),
                           **{'src/' + p: v for p, v in new.items()}}.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
    monkeypatch.setattr(update, 'gpu_completed', lambda _: False)
    with pytest.raises(RuntimeError, match='not completed'):
        update.apply_bundle(project, bundle, tmp_path / 'output')
    assert (project / 'src/analysis.py').read_bytes() == old['analysis.py']
    monkeypatch.setattr(update, 'gpu_completed', lambda _: True)
    (project / 'src/analysis.py').write_bytes(b'unexpected change')
    with pytest.raises(ValueError, match='Source changed'):
        update.apply_bundle(project, bundle, tmp_path / 'output')
    (project / 'src/analysis.py').write_bytes(old['analysis.py'])
    update.apply_bundle(project, bundle, tmp_path / 'output')
    update.apply_bundle(project, bundle, tmp_path / 'output')  # Idempotent retry.
    for name, data in new.items():
        assert (project / 'src' / name).read_bytes() == data
    history = next((tmp_path / 'output/analysis/deployment').glob('*/before'))
    assert (history / 'analysis.py').read_bytes() == old['analysis.py']
    assert Path(history.parent / 'update.json').exists()
