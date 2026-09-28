import numpy as np

from src.analysis import ablation_task, fit_predictors, paired_effect, parallel_map
from src.artifacts import StopFlag, read_json
from src.config import load_config
from src.diagnostics import ALL_FEATURES
from src.retrieval import pipeline


def rows_fixture():
    rng = np.random.default_rng(20)
    rows = []
    for corpus, language in (('a', 'en'), ('b', 'en'), ('c', 'en'), ('h', 'hi'), ('n', 'bn'), ('t', 'te')):
        for family in ('linear', 'clustered', 'quantized', 'learned'):
            x = rng.uniform(size=6)
            rows.append(dict(row_id=str(len(rows)), family=family, corpus_id=corpus, language=language,
                representation_id=family, collection=corpus, budget='primary', seed=0, primary_budget=True,
                diagnostics=dict(zip(ALL_FEATURES, x)), margin=float(2*x[1]-x[4]+x[5]-.7), margin_variance=.1))
    return rows


def test_parallel_validation_is_identical_to_serial(tmp_path, monkeypatch):
    rows = rows_fixture()
    monkeypatch.setenv('CSX_ANALYSIS_WORKERS', '1')
    model, serial = fit_predictors(rows, tmp_path / 'serial', samples=20)
    monkeypatch.setenv('CSX_ANALYSIS_WORKERS', '2')
    other, parallel = fit_predictors(rows, tmp_path / 'parallel', samples=20)
    assert serial == parallel
    assert model == other
    for name in ('predictor.json', 'english_predictor.json', 'validation.json'):
        assert (tmp_path / 'serial' / name).read_bytes() == (tmp_path / 'parallel' / name).read_bytes()


def test_parallel_ablations_preserve_bootstrap_stream_and_order(tmp_path, monkeypatch):
    rng = np.random.default_rng(20)
    jobs = []
    for i in range(4):
        path = tmp_path / f'{i}.npy'
        values = rng.random((32, 8)) if i else np.ones((32, 8))
        np.save(path, values)
        jobs.append((str(path), str(i), 30, True))
    serial = list(parallel_map(ablation_task, jobs))
    monkeypatch.setenv('CSX_ANALYSIS_WORKERS', '2')
    assert serial == list(parallel_map(ablation_task, jobs))
    for job, payload in zip(jobs, serial):
        values = np.load(job[0])
        # The row effect comes from the same outcomes, at the same sample count.
        row_reference = paired_effect(values, 30)
        assert all(payload['effect'][k] == row_reference[k]
                   for k in ('raw_delta', 'margin', 'raw_ci95', 'p_value'))
        for bit, effect in enumerate(payload['ablations']):
            off = [i for i in range(8) if f'{i:03b}'[bit] == '0']
            on = [i for i in range(8) if f'{i:03b}'[bit] == '1']
            comparison = np.repeat(values[:, off].mean(axis=1)[:, None], 8, axis=1)
            comparison[:, -1] = values[:, on].mean(axis=1)
            ref = paired_effect(comparison, 30)
            assert all(effect[k] == ref[k] for k in ('raw_delta', 'raw_ci95', 'p_value'))


def test_diagnostic_cache_reuses_and_invalidates_outcomes(tmp_path, monkeypatch):
    config = load_config('configs/fixture.yaml')
    config['output_root'] = str(tmp_path / 'run')
    with StopFlag() as stop:
        for dataset in ('train', 'english'):
            pipeline.prepare(config, dataset, stop)
        pipeline.encode_sample(config, stop)
        pipeline.fit_basis(config, 'identity', 'pooled', 0, stop)
        pipeline.encode_corpus(config, 'english', 'identity', 'pooled', 0, stop)
        pipeline.build_index(config, 'english', 'identity', 'pooled', 0, 'small', stop)
        evaluation = pipeline.evaluate(config, 'english', 'identity', 'pooled', 0, 'small', stop)
    rows = pipeline.diagnose(config)
    directory = pipeline.root(config) / 'analysis'
    before = (directory / 'component_ablations.json').read_bytes()
    stamp = (directory / 'component_ablations.json').stat().st_mtime_ns
    assert pipeline.diagnose(config) == rows
    assert (directory / 'component_ablations.json').stat().st_mtime_ns == stamp
    # Changed inputs select a fresh content-addressed analysis directory.
    values = np.load(evaluation / 'outcomes.npy')
    values[:, -1] += .1
    np.save(evaluation / 'outcomes.npy', values)
    pipeline.diagnose(config)
    assert len(list((directory / 'diagnostics').glob('*/manifest.json'))) == 2
    assert (directory / 'component_ablations.json').read_bytes() != before
    assert all(read_json(p)['status'] == 'complete' for p in (directory / 'diagnostics').glob('*/manifest.json'))


def test_interrupted_diagnostics_reuse_completed_rows(tmp_path, monkeypatch):
    import pytest

    from src.artifacts import StopRequested
    config = load_config('configs/fixture.yaml')
    config['output_root'] = str(tmp_path / 'run')
    with StopFlag() as stop:
        for dataset in ('train', 'english'):
            pipeline.prepare(config, dataset, stop)
        pipeline.encode_sample(config, stop)
        for basis in ('identity', 'gaussian'):
            pipeline.fit_basis(config, basis, 'pooled', 0, stop)
            pipeline.encode_corpus(config, 'english', basis, 'pooled', 0, stop)
            pipeline.build_index(config, 'english', basis, 'pooled', 0, 'small', stop)
            pipeline.evaluate(config, 'english', basis, 'pooled', 0, 'small', stop)
    original = pipeline.parallel_map
    def interrupt(function, jobs):
        yield function(jobs[0])
        raise StopRequested()
    monkeypatch.setattr(pipeline, 'parallel_map', interrupt)
    with pytest.raises(StopRequested):
        pipeline.diagnose(config)
    cached = list((pipeline.root(config) / 'analysis/diagnostics').glob('*/ablation_rows/*.json'))
    assert len(cached) == 1
    stamp = cached[0].stat().st_mtime_ns
    monkeypatch.setattr(pipeline, 'parallel_map', original)
    pipeline.diagnose(config)
    assert cached[0].stat().st_mtime_ns == stamp
    assert len(read_json(pipeline.root(config) / 'analysis/component_ablations.json')) == 6
