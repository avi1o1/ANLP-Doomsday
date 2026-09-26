import copy
from pathlib import Path

import numpy as np
from scipy import sparse

from src.artifacts import StopFlag, read_json
from src.config import load_config
from src.experiments import experiment_rows, write_manifest
from src.reporting import report
from src.retrieval import pipeline
from src.retrieval.baselines import baseline


def test_offline_pipeline_resume_shards_baselines_and_report(tmp_path):
    config = load_config("configs/fixture.yaml")
    config["output_root"] = str(tmp_path / "run")
    config["bases"] = config["bases"][:1]
    config["granularities"] = ["pooled"]
    with StopFlag() as stop:
        pipeline.prepare(config, "train", stop)
        pipeline.prepare(config, "english", stop)
        pipeline.encode_sample(config, stop)
        pipeline.fit_basis(config, "identity", "pooled", 0, stop)
        for shard in (1, 0):
            pipeline.encode_corpus(config, "english", "identity", "pooled", 0, stop, shard, 2)
        pipeline.encode_corpus(config, "english", "identity", "pooled", 0, stop)
        encoded = pipeline.encoded_path(config, "english", "identity", "pooled", 0, "documents")
        shards = read_json(encoded / "shards.json")
        matrix = sparse.vstack([sparse.load_npz(Path(s["path"]) / "vectors.npz") for s in shards])
        assert matrix.shape[0] == 30
        pipeline.build_index(config, "english", "identity", "pooled", 0, "small", stop)
        path = pipeline.evaluate(config, "english", "identity", "pooled", 0, "small", stop)
        before = (path / "result.json").stat().st_mtime_ns
        pipeline.evaluate(config, "english", "identity", "pooled", 0, "small", stop)
        assert (path / "result.json").stat().st_mtime_ns == before
        outcomes = np.load(path / "outcomes.npy")
        assert outcomes.shape == (6, 8) and np.isfinite(outcomes).all()
        for method in ("bm25", "dense"):
            baseline(config, "english", method, stop)
    directory = report(config)
    assert (directory / "retrieval.csv").exists()
    assert read_json(directory / "inventory.json")["completed_retrieval_rows"] == 1
    sample = pipeline.root(config) / "sample"
    train = {r["doc_id"] for r in __import__("src.artifacts", fromlist=["jsonl"]).jsonl(sample / "train_ids.jsonl")}
    val = {r["doc_id"] for r in __import__("src.artifacts", fromlist=["jsonl"]).jsonl(sample / "validation_ids.jsonl")}
    assert not train & val


def test_dense_query_batch_size_covers_every_query(tmp_path):
    config = load_config("configs/fixture.yaml")
    config["output_root"] = str(tmp_path)
    config["dense_search"]["query_batch_size"] = 4
    config["evaluation_batch_size"] = 2
    with StopFlag() as stop:
        pipeline.prepare(config, "english", stop)
        path = baseline(config, "english", "dense", stop)
        assert read_json(path / "result.json")["examples"] == 6
        assert len(read_json(path / "batch-00000000.json")) == 4
        assert len(read_json(path / "batch-00000004.json")) == 2
        baseline(config, "english", "dense", stop)


def test_manifest_counts_groups_and_resources(tmp_path):
    config = load_config("configs/research.yaml")
    rows = experiment_rows(config)
    assert len(rows) == 7 * 2 * len([d for d in config["datasets"].values() if d["role"] == "evaluate"])
    assert {r["corpus_id"] for r in rows if r["collection"].startswith("trec_dl") or r["collection"] == "msmarco_dev"} == {"msmarco-passage"}
    manifest = write_manifest(config, tmp_path / "manifest")
    stages = {stage["name"]: stage for stage in manifest["stages"]}
    assert list(stages)[-2:] == ["fit-predictor", "report"]
    assert stages["encode-corpus"]["gpus"] == stages["fit-gpu"]["gpus"] == 1
    assert stages["fit-cpu"]["gpus"] == stages["index"]["gpus"] == 0
    manifest = write_manifest(config, tmp_path / "routing", "routing")
    assert next(stage for stage in manifest["stages"] if stage["name"] == "routing")["gpus"] == 2
    retrieval = write_manifest(config, tmp_path / "setting-a", "setting-a")
    assert retrieval["family"] == "primary" and retrieval["requested_family"] == "setting-a"
    attention = write_manifest(config, tmp_path / "setting-b", "setting-b")
    assert attention["family"] == "attention"


def test_shared_sae_dictionary_reuses_exact_weights(tmp_path):
    config = load_config("configs/fixture.yaml")
    config["output_root"] = str(tmp_path)
    original = next(b for b in config["bases"] if b["name"] == "sae")
    shared = {**copy.deepcopy(original), "name": "sae_shared", "dictionary_basis": "sae", "dictionary_granularity": "token", "secondary": True}
    config["bases"].append(shared)
    with StopFlag() as stop:
        pipeline.prepare(config, "train", stop)
        pipeline.encode_sample(config, stop)
        a = pipeline.fit_basis(config, "sae", "token", 0, stop)
        b = pipeline.fit_basis(config, "sae_shared", "pooled", 0, stop)
    with np.load(a / "weights.npz") as source, np.load(b / "weights.npz") as target:
        for key in source.files:
            np.testing.assert_array_equal(source[key], target[key])


def test_cached_evaluation_matches_uncached_and_resumes(tmp_path, monkeypatch):
    import pytest

    from src.artifacts import StopRequested
    config = load_config('configs/fixture.yaml')
    config['output_root'] = str(tmp_path / 'run')
    config['evaluation_batch_size'] = 2
    with StopFlag() as stop:
        for dataset in ('train', 'english'):
            pipeline.prepare(config, dataset, stop)
        pipeline.encode_sample(config, stop)
        pipeline.fit_basis(config, 'identity', 'pooled', 0, stop)
        pipeline.encode_corpus(config, 'english', 'identity', 'pooled', 0, stop)
        pipeline.build_index(config, 'english', 'identity', 'pooled', 0, 'small', stop)
        directory = pipeline.evaluate(config, 'english', 'identity', 'pooled', 0, 'small', stop)
        outcomes = np.load(directory / 'outcomes.npy').copy()
        batches = {str(p.relative_to(directory)): read_json(p) for p in directory.glob('[01][01][01]/*.json')}
        import shutil
        shutil.rmtree(directory)
        monkeypatch.setenv('CSX_POSTINGS_CACHE', str(tmp_path / 'cache'))
        class Interrupted:
            calls = 0
            def check(self):
                self.calls += 1
                if self.calls == 2 + len(read_json(pipeline.index_path(
                        config, 'english', 'identity', 'pooled', 0, 'small') / 'shards.json')):
                    raise StopRequested()
        with pytest.raises(StopRequested):
            pipeline.evaluate(config, 'english', 'identity', 'pooled', 0, 'small', Interrupted())
        saved = {p: p.read_bytes() for p in directory.glob('[01][01][01]/*.json')}
        assert saved
        pipeline.evaluate(config, 'english', 'identity', 'pooled', 0, 'small', stop)
        assert all(p.read_bytes() == data for p, data in saved.items())
        np.testing.assert_array_equal(np.load(directory / 'outcomes.npy'), outcomes)
        for name, records in batches.items():
            for expected, actual in zip(records, read_json(directory / name)):
                for field in ('query_id', 'ranking', 'metrics', 'posting_visits', 'query_nonzeros'):
                    assert expected[field] == actual[field]


def test_gpu_grouping_preserves_existing_cpu_checkpoints(tmp_path, monkeypatch):
    import functools

    import src.retrieval.gpu_scoring as gpu
    config = load_config('configs/fixture.yaml')
    config['output_root'] = str(tmp_path / 'run')
    config['evaluation_batch_size'] = 2
    with StopFlag() as stop:
        for dataset in ('train', 'english'):
            pipeline.prepare(config, dataset, stop)
        pipeline.encode_sample(config, stop)
        pipeline.fit_basis(config, 'identity', 'pooled', 0, stop)
        pipeline.encode_corpus(config, 'english', 'identity', 'pooled', 0, stop)
        pipeline.build_index(config, 'english', 'identity', 'pooled', 0, 'small', stop)
        directory = pipeline.evaluate(config, 'english', 'identity', 'pooled', 0, 'small', stop)
        outcomes = np.load(directory / 'outcomes.npy').copy()
        expected = {str(p.relative_to(directory)): read_json(p) for p in directory.glob('[01][01][01]/*.json')}
        retained = directory / '000/00000002.json'
        retained_bytes = retained.read_bytes()
        for p in directory.glob('[01][01][01]/*.json'):
            if p != retained:
                p.unlink()
        manifest = read_json(directory / 'manifest.json')
        manifest['status'] = 'interrupted'
        from src.artifacts import atomic_json
        atomic_json(directory / 'manifest.json', manifest)
        monkeypatch.setenv('CSX_EVALUATION_BACKEND', 'cuda')
        monkeypatch.setenv('CSX_GPU_QUERY_BATCH', '4')
        monkeypatch.setattr(gpu, 'BatchedScorer', functools.partial(gpu.BatchedScorer, device='cpu'))
        pipeline.evaluate(config, 'english', 'identity', 'pooled', 0, 'small', stop)
        assert retained.read_bytes() == retained_bytes
        np.testing.assert_array_equal(np.load(directory / 'outcomes.npy'), outcomes)
        for name, records in expected.items():
            for ref, actual in zip(records, read_json(directory / name)):
                for key in ('ranking', 'metrics', 'posting_visits', 'query_id'):
                    assert ref[key] == actual[key]
