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
