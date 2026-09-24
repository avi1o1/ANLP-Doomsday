import json

import numpy as np
import pytest
import torch
from scipy import sparse

from src.artifacts import StopFlag, read_json
from src.config import load_config
from src.experiments import build_tasks, experiment_rows
from src.retrieval import pipeline
from src.retrieval.bases import Basis
from src.retrieval.gpu_bases import TensorBasis, deterministic_topk
from src.retrieval.shared_encoding import encode_shared, join_shared


@pytest.mark.parametrize("kind", ["identity", "random", "pca", "kmeans", "pq", "rq", "sae"])
def test_tensor_basis_matches_reference(kind):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(40, 8)).astype("float32")
    specs = {"identity": {}, "random": {"components": 8}, "pca": {"components": 4},
             "kmeans": {"clusters": 4, "assignments": 2, "backend": "sklearn"},
             "pq": {"subspaces": 2, "codes": 4, "backend": "sklearn"},
             "rq": {"stages": 2, "codes": 4, "backend": "sklearn"},
             "sae": {"features": 16, "active": 4}}
    b = Basis({"kind": kind, **specs[kind]}, 8)
    if kind == "sae":
        b.arrays = {"encoder": rng.normal(size=(16, 8)).astype("float32"),
                    "center": x.mean(0), "bias": np.zeros(16, "float32")}
    else:
        b.fit(x)
    b.scale = 2.3
    adapter = TensorBasis(b, device="cpu", chunk_size=3)
    tokens = [x[:0], x[:7], x[7:9]]
    pooled = np.stack([np.zeros(8), x[:7].mean(0), x[7:9].mean(0)]).astype("float32")
    for granularity in ["pooled", "token"]:
        reference = b.aggregate(pooled, tokens, granularity).toarray()
        actual = adapter.aggregate(pooled, tokens, granularity).toarray()
        np.testing.assert_allclose(actual, reference, atol=2e-5, rtol=2e-5)


def test_topk_boundary_ties_and_zero_rows():
    x = torch.tensor([[3., 2., 2., 2., 1.], [0., 0., 0., 0., 0.]])
    ids, _ = deterministic_topk(x, 3)
    assert ids.tolist() == [[0, 1, 2], [0, 1, 2]]


def test_shared_encoder_sharding_resume_and_legacy_parity(tmp_path, monkeypatch):
    from src.retrieval import shared_encoding
    config = load_config("configs/fixture.yaml")
    config["output_root"] = str(tmp_path)
    config["shared_encoding"] = {"enabled": True, "backend": "cpu", "batch_size": 3}
    representations = [["identity", "pooled", 0], ["identity", "token", 0]]
    calls = []
    original = shared_encoding.make_encoder

    def make(spec):
        encoder = original(spec)
        encode = encoder.encode
        def wrapped(texts, *args, **kwargs):
            calls.extend(texts)
            return encode(texts, *args, **kwargs)
        encoder.encode = wrapped
        return encoder

    monkeypatch.setattr(shared_encoding, "make_encoder", make)
    with StopFlag() as stop:
        pipeline.prepare(config, "train", stop)
        pipeline.prepare(config, "english", stop)
        pipeline.encode_sample(config, stop)
        for _, granularity, _ in representations:
            pipeline.fit_basis(config, "identity", granularity, 0, stop)
        pipeline.encode_corpus(config, "english", "identity", "pooled", 0, stop, 0, 5)
        existing = pipeline.encoded_path(config, "english", "identity", "pooled", 0, "documents") / "shards/000000/vectors.npz"
        saved_time = existing.stat().st_mtime_ns
        for shard_id in [1, 0]:
            encode_shared(config, "english", "documents", representations, stop, shard_id, 2)
        assert len(calls) == 30  # One pass, not 30 times two representations.
        encode_shared(config, "english", "documents", representations, stop)
        assert len(calls) == 30
        assert existing.stat().st_mtime_ns == saved_time
        join_shared(config, "english", "documents", representations, stop)
        for name, granularity, seed in representations:
            directory = pipeline.encoded_path(config, "english", name, granularity, seed, "documents")
            paths = read_json(directory / "shards.json")
            actual = sparse.vstack([sparse.load_npz(p["path"] + "/vectors.npz") for p in paths])
            records = list(shared_encoding.jsonl(pipeline.prepared_paths(config, "english")[1] / "documents.jsonl"))
            enc = original(config["encoder"]).encode([shared_encoding.Document(**r).content for r in records])
            basis = Basis.load(pipeline.basis_path(config, name, granularity, seed))
            expected = basis.aggregate(enc.pooled, enc.tokens, granularity)
            expected = shared_encoding.sparsify(expected, pipeline.storage_cap(config, "english", "documents"))
            np.testing.assert_allclose(actual.toarray(), expected.toarray(), atol=1e-5)


def test_shared_interruption_retries_only_incomplete_shard(tmp_path, monkeypatch):
    from src.artifacts import StopRequested
    from src.retrieval import shared_encoding
    config = load_config("configs/fixture.yaml")
    config["output_root"] = str(tmp_path)
    config["shared_encoding"] = {"backend": "cpu", "batch_size": 7}
    reps = [["identity", "pooled", 0]]
    original = shared_encoding.make_encoder
    calls = []
    interrupt = [True]
    def make(spec):
        encoder = original(spec)
        encode = encoder.encode
        def wrapped(texts, *args, **kwargs):
            if interrupt[0] and calls:
                raise StopRequested()
            calls.extend(texts)
            return encode(texts, *args, **kwargs)
        encoder.encode = wrapped
        return encoder
    monkeypatch.setattr(shared_encoding, "make_encoder", make)
    with StopFlag() as stop:
        pipeline.prepare(config, "train", stop)
        pipeline.prepare(config, "english", stop)
        pipeline.encode_sample(config, stop)
        pipeline.fit_basis(config, "identity", "pooled", 0, stop)
        with pytest.raises(StopRequested):
            encode_shared(config, "english", "documents", reps, stop)
        interrupt[0] = False
        encode_shared(config, "english", "documents", reps, stop)
        assert len(calls) == 30
        join_shared(config, "english", "documents", reps, stop)


def test_shared_corpus_has_one_document_task_group():
    config = load_config("configs/research.yaml")
    tasks = build_tasks(config, experiment_rows(config))["encode-corpus"]
    shared = [t for t in tasks if t[t.index("--dataset")+1] in {"msmarco_dev", "trec_dl_2019", "trec_dl_2020"}]
    documents = [t for t in shared if t[t.index("--side")+1] == "documents"]
    assert len(documents) == 4
    assert len(json.loads(documents[0][documents[0].index("--representations")+1])) == 14
