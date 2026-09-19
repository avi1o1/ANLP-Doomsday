import json

import pytest

from src.artifacts import Artifact, StopFlag, StopRequested, atomic_json, read_json
from src.retrieval.data import Document, deterministic_sample, prepare_dataset, split_documents


def test_interrupted_artifact_and_corruption(tmp_path):
    path = tmp_path / "artifact"
    with pytest.raises(StopRequested), Artifact(path, {"seed": 0}):
        atomic_json(path / "checkpoint.json", {"cursor": 2})
        raise StopRequested()
    assert read_json(path / "manifest.json")["status"] == "interrupted"
    with Artifact(path, {"seed": 0}) as work:
        assert read_json(path / "checkpoint.json")["cursor"] == 2
        work.complete(["checkpoint.json"])
    with Artifact(path, {"seed": 0}) as work:
        assert work.reused
    with pytest.raises(ValueError), Artifact(path, {"seed": 1}):
        pass
    atomic_json(path / "checkpoint.json", {"cursor": 9})
    with pytest.raises(ValueError, match="corrupted"), Artifact(path, {"seed": 0}):
        pass


def test_compressed_statistics_are_atomic_and_deterministic(tmp_path):
    path = tmp_path / "reference.json.gz"
    value = {"counts": [0, 1, 2], "language": "తెలుగు"}
    atomic_json(path, value)
    first = path.read_bytes()
    assert read_json(path) == value
    atomic_json(path, value)
    assert path.read_bytes() == first


def test_sample_document_disjoint_and_order_invariant():
    documents = [Document(str(i), "", str(i)) for i in range(100)]
    a = deterministic_sample(documents, 30, 4)
    b = deterministic_sample(reversed(documents), 30, 4)
    assert a == b
    train, val = split_documents(a, 0.1, 4)
    assert not {d.doc_id for d in train} & {d.doc_id for d in val}
    assert len(val) == 3


def test_dataset_joins_shared_corpus(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "corpus.jsonl").write_text(json.dumps({"_id": "d", "title": "शीर्षक", "text": "text"}) + "\n")
    (raw / "queries.jsonl").write_text('{"_id":"q", "text":"question"}\n')
    (raw / "qrels.tsv").write_text("query-id\tcorpus-id\tscore\nq\td\t2\n")
    spec = dict(source="local", path=str(raw), corpus_id="shared", name="a", language="hi", role="evaluate")
    with StopFlag() as stop:
        a, corpus_a = prepare_dataset(spec, tmp_path / "out", stop)
        b, corpus_b = prepare_dataset({**spec, "name": "b"}, tmp_path / "out", stop)
        assert corpus_a == corpus_b and a != b
        assert read_json(a / "manifest.json")["metadata"]["language"] == "hi"
        (raw / "qrels.tsv").write_text("q missing 1\n")
        with pytest.raises(ValueError, match="unknown document"):
            prepare_dataset({**spec, "name": "bad"}, tmp_path / "out", stop)
