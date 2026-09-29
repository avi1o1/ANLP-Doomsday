import json

import pytest

from src.artifacts import Artifact, StopFlag, StopRequested, atomic_json, jsonl, read_json
from src.config import digest
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


def test_explicit_completed_reuse_keeps_original_provenance(tmp_path, monkeypatch):
    from src import artifacts

    path = tmp_path / "artifact"
    with Artifact(path, {"seed": 0}) as work:
        atomic_json(path / "value.json", {"value": 2})
        work.complete(["value.json"])
    original = read_json(path / "manifest.json")
    monkeypatch.setattr(artifacts, "provenance", lambda: {"source_hash": "new", "versions": {}})
    with pytest.raises(ValueError, match="identity changed"), Artifact(path, {"seed": 0}):
        pass
    audit = tmp_path / "audit.json"
    atomic_json(audit, {"manifests": {str(path): digest(original)}})
    monkeypatch.setenv("CSX_REUSE_COMPLETED", str(audit))
    with Artifact(path, {"seed": 0}) as work:
        assert work.reused
    assert read_json(path / "manifest.json") == original
    with pytest.raises(ValueError, match="identity changed"), Artifact(path, {"seed": 1}):
        pass
    atomic_json(path / "value.json", {"value": 99})
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


def test_documented_missing_judgments_are_preserved_and_audited(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "corpus.jsonl").write_text('{"_id":"d", "title":"", "text":"document"}\n')
    (raw / "queries.jsonl").write_text('{"_id":"q", "text":"question"}\n')
    (raw / "qrels.tsv").write_text("q missing 2\nq d 1\n")
    spec = dict(source="local", path=str(raw), corpus_id="shared", name="known", language="en",
                role="evaluate", known_missing_qrel_documents={"missing": "Published source-data error"})
    with StopFlag() as stop:
        collection, corpus = prepare_dataset(spec, tmp_path / "out", stop)
        assert list(jsonl(collection / "qrels.jsonl")) == [
            {"query_id": "q", "doc_id": "missing", "relevance": 2},
            {"query_id": "q", "doc_id": "d", "relevance": 1},
        ]
        assert len(list(jsonl(corpus / "documents.jsonl"))) == 1
        issues = read_json(collection / "data_issues.json")["missing_document_judgments"]
        assert len(issues) == 1 and issues[0]["doc_id"] == "missing"
        assert read_json(collection / "manifest.json")["metadata"]["known_missing_document_judgments"] == 1
        (raw / "qrels.tsv").write_text("q unexpected 1\n")
        with pytest.raises(ValueError, match="unknown document unexpected"):
            prepare_dataset({**spec, "name": "unexpected"}, tmp_path / "out", stop)


def test_compact_json_matches_indented_content(tmp_path):
    from src.artifacts import atomic_json, read_json

    value = {"b": [1, 2.5, "x"], "a": {"n": None}}
    atomic_json(tmp_path / "wide.json", value)
    atomic_json(tmp_path / "tight.json", value, compact=True)
    assert read_json(tmp_path / "wide.json") == read_json(tmp_path / "tight.json") == value
    assert "\n  " in (tmp_path / "wide.json").read_text()
    assert (tmp_path / "tight.json").read_text() == '{"a":{"n":null},"b":[1,2.5,"x"]}\n'
