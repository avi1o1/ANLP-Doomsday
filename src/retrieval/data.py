"""Streaming ir_datasets and local BEIR-format adapters with disk-backed validation."""

from __future__ import annotations

import hashlib
import heapq
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

from src.artifacts import Artifact, StopFlag, file_hash, jsonl, write_jsonl


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str

    @property
    def content(self):
        return "\n".join(part for part in (self.title.strip(), self.text.strip()) if part)


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


@dataclass(frozen=True)
class Qrel:
    query_id: str
    doc_id: str
    relevance: int


class IRDataset:
    def __init__(self, spec):
        import ir_datasets

        self.spec = spec
        self.dataset = ir_datasets.load(spec["id"])

    def documents(self):
        for d in self.dataset.docs_iter():
            yield Document(str(d.doc_id), getattr(d, "title", "") or "", d.text)

    def queries(self):
        for q in self.dataset.queries_iter():
            yield Query(str(q.query_id), q.text)

    def qrels(self):
        for r in self.dataset.qrels_iter():
            yield Qrel(str(r.query_id), str(r.doc_id), int(r.relevance))


class LocalDataset:
    """LIMIT and offline fixtures: corpus/queries JSONL, TREC or BEIR TSV qrels.

    Source checksums are part of artifact identity. No invented relevance labels.
    """

    def __init__(self, spec):
        self.spec = spec
        self.root = Path(spec["path"])

    def documents(self):
        for d in jsonl(self.root / "corpus.jsonl"):
            yield Document(str(d.get("doc_id", d.get("_id"))), d.get("title", ""), d["text"])

    def queries(self):
        for q in jsonl(self.root / "queries.jsonl"):
            yield Query(str(q.get("query_id", q.get("_id"))), q["text"])

    def qrels(self):
        path = self.root / self.spec.get("qrels", "qrels.tsv")
        with path.open() as handle:
            for line in handle:
                fields = line.split()
                if not fields or fields[0] in {"query-id", "query_id"}:
                    continue
                if len(fields) == 4:
                    qid, _, did, grade = fields
                elif len(fields) == 3:
                    qid, did, grade = fields
                else:
                    raise ValueError(f"Invalid qrels line: {line!r}")
                yield Qrel(qid, did, int(grade))


class HFJSONDataset(LocalDataset):
    """Released BEIR/MTEB JSONL files, used for LIMIT when absent from ir_datasets."""

    def __init__(self, spec):
        from huggingface_hub import hf_hub_download

        self.spec = spec
        self.files = {name: Path(hf_hub_download(spec["id"], name, repo_type="dataset",
                     revision=spec.get("revision", "main")))
                      for name in ("corpus.jsonl", "queries.jsonl", "qrels.jsonl")}
        self.root = self.files["corpus.jsonl"].parent
        self.identity = {"resolved_revision": self.root.name,
                         "source_files": {name: file_hash(path) for name, path in self.files.items()}}

    def qrels(self):
        for record in jsonl(self.files["qrels.jsonl"]):
            yield Qrel(str(record["query-id"]), str(record["corpus-id"]), int(record["score"]))


def adapter(spec):
    if spec["source"] == "ir_datasets":
        return IRDataset(spec)
    if spec["source"] == "local":
        return LocalDataset(spec)
    if spec["source"] == "hf_json":
        return HFJSONDataset(spec)
    raise ValueError(f"Unknown dataset source: {spec['source']}")


def corpus_key(spec):
    return hashlib.sha256(spec["corpus_id"].encode()).hexdigest()[:16]


def prepare_dataset(spec, output_root, stop: StopFlag):
    root = Path(output_root) / "datasets"
    source = adapter(spec)
    source_identity = dict(spec)
    corpus_identity = {"source": spec["source"], "corpus_id": spec["corpus_id"],
                       "text_policy": "nonempty_title_newline_text_v1"}
    if spec["source"] == "ir_datasets":
        corpus_identity.update(docs_namespace=source.dataset.docs_namespace(),
                               declared_document_count=source.dataset.docs_count())
    if spec["source"] == "hf_json":
        source_identity.update(source.identity)
        corpus_identity["corpus_sha256"] = source.identity["source_files"]["corpus.jsonl"]
    if spec["source"] == "local":
        for key in ("corpus.jsonl", "queries.jsonl", spec.get("qrels", "qrels.tsv")):
            source_identity[key] = file_hash(Path(spec["path"]) / key)
        corpus_identity["corpus_sha256"] = source_identity["corpus.jsonl"]
    corpus = root / "corpora" / corpus_key(spec)
    with Artifact(corpus, corpus_identity) as work:
        if not work.reused:
            # Rebuild an interrupted preparation transaction. GPU stages resume at shard boundaries.
            db_path = corpus / "ids.sqlite"
            if db_path.exists():
                db_path.unlink()
            with sqlite3.connect(db_path) as db, (corpus / "documents.jsonl.tmp").open("w") as out:
                db.execute("CREATE TABLE documents (id TEXT PRIMARY KEY)")
                count = 0
                for d in source.documents():
                    if not d.doc_id or d.doc_id == "None":
                        raise ValueError("Missing document ID")
                    try:
                        db.execute("INSERT INTO documents VALUES (?)", (d.doc_id,))
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(f"Duplicate document ID: {d.doc_id}") from exc
                    out.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
                    count += 1
                    if count % 10000 == 0:
                        stop.check()
                        db.commit()
                db.commit()
            if not count:
                raise ValueError("Document corpus is empty")
            (corpus / "documents.jsonl.tmp").replace(corpus / "documents.jsonl")
            expected = spec.get("expected_documents")
            if expected is not None and count != expected:
                raise ValueError(f"Expected {expected} documents, found {count}")
            work.complete(["documents.jsonl", "ids.sqlite"], document_count=count)
    dataset_key = spec.get("name", spec.get("id", Path(spec.get("path", "local")).name)).replace("/", "__")
    collection = root / "collections" / dataset_key
    with Artifact(collection, source_identity, parents=[corpus]) as work:
        if not work.reused:
            query_ids = set()
            queries = []
            for q in source.queries():
                if not q.query_id or q.query_id == "None" or q.query_id in query_ids:
                    raise ValueError(f"Missing or duplicate query ID: {q.query_id}")
                query_ids.add(q.query_id)
                queries.append(asdict(q))
            expected = spec.get("expected_queries")
            if expected is not None and len(queries) != expected:
                raise ValueError(f"Expected {expected} queries, found {len(queries)}")
            count = 0
            with sqlite3.connect(f"file:{corpus / 'ids.sqlite'}?mode=ro", uri=True) as db:
                def checked_qrels():
                    nonlocal count
                    for r in source.qrels():
                        if r.query_id not in query_ids:
                            raise ValueError(f"Qrel references unknown query {r.query_id}")
                        if db.execute("SELECT 1 FROM documents WHERE id=?", (r.doc_id,)).fetchone() is None:
                            raise ValueError(f"Qrel references unknown document {r.doc_id}")
                        count += 1
                        if count % 10000 == 0:
                            stop.check()
                        yield asdict(r)
                write_jsonl(collection / "qrels.jsonl", checked_qrels())
            if not count or not queries:
                raise ValueError("Evaluation/development collection requires queries and qrels")
            write_jsonl(collection / "queries.jsonl", queries)
            work.complete(["queries.jsonl", "qrels.jsonl"], query_count=len(queries),
                          qrel_count=count, corpus_path=str(corpus.resolve()),
                          corpus_id=spec["corpus_id"], language=spec["language"])
    return collection, corpus


def deterministic_sample(documents, count, seed):
    """Bottom-hash sample: independent of stream order, bounded memory."""
    if count <= 0:
        raise ValueError("Sample size must be positive")
    heap = []
    for d in documents:
        key = int.from_bytes(hashlib.sha256(f"{seed}:{d.doc_id}".encode()).digest(), "big")
        entry = (-key, d.doc_id, d)
        if len(heap) < count:
            heapq.heappush(heap, entry)
        elif key < -heap[0][0]:
            heapq.heapreplace(heap, entry)
    return [entry[2] for entry in sorted(heap, key=lambda e: (-e[0], e[1]))]


def split_documents(documents, validation_fraction, seed):
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    ordered = sorted(documents, key=lambda d: hashlib.sha256(f"split:{seed}:{d.doc_id}".encode()).digest())
    n = max(1, int(len(ordered) * validation_fraction))
    if n >= len(ordered):
        raise ValueError("Need at least two fitting documents")
    return ordered[n:], ordered[:n]
