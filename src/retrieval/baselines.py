"""Full-corpus lexical BM25, official E5 dense, and external SPLADE references."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy import sparse

from src.artifacts import Artifact, atomic_json, jsonl, read_json
from src.config import root
from src.retrieval.data import Document, corpus_key
from src.retrieval.encoding import SpladeEncoder, make_encoder
from src.retrieval.index import (
    InvertedIndex,
    counts_matrix,
    lexical_tokens,
    merge_rankings,
    qrels_mapping,
    reference_metrics,
    standard_metrics,
)
from src.retrieval.pipeline import chunks, prepared_paths, spec_for
from src.scoring import CollectionStats, Switches, stable_topk


def baseline(config, dataset, method, stop):
    if method not in {"bm25", "dense", "splade"}:
        raise ValueError("Baseline must be bm25, dense, or splade")
    collection, corpus = prepared_paths(config, dataset)
    model_spec = config.get("external", {}).get("splade", {}) if method == "splade" else config["encoder"]
    directory = root(config) / "baselines" / method / corpus_key(spec_for(config, dataset))
    encoder, vocabulary = None, None
    with Artifact(directory, {"method": method, "model": model_spec if method != "bm25" else None,
                              "tokenizer": "unicode_NFC_casefold_LNM", "shard_size": config.get("shard_size", 4096)}, [corpus]) as work:
        if not work.reused:
            if method == "bm25":
                terms = set()
                for batch in chunks(jsonl(corpus / "documents.jsonl"), 4096):
                    stop.check()
                    for record in batch:
                        terms.update(lexical_tokens(Document(**record).content))
                vocabulary = {term: i for i, term in enumerate(sorted(terms))}
                atomic_json(directory / "vocabulary.json", vocabulary)
            else:
                encoder = SpladeEncoder(**model_spec) if method == "splade" else make_encoder(model_spec)
            shards, outputs, stats = [], [], None
            for number, records in enumerate(chunks(jsonl(corpus / "documents.jsonl"), config.get("shard_size", 4096))):
                stop.check()
                shard = directory / f"{number:06d}"
                with Artifact(shard, {"number": number, "method": method, "model": model_spec}, [corpus]) as part:
                    if not part.reused:
                        texts = [Document(**r).content for r in records]
                        if method == "bm25":
                            values = counts_matrix(texts, vocabulary)
                        else:
                            blocks = []
                            for batch in chunks(texts, config["encoder"].get("batch_size", 8)):
                                stop.check()
                                encoded = encoder.encode(batch)
                                blocks.append(encoded if method == "splade" else encoded.dense)
                            values = sparse.vstack(blocks) if method == "splade" else np.concatenate(blocks)
                        if method == "dense":
                            np.save(shard / "vectors.npy", values)
                        else:
                            sparse.save_npz(shard / "vectors.npz", values)
                        atomic_json(shard / "ids.json", [r["doc_id"] for r in records])
                        part.complete(["ids.json", "vectors.npy" if method == "dense" else "vectors.npz"],
                                      model_revision=encoder.revision if encoder else None)
                shards.append(str(shard))
                if method != "dense":
                    values = sparse.load_npz(shard / "vectors.npz")
                    if stats is None:
                        stats = CollectionStats.empty(values.shape[1])
                    stats.update(values)
            if stats:
                atomic_json(directory / "statistics.json", stats.to_dict())
                outputs.append("statistics.json")
            if method == "bm25":
                outputs.append("vocabulary.json")
            atomic_json(directory / "shards.json", shards)
            work.complete(outputs + ["shards.json"])
    evaluation = root(config) / "baseline_evaluations" / dataset / method
    with Artifact(evaluation, {"dataset": dataset, "method": method}, [directory, collection]) as work:
        if work.reused:
            return evaluation
        queries = list(jsonl(collection / "queries.jsonl"))
        judgments = qrels_mapping(jsonl(collection / "qrels.jsonl"))
        queries = [q for q in queries if q["query_id"] in judgments]
        if method == "bm25":
            vocabulary = read_json(directory / "vocabulary.json")
            matrix = counts_matrix([q["text"] for q in queries], vocabulary, binary=True)
        else:
            if encoder is None:
                encoder = SpladeEncoder(**model_spec) if method == "splade" else make_encoder(model_spec)
            blocks = [encoder.encode([q["text"] for q in batch], "query")
                      for batch in chunks(queries, config["encoder"].get("batch_size", 8))]
            matrix = sparse.vstack(blocks) if method == "splade" else np.concatenate([b.dense for b in blocks])
        stats = CollectionStats.from_dict(read_json(directory / "statistics.json")) if method != "dense" else None
        records, outputs = [], []
        for start in range(0, len(queries), config.get("evaluation_batch_size", 32)):
            stop.check()
            filename = f"batch-{start:08d}.json"
            if (evaluation / filename).exists():
                records.extend(read_json(evaluation / filename))
                outputs.append(filename)
                continue
            end = min(len(queries), start + config.get("evaluation_batch_size", 32))
            ranking = [[] for _ in range(end-start)]
            tick = time.perf_counter()
            for path in read_json(directory / "shards.json"):
                stop.check()
                shard = Path(path)
                ids = read_json(shard / "ids.json")
                if method == "dense":
                    values = np.load(shard / "vectors.npy")
                    if config.get("fixture", False):
                        scores = matrix[start:end] @ values.T
                    else:
                        import faiss

                        index = faiss.IndexFlatIP(values.shape[1])
                        index.add(values)
                        # Full shard search retains exact deterministic ties even at the top-k cutoff.
                        distances, neighbors = index.search(np.ascontiguousarray(matrix[start:end]), len(ids))
                        scores = np.empty_like(distances)
                        np.put_along_axis(scores, neighbors, distances, axis=1)
                    partial = [[(ids[j], float(s[j])) for j in stable_topk(s, ids, 1000,
                        queries[start+i]["query_id"] if spec_for(config, dataset).get("exclude_identical_ids") else None)]
                               for i, s in enumerate(scores)]
                else:
                    index = InvertedIndex(sparse.load_npz(shard / "vectors.npz"), ids, stats,
                                          Switches(True, True, True) if method == "bm25" else Switches())
                    partial = [index.search(matrix[i], 1000,
                        queries[i]["query_id"] if spec_for(config, dataset).get("exclude_identical_ids") else None)[0]
                               for i in range(start, end)]
                ranking = [merge_rankings([a, b], 1000) for a, b in zip(ranking, partial)]
            elapsed = time.perf_counter() - tick
            batch_records = []
            for i, query in enumerate(queries[start:end]):
                qid = query["query_id"]
                values = (reference_metrics(ranking[i], judgments[qid]) if config.get("fixture", False)
                          else standard_metrics(qid, ranking[i], judgments[qid]))
                batch_records.append({"query_id": qid, "ranking": ranking[i], "metrics": values,
                                      "amortized_search_seconds": elapsed / (end-start)})
            atomic_json(evaluation / filename, batch_records)
            records.extend(batch_records)
            outputs.append(filename)
        atomic_json(evaluation / "result.json", {
            "collection": dataset, "method": method, "external": method == "splade",
            "fixture": config.get("fixture", False), "corpus_id": spec_for(config, dataset)["corpus_id"],
            "metrics": {k: float(np.mean([r["metrics"][k] for r in records])) for k in records[0]["metrics"]},
            "examples": len(records)})
        work.complete(outputs + ["result.json"])
    return evaluation
