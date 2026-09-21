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

BM25S_FILES = ["data.csc.index.npy", "indices.csc.index.npy", "indptr.csc.index.npy",
               "vocab.index.json", "params.index.json", "nonoccurrence_array.index.npy"]


class _TokenCorpus:
    """Re-iterable token stream: BM25S makes two passes while building its index."""

    def __init__(self, path, stop):
        self.path, self.stop = path, stop

    def __iter__(self):
        for number, record in enumerate(jsonl(self.path)):
            if number % 4096 == 0:
                self.stop.check()
            yield lexical_tokens(Document(**record).content)


def _bm25_config(config):
    return {"implementation": "bm25s", "method": "lucene", "k1": 1.2, "b": 0.75,
            "backend": "numba" if not config.get("fixture", False) else "numpy",
            "csc_backend": "numpy", "mmap": not config.get("fixture", False),
            "threads": int(config.get("bm25", {}).get("threads", 0)),
            "candidate_buffer": int(config.get("bm25", {}).get("candidate_buffer", 4096)),
            **config.get("bm25", {})}


def _bm25_full_scores(retriever, tokens):
    """Exact BM25S scores used only when a candidate-boundary tie is unresolved."""
    scores = np.zeros(int(retriever.scores["num_docs"]), dtype=np.float32)
    for token in dict.fromkeys(tokens):
        feature = retriever.vocab_dict.get(token)
        if feature is None:
            continue
        start, end = retriever.scores["indptr"][feature:feature + 2]
        rows = retriever.scores["indices"][start:end]
        scores[rows] += retriever.scores["data"][start:end]
    return scores


def _bm25_rank(retriever, tokens, doc_ids, k, exclude_id, settings, candidates=None):
    # Collections that request this option define queries over their own document IDs.
    eligible = len(doc_ids) - int(exclude_id is not None)
    wanted = min(k, eligible)
    if wanted <= 0:
        return []
    candidate_k = min(len(doc_ids), wanted + settings["candidate_buffer"] + int(exclude_id is not None))
    if candidates is None:
        indices, scores = retriever.retrieve([list(dict.fromkeys(tokens))], k=candidate_k, sorted=True,
                                             show_progress=False, n_threads=settings["threads"])
        indices, scores = indices[0], scores[0]
    else:
        indices, scores = candidates
    indices, scores = np.asarray(indices), np.asarray(scores, dtype=np.float32)
    names = np.asarray(doc_ids, dtype=str)[indices]
    keep = names != exclude_id if exclude_id is not None else np.ones(len(names), dtype=bool)
    order = np.lexsort((names[keep], -scores[keep]))
    selected_indices, selected_scores = indices[keep][order], scores[keep][order]
    # BM25S may choose an arbitrary subset when a score tie crosses its requested k.
    # Resolve that uncommon case against its mmap'd CSC arrays.
    if candidate_k < len(doc_ids) and len(selected_scores) >= wanted and selected_scores[wanted - 1] == scores[-1]:
        full = _bm25_full_scores(retriever, tokens)
        selected_indices = stable_topk(full, doc_ids, wanted, exclude_id)
        selected_scores = full[selected_indices]
    else:
        selected_indices, selected_scores = selected_indices[:wanted], selected_scores[:wanted]
    return [(str(doc_ids[i]), float(score)) for i, score in zip(selected_indices, selected_scores)]


def _dense_config(config):
    defaults = {"backend": "cpu" if config.get("fixture", False) else "gpu",
                "query_batch_size": 256, "candidate_buffer": 128, "gpu_id": 0}
    return {**defaults, **config.get("dense_search", {})}


def _gpu_dense_ranking(index, values, queries, ids, exclude_ids, settings):
    """Exact top-k from a GPU shard, with CPU fallback only at a tied cutoff."""
    wanted = min(1000, len(ids))
    candidate_k = min(len(ids), wanted + settings["candidate_buffer"] + 1)
    distances, neighbors = index.search(np.ascontiguousarray(queries, dtype=np.float32), candidate_k)
    output = []
    for query, exclude_id, raw_scores, rows in zip(queries, exclude_ids, distances, neighbors):
        scores = raw_scores
        valid = rows >= 0
        rows, scores = rows[valid], scores[valid]
        names = np.asarray(ids, dtype=str)[rows]
        if exclude_id is not None:
            keep = names != exclude_id
            rows, scores, names = rows[keep], scores[keep], names[keep]
        order = np.lexsort((names, -scores))
        rows, scores = rows[order], scores[order]
        # GPU Faiss returns an arbitrary subset at a tied candidate boundary.
        # Full CPU scores preserve the project's documented deterministic doc-ID tie rule.
        if candidate_k < len(ids) and len(scores) >= wanted and scores[wanted - 1] == raw_scores[-1]:
            full = np.asarray(query @ values.T, dtype=np.float32)
            selected = stable_topk(full, ids, wanted, exclude_id)
            output.append([(str(ids[i]), float(full[i])) for i in selected])
        else:
            output.append([(str(ids[i]), float(score)) for i, score in zip(rows[:wanted], scores[:wanted])])
    return output


def baseline(config, dataset, method, stop):
    if method not in {"bm25", "dense", "splade"}:
        raise ValueError("Baseline must be bm25, dense, or splade")
    collection, corpus = prepared_paths(config, dataset)
    model_spec = config.get("external", {}).get("splade", {}) if method == "splade" else config["encoder"]
    directory = root(config) / "baselines" / method / corpus_key(spec_for(config, dataset))
    encoder, vocabulary = None, None
    bm25_settings = _bm25_config(config) if method == "bm25" else None
    dense_settings = _dense_config(config) if method == "dense" else None
    with Artifact(directory, {"method": method, "model": model_spec if method != "bm25" else None,
                              "bm25": bm25_settings,
                              "tokenizer": "unicode_NFC_casefold_LNM", "shard_size": config.get("shard_size", 4096)}, [corpus]) as work:
        if not work.reused:
            if method == "bm25":
                import bm25s

                ids = [record["doc_id"] for record in jsonl(corpus / "documents.jsonl")]
                if len(ids) != len(set(ids)):
                    raise ValueError("Document IDs must be unique")
                atomic_json(directory / "ids.json", ids)
                retriever = bm25s.BM25(k1=bm25_settings["k1"], b=bm25_settings["b"],
                                       method=bm25_settings["method"], backend=bm25_settings["backend"],
                                       csc_backend=bm25_settings["csc_backend"])
                retriever.index(_TokenCorpus(corpus / "documents.jsonl", stop), show_progress=True)
                retriever.save(directory, show_progress=False)
                work.complete(["ids.json"] + [name for name in BM25S_FILES if (directory / name).exists()],
                              documents=len(ids), nonzeros=len(retriever.scores["data"]))
                retriever = None
            else:
                encoder = SpladeEncoder(**model_spec) if method == "splade" else make_encoder(model_spec)
            if method == "bm25":
                pass
            else:
                shards, outputs, stats = [], [], None
            for number, records in (() if method == "bm25" else enumerate(chunks(jsonl(corpus / "documents.jsonl"), config.get("shard_size", 4096)))):
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
            if method != "bm25":
                if stats:
                    atomic_json(directory / "statistics.json", stats.to_dict())
                    outputs.append("statistics.json")
                atomic_json(directory / "shards.json", shards)
                work.complete(outputs + ["shards.json"])
    evaluation = root(config) / "baseline_evaluations" / dataset / method
    with Artifact(evaluation, {"dataset": dataset, "method": method, "dense_search": dense_settings}, [directory, collection]) as work:
        if work.reused:
            return evaluation
        queries = list(jsonl(collection / "queries.jsonl"))
        judgments = qrels_mapping(jsonl(collection / "qrels.jsonl"))
        queries = [q for q in queries if q["query_id"] in judgments]
        if method == "bm25":
            import bm25s

            doc_ids = read_json(directory / "ids.json")
            retriever = bm25s.BM25.load(directory, load_corpus=False, mmap=bm25_settings["mmap"],
                                        override_params={"backend": bm25_settings["backend"],
                                                         "csc_backend": bm25_settings["csc_backend"]},
                                        show_progress=False)
            query_tokens = [lexical_tokens(q["text"]) for q in queries]
        else:
            if encoder is None:
                encoder = SpladeEncoder(**model_spec) if method == "splade" else make_encoder(model_spec)
            blocks = [encoder.encode([q["text"] for q in batch], "query")
                      for batch in chunks(queries, config["encoder"].get("batch_size", 8))]
            matrix = sparse.vstack(blocks) if method == "splade" else np.concatenate([b.dense for b in blocks])
        gpu_resources = None
        if method == "dense" and dense_settings["backend"] == "gpu" and not config.get("fixture", False):
            import faiss

            if not hasattr(faiss, "StandardGpuResources"):
                raise RuntimeError("Dense GPU evaluation requires a GPU-enabled Faiss build; see environment/faiss-gpu.yaml")
            gpu_resources = faiss.StandardGpuResources()
        stats = CollectionStats.from_dict(read_json(directory / "statistics.json")) if method == "splade" else None
        records, outputs = [], []
        batch_size = dense_settings["query_batch_size"] if method == "dense" else config.get("evaluation_batch_size", 32)
        for start in range(0, len(queries), batch_size):
            stop.check()
            filename = f"batch-{start:08d}.json"
            if (evaluation / filename).exists():
                records.extend(read_json(evaluation / filename))
                outputs.append(filename)
                continue
            end = min(len(queries), start + config.get("evaluation_batch_size", 32))
            ranking = [[] for _ in range(end-start)]
            tick = time.perf_counter()
            if method == "bm25":
                token_batch = [list(dict.fromkeys(query_tokens[i])) for i in range(start, end)]
                candidate_k = min(len(doc_ids), 1001 + bm25_settings["candidate_buffer"])
                candidate_ids, candidate_scores = retriever.retrieve(
                    token_batch, k=candidate_k, sorted=True, show_progress=False,
                    n_threads=bm25_settings["threads"])
                ranking = [_bm25_rank(retriever, query_tokens[i], doc_ids, 1000,
                           queries[i]["query_id"] if spec_for(config, dataset).get("exclude_identical_ids") else None,
                           bm25_settings, (candidate_ids[i-start], candidate_scores[i-start]))
                           for i in range(start, end)]
            for path in ([] if method == "bm25" else read_json(directory / "shards.json")):
                stop.check()
                shard = Path(path)
                ids = read_json(shard / "ids.json")
                if method == "dense":
                    values = np.load(shard / "vectors.npy")
                    excluded = [queries[start+i]["query_id"] if spec_for(config, dataset).get("exclude_identical_ids") else None
                                for i in range(end-start)]
                    if config.get("fixture", False) or dense_settings["backend"] == "cpu":
                        scores = matrix[start:end] @ values.T
                        partial = [[(ids[j], float(s[j])) for j in stable_topk(s, ids, 1000, excluded[i])]
                                   for i, s in enumerate(scores)]
                    else:
                        import faiss

                        index = faiss.GpuIndexFlatIP(gpu_resources, values.shape[1])
                        index.add(values)
                        partial = _gpu_dense_ranking(index, values, matrix[start:end], ids, excluded, dense_settings)
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
            "implementation": bm25_settings["implementation"] if method == "bm25" else None,
            "dense_search": dense_settings if method == "dense" else None,
            "index_bytes": sum((directory / name).stat().st_size for name in BM25S_FILES
                               if method == "bm25" and (directory / name).exists()) if method == "bm25" else None,
            "fixture": config.get("fixture", False), "corpus_id": spec_for(config, dataset)["corpus_id"],
            "metrics": {k: float(np.mean([r["metrics"][k] for r in records])) for k in records[0]["metrics"]},
            "examples": len(records)})
        work.complete(outputs + ["result.json"])
    return evaluation
