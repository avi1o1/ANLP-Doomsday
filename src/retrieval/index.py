"""Exact postings retrieval and standard per-query IR metrics."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy import sparse

from src.scoring import CollectionStats, nonnegative_csr, stable_topk, transform_documents


class InvertedIndex:
    def __init__(self, documents, doc_ids, stats: CollectionStats, switches):
        if len(doc_ids) != documents.shape[0] or len(set(doc_ids)) != len(doc_ids):
            raise ValueError("Document IDs must be unique and match matrix rows")
        self.doc_ids = np.asarray(doc_ids, dtype=str)
        self.postings = transform_documents(documents, stats, switches).tocsc()

    def search(self, query, k=1000, exclude_id=None):
        query = nonnegative_csr(query)
        if query.shape != (1, self.postings.shape[1]):
            raise ValueError("Expected one query with the index vocabulary")
        scores = np.zeros(len(self.doc_ids), dtype=np.float32)
        visits = 0
        for feature, weight in zip(query.indices, query.data):
            a, b = self.postings.indptr[feature:feature + 2]
            ids = self.postings.indices[a:b]
            scores[ids] += weight * self.postings.data[a:b]
            visits += b - a
        order = stable_topk(scores, self.doc_ids, k, exclude_id)
        return [(str(self.doc_ids[i]), float(scores[i])) for i in order], visits


def merge_rankings(rankings, k=1000):
    combined = {}
    for ranking in rankings:
        for doc_id, score in ranking:
            if doc_id in combined:
                raise ValueError(f"Overlapping document shards: {doc_id}")
            combined[doc_id] = score
    ids = list(combined)
    scores = list(combined.values())
    order = stable_topk(scores, ids, k)
    return [(ids[i], scores[i]) for i in order]


def qrels_mapping(records):
    result = defaultdict(dict)
    for r in records:
        old = result[r["query_id"]].get(r["doc_id"])
        if old is not None and old != r["relevance"]:
            raise ValueError("Conflicting relevance judgments")
        result[r["query_id"]][r["doc_id"]] = r["relevance"]
    return dict(result)


def reference_metrics(ranking, judgments):
    """Small independent reference; primary evaluation uses ir_measures.

    Graded nDCG uses linear gains, matching trec_eval / ir_measures defaults.
    Zero-positive judged queries receive zero. Unjudged queries are not evaluated.
    """
    relevant = {d for d, grade in judgments.items() if grade > 0}
    grades = [max(0, judgments.get(doc, 0)) for doc, _ in ranking]
    ideal = sorted((max(0, x) for x in judgments.values()), reverse=True)[:10]
    dcg = sum(grade / math.log2(i + 2) for i, grade in enumerate(grades[:10]))
    idcg = sum(grade / math.log2(i + 2) for i, grade in enumerate(ideal))
    reciprocal = next((1 / (i + 1) for i, g in enumerate(grades[:10]) if g > 0), 0.0)
    return {"ndcg@10": dcg / idcg if idcg else 0.0, "mrr@10": reciprocal,
            "recall@100": sum(g > 0 for g in grades[:100]) / len(relevant) if relevant else 0.0,
            "recall@1000": sum(g > 0 for g in grades[:1000]) / len(relevant) if relevant else 0.0}


def standard_metrics(query_id, ranking, judgments):
    import ir_measures
    from ir_measures import RR, R, nDCG

    qrels = [ir_measures.Qrel(query_id, d, int(g)) for d, g in judgments.items()]
    # Assign strictly ordered synthetic scores to preserve the pipeline's deterministic
    # doc-ID tie rule across trec_eval versions. Original scores remain in run files.
    run = [ir_measures.ScoredDoc(query_id, d, float(len(ranking) - i))
           for i, (d, _) in enumerate(ranking)]
    measures = {nDCG @ 10: "ndcg@10", RR @ 10: "mrr@10", R @ 100: "recall@100", R @ 1000: "recall@1000"}
    values = ir_measures.calc_aggregate(list(measures), qrels, run)
    return {name: float(values[measure]) for measure, name in measures.items()}


def lexical_tokens(text):
    """Unicode word tokenizer; retain combining marks for Indic scripts."""
    import unicodedata

    words, current = [], []
    for char in unicodedata.normalize("NFC", text).casefold():
        if unicodedata.category(char)[0] in {"L", "N", "M"}:
            current.append(char)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return words


def counts_matrix(texts, vocabulary, binary=False):
    from collections import Counter

    rows, cols, values = [], [], []
    for row, text in enumerate(texts):
        counts = Counter(lexical_tokens(text))
        for term, count in counts.items():
            if term in vocabulary:
                rows.append(row)
                cols.append(vocabulary[term])
                values.append(1 if binary else count)
    return sparse.csr_matrix((values, (rows, cols)), shape=(len(texts), len(vocabulary)), dtype=np.float32)
