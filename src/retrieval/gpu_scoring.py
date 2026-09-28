"""Bounded batched GPU retrieval preserving the CPU's FP32 addition order.

This deliberately avoids GEMM, TF32 and fused multiply-add: each sorted query
feature contributes one separately rounded multiply and add, as in postings search.
"""
from __future__ import annotations

import numpy as np

from src.scoring import nonnegative_csr


class BatchedScorer:
    def __init__(self, query_matrix, document_ids, k=1000, excluded=None, device='cuda', block=4096):
        import torch
        self.torch, self.device = torch, device
        if block < 1:
            raise ValueError('Document block must be positive')
        # The scoring buffer is vocabulary by documents, so this bounds device memory
        # independently of how many documents an index shard holds.
        self.block = block
        self.queries = nonnegative_csr(query_matrix)
        self.names = np.asarray(document_ids, dtype=str)
        if len(np.unique(self.names)) != len(self.names):
            raise ValueError('Document shards must have disjoint unique IDs')
        lexical_order = np.argsort(self.names, kind='stable')
        self.lexical_names = self.names[lexical_order]
        self.lexical_ids = np.empty(len(self.names), dtype=np.int64)
        self.lexical_ids[lexical_order] = np.arange(len(self.names))
        self.k = min(k, len(self.names))
        rows, width = self.queries.shape[0], max(np.diff(self.queries.indptr), default=0)
        features = np.zeros((rows, width), dtype=np.int64)
        weights = np.zeros((rows, width), dtype=np.float32)
        for row in range(rows):
            a, b = self.queries.indptr[row:row + 2]
            features[row, :b-a], weights[row, :b-a] = self.queries.indices[a:b], self.queries.data[a:b]
        self.features = torch.as_tensor(features, device=device)
        self.weights = torch.as_tensor(weights, device=device)
        self.scores = torch.empty((rows, 0), dtype=torch.float32, device=device)
        self.ids = torch.empty((rows, 0), dtype=torch.int64, device=device)
        self.visits = np.zeros(rows, dtype=np.int64)
        self.excluded = np.full(rows, -1, dtype=np.int64)
        for row, name in enumerate(excluded or [None] * rows):
            if name is not None:
                pos = np.searchsorted(self.lexical_names, name)
                if pos < len(self.names) and self.lexical_names[pos] == name:
                    self.excluded[row] = pos
        self.excluded_gpu = torch.as_tensor(self.excluded, device=device)

    def add(self, postings, offset):
        matrix = postings.postings
        size = matrix.shape[0]
        if not np.array_equal(postings.doc_ids, self.names[offset:offset + size]):
            raise ValueError('Shard document order changed')
        if matrix.shape[1] != self.queries.shape[1]:
            raise ValueError('Query and document vocabulary mismatch')
        # Folding a block into the running top-k uses the same total order as folding
        # a whole shard, so the block size changes memory and never a ranking.
        rows = matrix.tocsr()
        for start in range(0, size, self.block):
            self._merge(rows[start:min(size, start + self.block)].tocsc(), offset + start)
        counts = np.diff(matrix.indptr)
        owners = np.repeat(np.arange(self.queries.shape[0]), np.diff(self.queries.indptr))
        self.visits += np.bincount(owners, weights=counts[self.queries.indices],
                                   minlength=self.queries.shape[0]).astype(np.int64)

    def _merge(self, x, offset):
        torch = self.torch
        size = x.shape[0]
        with torch.inference_mode():
            dense = torch.zeros((x.shape[1], size), dtype=torch.float32, device=self.device)
            feature_ids = np.repeat(np.arange(x.shape[1], dtype=np.int64), np.diff(x.indptr))
            dense[torch.as_tensor(feature_ids, device=self.device),
                  torch.as_tensor(x.indices.astype(np.int64), device=self.device)] = torch.as_tensor(x.data, device=self.device)
            scores = torch.zeros((self.queries.shape[0], size), dtype=torch.float32, device=self.device)
            for column in range(self.features.shape[1]):
                # Keep multiply and add as distinct eager operations (no compilation).
                contribution = dense.index_select(0, self.features[:, column]) * self.weights[:, column, None]
                scores.add_(contribution)
            del dense
            ids = torch.as_tensor(self.lexical_ids[offset:offset+size], device=self.device).expand(scores.shape[0], -1)
            scores.masked_fill_(ids == self.excluded_gpu[:, None], -torch.inf)
            scores = torch.cat((self.scores, scores), dim=1)
            ids = torch.cat((self.ids, ids), dim=1)
            # Stable lexicographic sort: score descending, original document ID ascending.
            by_id = torch.argsort(ids, dim=1, stable=True)
            scores, ids = scores.gather(1, by_id), ids.gather(1, by_id)
            by_score = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :self.k]
            self.scores, self.ids = scores.gather(1, by_score), ids.gather(1, by_score)

    def finish(self):
        scores, ids = self.scores.cpu().numpy(), self.ids.cpu().numpy()
        return [[(str(self.lexical_names[j]), float(s)) for j, s in zip(row_ids, row_scores) if np.isfinite(s)]
                for row_ids, row_scores in zip(ids, scores)], self.visits
