"""Nonnegative feature scoring with independent, budget-preserving switches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
from scipy import sparse


def nonnegative_csr(values) -> sparse.csr_matrix:
    x = sparse.csr_matrix(values, dtype=np.float32, copy=True)
    x.sum_duplicates()
    x.eliminate_zeros()
    x.sort_indices()
    if not np.isfinite(x.data).all() or (x.data < 0).any():
        raise ValueError("Features must be finite and nonnegative")
    return x


def sparsify(values, budget: int | None) -> sparse.csr_matrix:
    x = nonnegative_csr(values)
    if budget is None:
        return x
    if budget <= 0:
        raise ValueError("Feature budget must be positive")
    counts = np.diff(x.indptr)
    if not counts.size or counts.max() <= budget:
        return x
    keep = np.empty(int(np.minimum(counts, budget).sum()), dtype=np.int64)
    cursor = 0
    for row in range(x.shape[0]):
        a, b = x.indptr[row], x.indptr[row + 1]
        if b - a <= budget:
            keep[cursor:cursor + b - a] = np.arange(a, b)
            cursor += b - a
            continue
        data = x.data[a:b]
        # Partition to the cutoff value, then order only the entries tied with it.
        # Feature ID is the deterministic secondary key inside that band.
        threshold = data[np.argpartition(-data, budget - 1)[:budget]].min()
        above = np.flatnonzero(data > threshold)
        band = np.flatnonzero(data == threshold)
        band = band[np.argsort(x.indices[a:b][band], kind="stable")[:budget - len(above)]]
        keep[cursor:cursor + budget] = a + np.concatenate((above, band))
        cursor += budget
    rows = np.repeat(np.arange(x.shape[0]), np.minimum(counts, budget))
    result = sparse.csr_matrix((x.data[keep], (rows, x.indices[keep])), shape=x.shape, dtype=np.float32)
    result.sort_indices()
    return result


@dataclass(frozen=True)
class Switches:
    idf: bool = False
    saturation: bool = False
    length: bool = False
    k1: float = 1.2
    b: float = 0.75

    def __post_init__(self):
        if not np.isfinite(self.k1) or self.k1 <= 0 or not 0 <= self.b <= 1:
            raise ValueError("Require finite k1 > 0 and 0 <= b <= 1")

    @property
    def key(self):
        return f"{int(self.idf)}{int(self.saturation)}{int(self.length)}"

    @classmethod
    def parse(cls, key: str, k1=1.2, b=0.75):
        if len(key) != 3 or set(key) - {"0", "1"}:
            raise ValueError("Switch configuration must contain three bits (IDF, saturation, length)")
        return cls(*(c == "1" for c in key), k1=k1, b=b)

    def to_dict(self):
        return asdict(self)


def configurations(k1=1.2, b=0.75):
    return [Switches(*bits, k1=k1, b=b) for bits in product((False, True), repeat=3)]


def smoothed_idf(counts, opportunities):
    counts, opportunities = np.broadcast_arrays(
        np.asarray(counts, dtype=np.float64), np.asarray(opportunities, dtype=np.float64)
    )
    if not np.isfinite(counts).all() or not np.isfinite(opportunities).all():
        raise ValueError("Frequency counts must be finite")
    if (counts < 0).any() or (counts > opportunities).any() or (opportunities < 0).any():
        raise ValueError("Require 0 <= occurrence count <= opportunities")
    return np.log1p((opportunities - counts + 0.5) / (counts + 0.5))


@dataclass
class CollectionStats:
    n_items: int
    df: np.ndarray
    mass: np.ndarray
    length_sum: float
    nnz_sum: int

    @classmethod
    def empty(cls, vocabulary_size):
        return cls(0, np.zeros(vocabulary_size, dtype=np.int64),
                   np.zeros(vocabulary_size, dtype=np.float64), 0.0, 0)

    def update(self, values):
        x = nonnegative_csr(values)
        if x.shape[1] != len(self.df):
            raise ValueError("Vocabulary size mismatch")
        self.n_items += x.shape[0]
        self.df += np.bincount(x.indices, minlength=x.shape[1])
        self.mass += np.asarray(x.sum(axis=0), dtype=np.float64).ravel()
        self.length_sum += float(x.data.sum(dtype=np.float64))
        self.nnz_sum += x.nnz

    @property
    def avg_length(self):
        return self.length_sum / self.n_items if self.n_items else 0.0

    @property
    def idf(self):
        return smoothed_idf(self.df, self.n_items)

    def to_dict(self):
        return {"n_items": self.n_items, "df": self.df.tolist(), "mass": self.mass.tolist(),
                "length_sum": self.length_sum, "nnz_sum": self.nnz_sum}

    @classmethod
    def from_dict(cls, data):
        return cls(data["n_items"], np.asarray(data["df"], dtype=np.int64),
                   np.asarray(data["mass"], dtype=np.float64), data["length_sum"], data["nnz_sum"])


def transform_documents(values, stats: CollectionStats, switches: Switches):
    x = nonnegative_csr(values)
    if x.shape[1] != len(stats.df):
        raise ValueError("Vocabulary size mismatch")
    lengths = np.asarray(x.sum(axis=1)).ravel().astype(np.float64)
    ell = np.ones(x.shape[0], dtype=np.float64)
    if switches.length and stats.avg_length > 0:
        ell = 1 - switches.b + switches.b * lengths / stats.avg_length
    row_ell = np.repeat(ell, np.diff(x.indptr))
    f = x.data.astype(np.float64)
    if switches.saturation:
        f = (switches.k1 + 1) * f / (f + switches.k1 * row_ell)
    else:
        f = f / row_ell  # empty rows have no entries, even when b=1
    if switches.idf:
        f *= stats.idf[x.indices]
    x.data = f.astype(np.float32)
    return x


def exhaustive_scores(queries, documents, stats, switches):
    q = nonnegative_csr(queries)
    return (q @ transform_documents(documents, stats, switches).T).toarray()


def stable_topk(scores, ids, k, exclude_id=None):
    """Exact top-k, including deterministic handling of ties at the boundary."""
    values = np.asarray(scores)
    if not np.isfinite(values).all():
        raise ValueError("Ranking scores must be finite")
    if len(values) != len(ids) or k < 0:
        raise ValueError("Invalid ranking dimensions or k")
    names = np.asarray(ids, dtype=str)
    eligible = np.flatnonzero(names != exclude_id) if exclude_id is not None else np.arange(len(values))
    k = min(k, len(eligible))
    if k == 0:
        return np.empty(0, dtype=np.int64)
    if k < len(eligible):
        threshold = np.partition(values[eligible], len(eligible) - k)[len(eligible) - k]
        candidate = eligible[values[eligible] >= threshold]
    else:
        candidate = eligible
    return candidate[np.lexsort((names[candidate], -values[candidate]))[:k]]
