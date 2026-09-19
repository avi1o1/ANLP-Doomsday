"""Mechanism controls and linear-map numerical examples, without invented qrels."""

import numpy as np
from scipy import sparse

from src.artifacts import atomic_json
from src.config import root
from src.diagnostics import collection_diagnostics
from src.scoring import CollectionStats, Switches, exhaustive_scores


def run_controls(config):
    # Cyclic binary features give exactly uniform df and constant document length.
    n, vocabulary, active = 128, 64, 8
    documents = np.zeros((n, vocabulary), dtype=np.float32)
    for row in range(n):
        documents[row, (row + np.arange(active)) % vocabulary] = 1
    stats = CollectionStats.empty(vocabulary)
    stats.update(sparse.csr_matrix(documents))
    q = documents[:4]
    raw = exhaustive_scores(q, documents, stats, Switches())
    length = exhaustive_scores(q, documents, stats, Switches(length=True))
    frequency = exhaustive_scores(q, documents, stats, Switches(idf=True))
    checks = {"constant_length_max_error": float(np.max(np.abs(raw-length))),
              "uniform_idf_rescaling_max_error": float(np.max(np.abs(frequency-raw*stats.idf[0]))),
              "diagnostics": collection_diagnostics(stats), "fixture": True,
              "purpose": "mechanism algebra; no retrieval-quality or corpus-generalization claim"}
    if checks["constant_length_max_error"] > 1e-6 or checks["uniform_idf_rescaling_max_error"] > 1e-5:
        raise AssertionError("Mechanism control failed")
    q_linear = np.array([1.0, 1.0])
    docs = np.array([[2.0, 0.0], [0.0, 1.5]])
    c = np.diag([1.0, 2.0])
    d = np.diag([1.0, 1.0])
    direct = (q_linear @ c.T) @ d @ (docs @ c.T).T
    identity = q_linear @ c.T @ d @ c @ docs.T
    checks["theory"] = {"original_scores": (q_linear @ docs.T).tolist(),
                         "feature_scores": direct.tolist(), "bilinear_scores": identity.tolist(),
                         "identity_error": float(np.max(np.abs(direct-identity))),
                         "ranking_reversal": int(np.argmax(q_linear @ docs.T)) != int(np.argmax(identity))}
    atomic_json(root(config) / "controls" / "mechanisms.json", checks)
    return checks
