import numpy as np
import pytest

from src.artifacts import StopFlag
from src.retrieval.bases import Basis

SPECS = [dict(kind="identity"), dict(kind="random", components=8), dict(kind="pca", components=4),
         dict(kind="kmeans", clusters=4, assignments=2, backend="sklearn"),
         dict(kind="pq", subspaces=2, codes=4, backend="sklearn"),
         dict(kind="rq", stages=2, codes=4, backend="sklearn"),
         dict(kind="sae", features=16, active=4, epochs=2, batch_size=16, micro_batch_size=4, checkpoint_steps=1)]


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s["kind"])
def test_basis_nonnegative_serialization_and_quantized_length(spec, tmp_path):
    x = np.random.default_rng(0).normal(size=(40, 8)).astype(np.float32)
    with StopFlag() as stop:
        basis = Basis(spec, 8).fit(x[:32], x[32:], tmp_path / "checkpoint", stop)
    values = basis.transform(x)
    assert values.shape == (40, basis.vocabulary_size)
    assert (values.data >= 0).all() and np.isfinite(values.data).all()
    basis.save(tmp_path)
    np.testing.assert_allclose(Basis.load(tmp_path).transform(x).toarray(), values.toarray())
    if spec["kind"] in {"pq", "rq"}:
        np.testing.assert_allclose(values.sum(1), 2)
        ids = values[0].indices
        assert ids[0] < 4 <= ids[1]
    if spec["kind"] == "sae":
        assert max(np.diff(values.indptr)) <= 4
        np.testing.assert_allclose(np.linalg.norm(basis.arrays["decoder"], axis=0), 1, atol=1e-5)


def test_pool_then_rectify_differs_from_token_sum():
    states = np.array([[1, -2], [-1, 2]], dtype=np.float32)
    basis = Basis({"kind": "identity"}, 2)
    pooled = basis.aggregate(states.mean(0)[None], [states], "pooled")
    token = basis.aggregate(states.mean(0)[None], [states], "token")
    assert pooled.nnz == 0 and token.nnz == 4


def test_faiss_codebook_backend_and_dense_reference():
    faiss = pytest.importorskip("faiss")
    from src.retrieval.bases import fit_centers

    previous = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    try:
        x = np.random.default_rng(12).normal(size=(256, 8)).astype(np.float32)
        centers = fit_centers(x, 4, 0)
        assert centers.shape == (4, 8) and np.isfinite(centers).all()
        index = faiss.IndexFlatIP(8)
        index.add(x)
        distances, ids = index.search(x[:3], len(x))
        scores = np.empty_like(distances)
        np.put_along_axis(scores, ids, distances, axis=1)
        np.testing.assert_allclose(scores, x[:3] @ x.T, atol=3e-6)
    finally:
        faiss.omp_set_num_threads(previous)
