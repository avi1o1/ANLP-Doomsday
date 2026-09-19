import numpy as np
import pytest
from scipy import sparse

from src.diagnostics import collection_diagnostics
from src.retrieval.index import InvertedIndex, merge_rankings, reference_metrics, standard_metrics
from src.scoring import (
    CollectionStats,
    Switches,
    configurations,
    exhaustive_scores,
    smoothed_idf,
    sparsify,
    stable_topk,
    transform_documents,
)


@pytest.mark.parametrize("switches", configurations())
def test_postings_exhaustive_shard_parity(switches):
    d = sparse.csr_matrix([[0, 0, 0, 0], [2, 0, 1, 0], [2, 0, 1, 0], [0, 3, 0, 0]], dtype=np.float32)
    q = sparse.csr_matrix([[1, 0, 4, 2], [0, 0, 0, 0]], dtype=np.float32)
    ids = ["z", "b", "a", "c"]
    stats = CollectionStats.empty(4)
    stats.update(d)
    scores = exhaustive_scores(q, d, stats, switches)
    for i in range(2):
        expected = [(ids[j], float(scores[i, j])) for j in stable_topk(scores[i], ids, 4)]
        full, visits = InvertedIndex(d, ids, stats, switches).search(q[i], 4)
        halves = [InvertedIndex(d[a:b], ids[a:b], stats, switches).search(q[i], 4)[0] for a, b in ((0, 2), (2, 4))]
        assert [r[0] for r in expected] == [r[0] for r in full] == [r[0] for r in merge_rankings(halves, 4)]
        np.testing.assert_allclose([x[1] for x in full], [x[1] for x in expected], rtol=1e-6)
        assert visits >= 0


def test_independent_switches_and_constant_length():
    d = sparse.csr_matrix([[3, 1], [1, 3]], dtype=np.float32)
    stats = CollectionStats.empty(2)
    stats.update(d)
    for saturation in (False, True):
        a = transform_documents(d, stats, Switches(saturation=saturation)).toarray()
        b = transform_documents(d, stats, Switches(saturation=saturation, length=True)).toarray()
        np.testing.assert_allclose(a, b)
    np.testing.assert_array_equal(transform_documents(d, stats, Switches()).toarray(), d.toarray())
    only_length = transform_documents(sparse.csr_matrix([[6, 2]]), stats, Switches(length=True)).toarray()
    np.testing.assert_allclose(only_length, [[6/1.75, 2/1.75]])


def test_ties_empty_unseen_and_invalid():
    np.testing.assert_array_equal(sparsify([[1, 1, 1]], 2).toarray(), [[1, 1, 0]])
    stats = CollectionStats.empty(4)
    stats.update(sparse.csr_matrix((2, 4)))
    diag = collection_diagnostics(stats)
    assert diag["dead_feature_fraction"] == 1 and diag["frequency_slope"] == 0
    assert np.isfinite(stats.idf).all()
    assert sparsify([[0, 0]], None).nnz == 0
    with pytest.raises(ValueError):
        smoothed_idf([2], [1])
    with pytest.raises(ValueError):
        sparsify([[-1, 2]], 1)


def test_identical_id_exclusion_does_not_drop_the_last_eligible_result():
    scores = [5, 4, 3, 2]
    ids = ["self", "a", "b", "c"]
    np.testing.assert_array_equal(stable_topk(scores, ids, 3, "self"), [1, 2, 3])
    assert len(stable_topk([0], ["self"], 1000, "self")) == 0


def test_trec_metric_reference():
    pytest.importorskip("ir_measures")
    ranking = [("c", 5), ("a", 4), ("b", 3)]
    qrels = {"a": 2, "b": 1, "c": 0, "d": 3}
    actual, expected = standard_metrics("q", ranking, qrels), reference_metrics(ranking, qrels)
    for metric in actual:
        assert actual[metric] == pytest.approx(expected[metric])
