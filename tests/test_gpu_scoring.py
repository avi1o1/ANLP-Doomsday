import numpy as np
import pytest
from scipy import sparse

from src.retrieval.gpu_scoring import BatchedScorer
from src.retrieval.index import InvertedIndex, merge_rankings
from src.scoring import CollectionStats, configurations


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('switches', configurations())
def test_batched_ordered_scoring_exact_sharded_parity(device, switches):
    import torch
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    rng = np.random.default_rng(3)
    values = rng.uniform(0, 3, (33, 17)).astype(np.float32)
    values[rng.random(values.shape) < .8] = 0
    values[0] = 0
    values[1] = values[2]
    documents = sparse.csr_matrix(values)
    queries = sparse.csr_matrix(np.concatenate([values[:9], np.zeros((1, 17), dtype=np.float32)]))
    names = [str(i) for i in range(33)]  # Lexical order differs from numeric order.
    stats = CollectionStats.empty(17)
    stats.update(documents)
    excluded = ['2'] + [None] * (queries.shape[0] - 1)
    scorer = BatchedScorer(queries, names, 12, excluded, device)
    reference = [[] for _ in range(queries.shape[0])]
    visits = np.zeros(queries.shape[0], dtype=np.int64)
    for start, end in ((0, 11), (11, 21), (21, 33)):
        index = InvertedIndex(documents[start:end], names[start:end], stats, switches)
        scorer.add(index, start)
        for row in range(queries.shape[0]):
            ranking, count = index.search(queries[row], 12, excluded[row])
            reference[row] = merge_rankings([reference[row], ranking], 12)
            visits[row] += count
    rankings, actual_visits = scorer.finish()
    assert rankings == reference
    np.testing.assert_array_equal(actual_visits, visits)
