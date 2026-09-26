import sqlite3
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from scipy import sparse

from src.artifacts import atomic_json
from src.retrieval.index import InvertedIndex
from src.retrieval.postings_cache import PostingsCache
from src.scoring import CollectionStats, configurations


@pytest.mark.parametrize('switches', configurations())
def test_cached_postings_exact_parity_and_recovery(tmp_path, switches):
    docs = sparse.csr_matrix([[0, 0, 0, 0], [2, 1, 0, 0], [2, 1, 0, 0], [0, 3, 2, 0]], dtype=np.float32)
    ids = ['z', 'b', 'a', 'c']
    stats = CollectionStats.empty(4)
    stats.update(docs)
    sparse.save_npz(tmp_path / 'vectors.npz', docs)
    atomic_json(tmp_path / 'ids.json', ids)
    shard = {'vectors': str(tmp_path / 'vectors.npz'), 'ids': str(tmp_path / 'ids.json')}
    cache = PostingsCache(tmp_path / 'cache', {'parent': 'immutable'}, stats, switches, 100000, 0)
    reference = InvertedIndex(docs, ids, stats, switches)
    for index in (cache.get(shard), cache.get(shard)):
        np.testing.assert_array_equal(index.postings.data, reference.postings.data)
        for values in ([1, 2, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]):
            q = sparse.csr_matrix([values], dtype=np.float32)
            for excluded in (None, 'a'):
                assert index.search(q, 4, excluded) == reference.search(q, 4, excluded)
    assert (cache.hits, cache.misses) == (1, 1)
    next((tmp_path / 'cache').glob('*.npz')).write_bytes(b'corrupted')
    np.testing.assert_array_equal(cache.get(shard).postings.data, reference.postings.data)
    assert cache.misses == 2


def test_cache_bound_and_concurrent_builders(tmp_path):
    docs = sparse.eye(20, format='csr', dtype=np.float32)
    stats = CollectionStats.empty(20)
    stats.update(docs)
    sparse.save_npz(tmp_path / 'vectors.npz', docs)
    atomic_json(tmp_path / 'ids.json', list(map(str, range(20))))
    shard = {'vectors': str(tmp_path / 'vectors.npz'), 'ids': str(tmp_path / 'ids.json')}
    cache_dir = tmp_path / 'cache'
    switch = configurations()[0]
    def build(_):
        cache = PostingsCache(cache_dir, {'parent': 'same'}, stats, switch, 10000, 0)
        return cache.get(shard).postings.toarray()
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(build, range(4)))
    assert len(list(cache_dir.glob('*.npz'))) == 1
    for output in outputs:
        np.testing.assert_array_equal(output, docs.toarray())
    for i in range(20):
        PostingsCache(cache_dir, {'parent': i}, stats, switch, 10000, 0).get(shard)
    assert sum(p.stat().st_size for p in cache_dir.glob('*.npz')) <= 10000
    with sqlite3.connect(cache_dir / 'entries.sqlite') as db:
        assert db.execute('SELECT SUM(bytes) FROM entries').fetchone()[0] <= 10000
    # No capacity: retain exact uncached execution rather than filling the disk.
    cache = PostingsCache(tmp_path / 'disabled', {}, stats, switch, 0, 0)
    np.testing.assert_array_equal(cache.get(shard).postings.toarray(), docs.toarray())
    assert not list((tmp_path / 'disabled').glob('*.npz'))
