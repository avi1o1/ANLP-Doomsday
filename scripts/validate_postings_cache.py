"""Check bit-exact postings/search parity on production shards before migration."""
import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy import sparse

from src.artifacts import atomic_json, file_hash, read_json
from src.config import digest, load_config
from src.retrieval.index import InvertedIndex
from src.retrieval.pipeline import encoded_path, index_path
from src.retrieval.postings_cache import PostingsCache
from src.scoring import CollectionStats, configurations, sparsify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('previous_inventory')
    parser.add_argument('result')
    parser.add_argument('--scratch', required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    config["output_root"] = os.environ.get("OUTPUT_ROOT", config["output_root"])
    reference_seconds = cached_seconds = 0.0
    comparisons = 0
    with tempfile.TemporaryDirectory(prefix='postings-validation-', dir=args.scratch) as directory:
        for basis in ('identity', 'gaussian'):
            for granularity in ('pooled', 'token'):
                index = index_path(config, 'msmarco_dev', basis, granularity, 0, 'q32_d128')
                stats = CollectionStats.from_dict(read_json(index / 'statistics.json'))
                query_dir = encoded_path(config, 'msmarco_dev', basis, granularity, 0, 'queries')
                first_query_shard = Path(read_json(query_dir / 'shards.json')[0]['path'])
                queries = sparsify(sparse.load_npz(first_query_shard / 'vectors.npz')[:4], 32)
                shards = read_json(index / 'shards.json')[:4]
                for switches in configurations():
                    cache = PostingsCache(directory, read_json(index / 'manifest.json'), stats, switches, 2**30, 0)
                    for shard in shards:
                        tick = time.perf_counter()
                        reference = InvertedIndex(sparse.load_npz(shard['vectors']), read_json(shard['ids']), stats, switches)
                        reference_seconds += time.perf_counter() - tick
                        cache.get(shard)  # Cold build, outside warm-load timing.
                        tick = time.perf_counter()
                        cached = cache.get(shard)
                        cached_seconds += time.perf_counter() - tick
                        for field in ('data', 'indices', 'indptr'):
                            np.testing.assert_array_equal(getattr(reference.postings, field), getattr(cached.postings, field))
                        for row in range(queries.shape[0]):
                            assert reference.search(queries[row], 1000) == cached.search(queries[row], 1000)
                            comparisons += 1
    source = Path(__file__).resolve().parents[1] / 'src'
    inventory = {str(p.relative_to(source)): file_hash(p) for p in sorted(source.rglob('*.py'))}
    result = {'before_source_hash': digest(read_json(args.previous_inventory)['source']),
              'after_source_hash': digest(inventory), 'exact_parity_passed': True,
              'ranking_comparisons': comparisons, 'reference_build_seconds': reference_seconds,
              'warm_cache_load_seconds': cached_seconds,
              'scope': '4 representations x 8 switches x 4 production shards x 4 queries; not full-job speedup'}
    atomic_json(args.result, result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
