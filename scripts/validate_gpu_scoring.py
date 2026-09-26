"""Production exact GPU ranking parity and end-to-end scoring benchmark."""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from src.artifacts import atomic_json, file_hash, read_json
from src.config import digest, load_config
from src.retrieval.gpu_scoring import BatchedScorer
from src.retrieval.index import InvertedIndex, merge_rankings
from src.retrieval.pipeline import encoded_path, index_path
from src.scoring import CollectionStats, configurations, sparsify


def measure(shards, queries, stats, switches, gpu):
    torch.cuda.synchronize()
    tick = time.perf_counter()
    names = np.concatenate([np.asarray(read_json(s['ids']), dtype=str) for s in shards])
    if gpu:
        scorer = BatchedScorer(queries, names)
        offset = 0
        for shard in shards:
            index = InvertedIndex(sparse.load_npz(shard['vectors']), read_json(shard['ids']), stats, switches)
            scorer.add(index, offset)
            offset += len(index.doc_ids)
        result, visits = scorer.finish()
    else:
        result, visits = [], []
        for start in range(0, queries.shape[0], 32):
            end = min(start+32, queries.shape[0])
            rankings = [[] for _ in range(end-start)]
            counts = np.zeros(end-start, dtype=np.int64)
            for shard in shards:
                index = InvertedIndex(sparse.load_npz(shard['vectors']), read_json(shard['ids']), stats, switches)
                for row in range(start, end):
                    ranking, count = index.search(queries[row])
                    rankings[row-start] = merge_rankings([rankings[row-start], ranking])
                    counts[row-start] += count
            result.extend(rankings)
            visits.extend(counts)
    torch.cuda.synchronize()
    return result, visits, time.perf_counter()-tick


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('previous_inventory')
    parser.add_argument('result')
    args = parser.parse_args()
    config = load_config(args.config)
    config['output_root'] = os.environ.get('OUTPUT_ROOT', config['output_root'])
    comparisons = 0
    torch.cuda.reset_peak_memory_stats()
    for basis in ('identity', 'gaussian', 'pca', 'kmeans', 'pq', 'rq', 'sae'):
        for granularity in ('pooled', 'token'):
            index = index_path(config, 'msmarco_dev', basis, granularity, 0, 'q32_d128')
            stats = CollectionStats.from_dict(read_json(index / 'statistics.json'))
            query_dir = encoded_path(config, 'msmarco_dev', basis, granularity, 0, 'queries')
            first = Path(read_json(query_dir / 'shards.json')[0]['path'])
            queries = sparsify(sparse.load_npz(first / 'vectors.npz')[:4], 32)
            shards = read_json(index / 'shards.json')[:4]
            for switches in configurations():
                reference, visits, _ = measure(shards, queries, stats, switches, False)
                actual, gpu_visits, _ = measure(shards, queries, stats, switches, True)
                assert reference == actual, (basis, granularity, switches)
                np.testing.assert_array_equal(visits, gpu_visits)
                comparisons += len(reference)
            print('Exact GPU parity:', basis, granularity, flush=True)
    # A full 512-query group, with the old 32-query loop and new grouped loop.
    index = index_path(config, 'msmarco_dev', 'identity', 'pooled', 0, 'q32_d128')
    stats = CollectionStats.from_dict(read_json(index / 'statistics.json'))
    query_dir = encoded_path(config, 'msmarco_dev', 'identity', 'pooled', 0, 'queries')
    first = Path(read_json(query_dir / 'shards.json')[0]['path'])
    queries = sparsify(sparse.load_npz(first / 'vectors.npz')[:512], 32)
    shards = read_json(index / 'shards.json')[:8]
    reference, visits, cpu_seconds = measure(shards, queries, stats, configurations()[0], False)
    actual, gpu_visits, gpu_seconds = measure(shards, queries, stats, configurations()[0], True)
    assert reference == actual
    np.testing.assert_array_equal(visits, gpu_visits)
    peak = torch.cuda.max_memory_reserved()
    assert peak < 9 * 2**30, peak
    assert gpu_seconds < cpu_seconds, (gpu_seconds, cpu_seconds)
    source = Path(__file__).resolve().parents[1] / 'src'
    inventory = {str(p.relative_to(source)): file_hash(p) for p in sorted(source.rglob('*.py'))}
    result = {'before_source_hash': digest(read_json(args.previous_inventory)['source']),
              'after_source_hash': digest(inventory), 'exact_parity_passed': True,
              'ranking_comparisons': comparisons + len(actual), 'cpu_seconds': cpu_seconds,
              'gpu_seconds': gpu_seconds, 'speedup': cpu_seconds / gpu_seconds,
              'query_group_size': queries.shape[0], 'shards': len(shards), 'peak_reserved_bytes': peak,
              'device': torch.cuda.get_device_name(), 'scope': 'Sampled shards; not full-corpus throughput'}
    atomic_json(args.result, result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
