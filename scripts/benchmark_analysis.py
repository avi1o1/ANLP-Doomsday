"""Synthetic fixed-protocol benchmark; never reads research outcomes."""
import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from src.analysis import ablation_task, fit_predictors, parallel_map
from src.artifacts import atomic_json
from src.diagnostics import ALL_FEATURES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    rng = np.random.default_rng(41)
    rows = []
    languages = ['en'] * 10 + ['hi', 'bn', 'te']
    for corpus, language in enumerate(languages):
        for basis, family in enumerate(('linear', 'linear', 'linear', 'clustered', 'quantized', 'quantized', 'learned')):
            for granularity in ('pooled', 'token'):
                x = rng.uniform(size=6)
                rows.append(dict(row_id=str(len(rows)), family=family, corpus_id=str(min(corpus, 7) if corpus < 10 else corpus),
                    language=language, representation_id=f'{basis}_{granularity}', collection=str(corpus), budget='primary',
                    seed=0, primary_budget=True, diagnostics=dict(zip(ALL_FEATURES, x)),
                    margin=float(2*x[1]-x[4]+x[5]-.7), margin_variance=.1))
    measurements, reference = {}, None
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        jobs = []
        for i in range(8):
            path = root / f'outcomes-{i}.npy'
            np.save(path, rng.random((6980, 8)))
            jobs.append((str(path), str(i), 2000))
        for workers in (1, args.workers):
            os.environ['CSX_ANALYSIS_WORKERS'] = str(workers)
            tick = time.perf_counter()
            effects = list(parallel_map(ablation_task, jobs))
            ablation_seconds = time.perf_counter()-tick
            tick = time.perf_counter()
            model, validations = fit_predictors(rows, root / str(workers), samples=2000)
            predictor_seconds = time.perf_counter()-tick
            payload = (effects, model, validations)
            if reference is not None:
                assert reference == payload, 'Serial/parallel statistical outputs differ'
            reference = payload
            measurements[str(workers)] = dict(ablation_seconds=ablation_seconds, predictor_seconds=predictor_seconds)
            print(workers, measurements[str(workers)], flush=True)
    atomic_json(args.output, {'exact_parity_passed': True, 'measurements': measurements,
        'scope': 'Synthetic: eight 6980-query ablations; 182-row twelve-variant predictor; 2000 bootstrap samples'})


if __name__ == '__main__':
    main()
