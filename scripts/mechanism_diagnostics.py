"""Label-free diagnostics of each built index, matched to what each scoring switch acts on.

Saturation compresses large feature values within a document; length normalisation
rescales documents by their mass; neither needs relevance labels to be anticipated.
For every index under OUTPUT_ROOT/indexes this reads the stored document vectors and
records:

- value_cv: coefficient of variation of the stored feature values (sampled 1 in 7);
- peak_ratio: mean over documents of the largest value over the mean value;
- top_share: mean share of a document's mass in its largest feature;
- length_cv and length_log_sd: dispersion of document mass;
- nnz_cv: dispersion of stored features per document.

Usage: python -m scripts.mechanism_diagnostics --config CONFIG --output-root ROOT OUT_JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from src.config import load_config
from src.retrieval.data import corpus_key
from src.retrieval.pipeline import spec_for


def diagnose_index(directory: Path):
    lengths, nnz, peaks, tops, values = [], [], [], [], []
    for shard in sorted(directory.glob("*.npz")):
        x = sparse.load_npz(shard).tocsr()
        mass, count = np.asarray(x.sum(axis=1)).ravel(), np.diff(x.indptr)
        keep = count > 0
        largest = x.max(axis=1).toarray().ravel()
        lengths.append(mass[keep])
        nnz.append(count[keep])
        peaks.append(largest[keep] / (mass[keep] / count[keep]))
        tops.append(largest[keep] / mass[keep])
        values.append(x.data[::7])
    length, count, peak, top, value = (np.concatenate(a) for a in (lengths, nnz, peaks, tops, values))
    return {"length_cv": float(length.std() / length.mean()), "length_log_sd": float(np.log(length).std()),
            "nnz_cv": float(count.std() / count.mean()) if count.mean() else 0.0,
            "peak_ratio": float(peak.mean()), "top_share": float(top.mean()),
            "value_cv": float(value.std() / value.mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    names = {corpus_key(spec_for(config, n)): n for n, d in config["datasets"].items()
             if d.get("enabled", True) and d["role"] == "evaluate"}
    records = []
    for directory in sorted((args.output_root / "indexes").glob("*/*/*")):
        representation, key, budget = directory.parts[-3], directory.parts[-2], directory.parts[-1]
        if key not in names or not list(directory.glob("*.npz")):
            continue
        basis, seed = representation.rsplit("__s", 1)
        records.append({"representation": basis, "seed": int(seed), "budget": budget, "collection": names[key],
                        **diagnose_index(directory)})
    args.output.write_text(json.dumps(records, indent=1) + "\n")
    print(f"{len(records)} indexes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
