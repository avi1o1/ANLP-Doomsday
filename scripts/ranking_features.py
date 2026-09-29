"""Label-free features of the rankings each scoring configuration returns.

The stored evaluation records hold, for every row, configuration and query, the top
100 documents with their scores. None of the features below reads relevance labels:

- disruption_<switch>: mean over the four configuration pairs that differ only in that
  switch of 1 - |A ∩ B| / 10, where A and B are the two top-10 lists;
- pseudo_<reference>_<configuration>: nDCG@10 against pseudo-labels, the top 10 of a
  reference retriever (text BM25, and the dense encoder where its run exists), scored
  as binary relevance;
- nqc_<configuration> and gap_<configuration>: score-shape statistics used in query
  performance prediction, the coefficient of variation of the top-100 scores and
  (s1 - s10) / |s1|.

Each row's features are written to OUT_DIR/<collection>/<representation>/<budget>/<tag>.json
as soon as they are computed, and rows already written are skipped, so the command can be
stopped and started again. At the end all rows are gathered into OUT_DIR.json.

Usage: python -m scripts.ranking_features OUTPUT_ROOT OUT_DIR [--workers N]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np

CONFIGURATIONS = [f"{i:03b}" for i in range(8)]
SWITCHES = {"idf": 0, "saturation": 1, "length": 2}
REFERENCES = ("bm25", "dense")
DEPTH = 10
IDEAL = sum(1 / math.log2(i + 2) for i in range(DEPTH))


def read_rankings(directory: Path, pattern: str):
    rankings = {}
    for path in sorted(directory.glob(pattern)):
        for record in json.loads(path.read_text()):
            rankings[record["query_id"]] = record["ranking"]
    return rankings


@lru_cache(maxsize=8)
def reference(root: str, collection: str, name: str):
    directory = Path(root) / "baseline_evaluations" / collection / name
    if not (directory / "result.json").exists():
        return None
    return {q: {d for d, _ in r[:DEPTH]} for q, r in read_rankings(directory, "batch-*.json").items()}


def pseudo_ndcg(ranking, relevant):
    gain = sum(1 / math.log2(i + 2) for i, (d, _) in enumerate(ranking[:DEPTH]) if d in relevant)
    return gain / IDEAL


def score_shape(ranking):
    scores = np.array([s for _, s in ranking], float)
    if len(scores) < 2 or not np.isfinite(scores).all():
        return float("nan"), float("nan")
    mean = abs(scores.mean())
    nqc = float(scores.std() / mean) if mean > 0 else float("nan")
    top = scores[min(DEPTH, len(scores)) - 1]
    gap = float((scores[0] - top) / abs(scores[0])) if scores[0] != 0 else float("nan")
    return nqc, gap


def row_features(task):
    root, row_dir, out_path = task
    row_dir, out_path = Path(row_dir), Path(out_path)
    collection = row_dir.parts[-4]
    rankings, tops, shapes = {}, {}, {}
    for c in CONFIGURATIONS:
        rankings[c] = read_rankings(row_dir / c, "*.json")
        tops[c] = {q: {d for d, _ in r[:DEPTH]} for q, r in rankings[c].items()}
        shapes[c] = np.nanmean([score_shape(r) for r in rankings[c].values()], axis=0)
    queries = sorted(set.intersection(*(set(tops[c]) for c in CONFIGURATIONS)))
    features = {"queries": len(queries)}
    for switch, bit in SWITCHES.items():
        changes = []
        for c in CONFIGURATIONS:
            if c[bit] == "1":
                continue
            other = c[:bit] + "1" + c[bit + 1:]
            changes.extend(1 - len(tops[c][q] & tops[other][q]) / DEPTH for q in queries)
        features[f"disruption_{switch}"] = float(np.mean(changes))
    for name in REFERENCES:
        labels = reference(root, collection, name)
        if labels is None:
            continue
        shared = [q for q in queries if q in labels]
        features[f"{name}_queries"] = len(shared)
        for c in CONFIGURATIONS:
            features[f"pseudo_{name}_{c}"] = float(np.mean([pseudo_ndcg(rankings[c][q], labels[q])
                                                            for q in shared]))
    for c in CONFIGURATIONS:
        features[f"nqc_{c}"], features[f"gap_{c}"] = (float(v) for v in shapes[c])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.parent / f"{out_path.name}.tmp"
    temporary.write_text(json.dumps(features) + "\n")
    os.replace(temporary, out_path)
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    tasks = []
    for row_dir in sorted((args.output_root / "evaluations").glob("*/*/*/*")):
        if not all((row_dir / c).is_dir() for c in CONFIGURATIONS) or not (row_dir / "result.json").exists():
            continue
        relative = row_dir.relative_to(args.output_root / "evaluations")
        out_path = args.out_dir / relative.parent / f"{relative.name}.json"
        if not out_path.exists():
            tasks.append((str(args.output_root), str(row_dir), str(out_path)))
    print(f"{len(tasks)} rows to do", flush=True)
    with Pool(args.workers) as pool:
        for done, _ in enumerate(pool.imap_unordered(row_features, tasks, chunksize=1), 1):
            if done % 50 == 0 or done == len(tasks):
                print(f"{done}/{len(tasks)}", flush=True)
    gathered = []
    for path in sorted(args.out_dir.glob("*/*/*/*.json")):
        collection, representation, budget = path.parts[-4:-1]
        gathered.append({"collection": collection, "representation": representation, "budget": budget,
                         "tag": path.name.removesuffix(".json"), **json.loads(path.read_text())})
    Path(f"{args.out_dir}.json").write_text(json.dumps(gathered, indent=0) + "\n")
    print(f"{len(gathered)} rows gathered", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
