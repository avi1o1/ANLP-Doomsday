"""Free document vectors whose every planned index is already built.

Once all indexes planned for a (representation, corpus) pair are complete, that
pair's encoded document vectors are never read again: evaluation reads the index
shards and the encoded queries, and takes only metadata from the encoded shards'
manifests. On a machine whose disk cannot hold every encoding and every index at
once, deleting those vectors is what lets indexing finish.

Only documents/.../shards/*/vectors.npz is removed. Manifests, identifiers and
per-document encoding records stay, so every downstream reader still works; query
encodings are never touched. Anything whose indexes are not all complete is left
alone. Every deletion is appended to OUTPUT_ROOT/freed_encodings.jsonl.

The price is that re-encoding would be needed to rebuild one of those indexes, and
the encoding step must not be re-run afterwards: its completed shards would fail
their checksum audit. Run it only after encoding has finished.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from src.artifacts import jsonl, read_json
from src.config import load_config
from src.retrieval.data import corpus_key
from src.retrieval.pipeline import encoded_path, index_path, spec_for


def planned_indexes(output_root: Path):
    """Index tasks of every launched job manifest, keyed by (corpus, representation)."""
    groups = defaultdict(set)
    for tasks in output_root.glob("jobs/*/*/index.jsonl"):
        for task in jsonl(tasks):
            argv = task["argv"]
            value = {argv[i]: argv[i + 1] for i in range(1, len(argv) - 1, 2)}
            groups[(value["--dataset"], value["--basis"], value["--granularity"], int(value["--seed"]))].add(
                value["--budget"])
    return groups


def complete(path: Path):
    manifest = path / "manifest.json"
    return manifest.exists() and read_json(manifest)["status"] == "complete"


def sweep(config, output_root: Path, dry_run: bool, record):
    freed, waiting = 0, 0
    by_corpus = defaultdict(list)
    for (dataset, basis, granularity, seed), budgets in planned_indexes(output_root).items():
        by_corpus[(corpus_key(spec_for(config, dataset)), basis, granularity, seed)].append((dataset, budgets))
    for (corpus, basis, granularity, seed), entries in sorted(by_corpus.items()):
        indexes = [index_path(config, dataset, basis, granularity, seed, budget)
                   for dataset, budgets in entries for budget in budgets]
        if not all(complete(p) for p in indexes):
            waiting += 1
            continue
        documents = encoded_path(config, entries[0][0], basis, granularity, seed, "documents")
        for vectors in sorted(documents.glob("shards/*/vectors.npz")):
            size = vectors.stat().st_size
            if not dry_run:
                vectors.unlink()
                record.write(json.dumps({"path": str(vectors), "bytes": size, "time": time.time()}) + "\n")
                record.flush()
            freed += size
    return freed, waiting


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="report what would be freed, delete nothing")
    parser.add_argument("--loop", type=float, default=0,
                        help="repeat every this many seconds, stopping once no pair is left waiting")
    args = parser.parse_args()
    config = load_config(args.config)
    config["output_root"] = str(args.output_root)
    with (args.output_root / "freed_encodings.jsonl").open("a") as record:
        while True:
            freed, waiting = sweep(config, args.output_root.resolve(), args.dry_run, record)
            verb = "would free" if args.dry_run else "freed"
            print(f"{time.strftime('%H:%M:%S')} {verb} {freed / 1e9:.1f} GB; "
                  f"{waiting} representation-corpus pairs still waiting for indexes", flush=True)
            if not args.loop or (waiting == 0 and not args.dry_run):
                return 0
            time.sleep(args.loop)


if __name__ == "__main__":
    raise SystemExit(main())
