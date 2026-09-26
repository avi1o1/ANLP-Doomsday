"""Approve partial evaluation checkpoints after a launcher-only source change.

The source inventory must be captured before deployment. Scientific source and
package versions must be identical. Run only after all old workers have stopped,
then run audit_resume --evaluation to verify completed artifacts.
"""
import argparse
from pathlib import Path

from src.artifacts import atomic_json, file_hash, jsonl, provenance, read_json
from src.config import digest


def migrate(root, previous, batch_size=32, evaluator_validation=None):
    source = Path(__file__).resolve().parents[1] / "src"
    current = {str(p.relative_to(source)): file_hash(p) for p in sorted(source.rglob("*.py"))}
    before = previous["source"]
    changed = {p for p in set(before) | set(current) if before.get(p) != current.get(p)}
    allowed = {"launch.py"}
    if evaluator_validation is not None:
        if (evaluator_validation["before_source_hash"] != digest(before) or
                evaluator_validation["after_source_hash"] != digest(current) or
                not evaluator_validation.get("exact_parity_passed")):
            raise ValueError("Evaluator validation does not match deployed sources")
        allowed |= {"retrieval/pipeline.py", "retrieval/postings_cache.py", "retrieval/gpu_scoring.py"}
    if not changed <= allowed:
        raise ValueError("Migration requires identical scientific source except validated execution changes")
    new = provenance()
    migrated = []
    for path in sorted((root / "evaluations").rglob("manifest.json")):
        old = read_json(path)
        if old["status"] == "complete":
            continue
        if old["status"] not in ("interrupted", "incomplete"):
            raise ValueError(f"Evaluation did not stop safely: {path}")
        if old["provenance"]["source_hash"] != digest(before) or old["provenance"]["versions"] != new["versions"]:
            raise ValueError(f"Unexpected checkpoint provenance: {path}")
        for parent in old["parents"]:
            manifest = read_json(Path(parent["path"]) / "manifest.json")
            if (manifest["status"] != "complete" or manifest["fingerprint"] != parent["fingerprint"]
                    or digest(manifest["outputs"]) != parent["outputs_hash"]):
                raise ValueError(f"Checkpoint parent changed: {parent['path']}")
        queries, collection = [Path(p["path"]) for p in old["parents"][1:]]
        judged = {r["query_id"] for r in jsonl(collection / "qrels.jsonl")}
        ids = [qid for s in read_json(queries / "shards.json")
               for qid in read_json(Path(s["path"]) / "ids.json") if qid in judged]
        batches = {}
        for batch in sorted(path.parent.glob("[01][01][01]/*.json")):
            start = int(batch.stem)
            records = read_json(batch)
            if (start % batch_size or start >= len(ids) or
                    [r["query_id"] for r in records] != ids[start:start + batch_size]):
                raise ValueError(f"Invalid query checkpoint: {batch}")
            batches[str(batch.relative_to(path.parent))] = file_hash(batch)
        audit = {"original_manifest": old, "batch_sha256": batches, "new_provenance": new,
                 "reason": "validated execution change; retained timings keep original meaning",
                 "evaluator_validation": evaluator_validation}
        previous_audit = path.parent / "scheduler_migration.json"
        if previous_audit.exists():
            atomic_json(path.parent / ("migration_history_" + file_hash(previous_audit)[:16] + ".json"),
                        read_json(previous_audit))
        atomic_json(previous_audit, audit)
        updated = {**old, "provenance": new, "fingerprint": digest({"config": old["config"],
            "parents": old["parents"], "source_hash": new["source_hash"], "versions": new["versions"]})}
        atomic_json(path, updated)
        migrated.append(str(path.parent))
    return migrated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("previous_inventory", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--evaluator-validation", type=Path)
    args = parser.parse_args()
    print(migrate(args.output_root, read_json(args.previous_inventory), args.batch_size,
                  read_json(args.evaluator_validation) if args.evaluator_validation else None))


if __name__ == "__main__":
    main()
