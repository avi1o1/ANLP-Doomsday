"""Validate completed retrieval artifacts before explicitly reusing an older run.

Run with the new environment, before submitting work. Original provenance is never
rewritten. Incomplete evaluations are moved aside; their files remain available.
"""

import argparse
import time
from pathlib import Path

from src.artifacts import atomic_json, file_hash, jsonl, provenance, read_json
from src.config import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--encoding", action="store_true", help="Also validate fitting and encoded shards")
    args = parser.parse_args()
    root = args.output_root.resolve()
    approved, archived = {}, []
    folders = ["datasets", "baselines", "baseline_evaluations"]
    if args.encoding:
        folders += ["sample", "bases", "encoded"]
    for subtree in folders:
        for path in sorted((root / subtree).rglob("manifest.json")):
            manifest = read_json(path)
            if manifest["status"] != "complete":
                if subtree == "baseline_evaluations" or (subtree == "encoded" and path.parent.parent.name == "shards"):
                    target = root / "resume_archive" / str(time.time_ns()) / path.parent.relative_to(root)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    path.parent.rename(target)
                    archived.append(str(target))
                    continue
                raise ValueError(f"Incomplete representation requires a separate recovery review: {path}")
            for output in manifest["outputs"]:
                item = path.parent / output["path"]
                if file_hash(item) != output["sha256"]:
                    raise ValueError(f"Corrupt artifact: {item}")
            if subtree == "baseline_evaluations":
                result = read_json(path.parent / "result.json")
                records = [r for p in sorted(path.parent.glob("batch-*.json")) for r in read_json(p)]
                ids = [r["query_id"] for r in records]
                collection = Path(manifest["parents"][1]["path"])
                judged = {r["query_id"] for r in jsonl(collection / "qrels.jsonl")}
                expected = [r["query_id"] for r in jsonl(collection / "queries.jsonl") if r["query_id"] in judged]
                if ids != expected or len(ids) != result["examples"] or len(ids) != len(set(ids)):
                    raise ValueError(f"Incomplete or duplicate evaluation queries: {path}")
            approved[str(path.parent)] = digest(manifest)
    if not approved:
        raise ValueError("No completed artifacts found")
    target = root / ("encoding_resume_approval.json" if args.encoding else "resume_approval.json")
    atomic_json(target, {"manifests": approved, "archived": archived,
                         "new_provenance": provenance(), "created": time.time()})
    print(f"Verified {len(approved)} completed artifacts; reuse audit: {target}", flush=True)


if __name__ == "__main__":
    main()
