"""Explicit reproduction status and evaluation of exported external feature shards."""

import sqlite3
from pathlib import Path

import numpy as np
from scipy import sparse

from src.analysis import paired_effect
from src.artifacts import Artifact, atomic_json, file_hash, jsonl, read_json
from src.config import digest, root
from src.retrieval.index import InvertedIndex, merge_rankings, qrels_mapping, standard_metrics
from src.retrieval.pipeline import chunks, prepared_paths
from src.scoring import CollectionStats, configurations, nonnegative_csr


def reproduction_status(config):
    spec = config.get("external", {}).get("latent_terms", {})
    checkpoint = spec.get("checkpoint")
    preprocessing = spec.get("preprocessing")
    available = bool(checkpoint and Path(checkpoint).is_file() and preprocessing)
    status = {"experiment": "latent_terms", "matching_artifacts_present": available,
              "claim": "reproduction_attempt_pending_verification" if available else "conceptual_replication",
              "checkpoint_sha256": file_hash(checkpoint) if available else None,
              "preprocessing": preprocessing,
              "reason": None if available else "Matching published checkpoints and preprocessing are not configured.",
              "conceptual_comparison": "Controlled E5 SAE pooled/token all-off versus all-on; different backbone and protocol.",
              "exact_reproduction_completed": False,
              "positive_result_is_correctness_gate": False}
    atomic_json(root(config) / "external" / "latent_terms_status.json", status)
    return status


def evaluate_external(config, dataset, manifest_path, stop):
    """Export format is documented in context/IMPLEMENTATION.md; IDs must cover the whole corpus."""
    manifest_path = Path(manifest_path).resolve()
    spec = read_json(manifest_path)
    if not spec.get("checkpoint_sha256") or not spec.get("preprocessing") or not spec.get("model"):
        raise ValueError("External exports require model/checkpoint digest and exact preprocessing metadata")
    collection, corpus = prepared_paths(config, dataset)
    directory = root(config) / "external" / dataset / digest(spec)[:16]
    def resolve(path):
        return manifest_path.parent / path
    files = [resolve(shard[key]) for side in ("documents", "queries") for shard in spec[side] for key in ("matrix", "ids")]
    with Artifact(directory, {"export": spec, "files": {str(p): file_hash(p) for p in files}}, [collection, corpus]) as work:
        if work.reused:
            return directory
        # Disk-backed coverage and uniqueness check across all feature shards.
        coverage = directory / "coverage.sqlite"
        if coverage.exists():
            coverage.unlink()
        stats = None
        with sqlite3.connect(coverage) as db:
            db.execute("CREATE TABLE encoded (id TEXT PRIMARY KEY)")
            db.execute("ATTACH DATABASE ? AS corpus", (str(corpus / "ids.sqlite"),))
            for shard in spec["documents"]:
                stop.check()
                ids = read_json(resolve(shard["ids"]))
                matrix = nonnegative_csr(sparse.load_npz(resolve(shard["matrix"])))
                if len(ids) != matrix.shape[0]:
                    raise ValueError("External document IDs do not match matrix rows")
                db.executemany("INSERT INTO encoded VALUES (?)", ((i,) for i in ids))
                if stats is None:
                    stats = CollectionStats.empty(matrix.shape[1])
                stats.update(matrix)
            mismatch = db.execute("SELECT id FROM encoded EXCEPT SELECT id FROM corpus.documents").fetchone()
            missing = db.execute("SELECT id FROM corpus.documents EXCEPT SELECT id FROM encoded").fetchone()
            if mismatch or missing:
                raise ValueError("External features must cover exactly the complete configured corpus")
        queries, ids = [], []
        for shard in spec["queries"]:
            matrix = nonnegative_csr(sparse.load_npz(resolve(shard["matrix"])))
            names = read_json(resolve(shard["ids"]))
            if len(names) != matrix.shape[0] or matrix.shape[1] != len(stats.df):
                raise ValueError("External query shape or vocabulary mismatch")
            ids.extend(names)
            queries.append(matrix)
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate external query IDs")
        qrels = qrels_mapping(jsonl(collection / "qrels.jsonl"))
        if set(qrels) - set(ids):
            raise ValueError("Missing judged external query IDs")
        queries = sparse.vstack(queries, format="csr")
        lookup = {qid: i for i, qid in enumerate(ids)}
        judged = sorted(qrels)
        outcomes, outputs = np.zeros((len(judged), 8)), []
        for column, switches in enumerate(configurations()):
            for batch_number, batch in enumerate(chunks(judged, config.get("evaluation_batch_size", 32))):
                stop.check()
                path = directory / switches.key / f"{batch_number:06d}.json"
                if not path.exists():
                    rankings = [[] for _ in batch]
                    for shard in spec["documents"]:
                        index = InvertedIndex(sparse.load_npz(resolve(shard["matrix"])),
                            read_json(resolve(shard["ids"])), stats, switches)
                        for i, qid in enumerate(batch):
                            partial = index.search(queries[lookup[qid]], 1000,
                                qid if config["datasets"][dataset].get("exclude_identical_ids") else None)[0]
                            rankings[i] = merge_rankings([rankings[i], partial], 1000)
                    atomic_json(path, [{"query_id": qid, "ranking": rankings[i],
                                       "metrics": standard_metrics(qid, rankings[i], qrels[qid])}
                                      for i, qid in enumerate(batch)])
                start = batch_number * config.get("evaluation_batch_size", 32)
                outcomes[start:start+len(batch), column] = [r["metrics"]["ndcg@10"] for r in read_json(path)]
                outputs.append(str(path.relative_to(directory)))
        effect = paired_effect(outcomes, config["analysis"]["bootstrap_samples"])
        atomic_json(directory / "result.json", {"external": True, "collection": dataset,
                    "reproduction_claim": spec.get("reproduction_claim", "conceptual_replication"),
                    "artifact_match_evidence": spec.get("artifact_match_evidence"), **effect})
        atomic_json(directory / "statistics.json", stats.to_dict())
        np.save(directory / "outcomes.npy", outcomes)
        work.complete(outputs + ["result.json", "statistics.json", "outcomes.npy", "coverage.sqlite"])
    return directory
