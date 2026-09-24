"""Compare old and shared GPU encoding on real documents and fitted dictionaries."""

import argparse
import time
from itertools import islice

import numpy as np
import torch
from scipy import sparse

from src.artifacts import atomic_json, jsonl, provenance
from src.config import load_config
from src.retrieval.bases import Basis
from src.retrieval.data import Document
from src.retrieval.encoding import make_encoder
from src.retrieval.gpu_bases import TensorBasis
from src.retrieval.pipeline import basis_path, chunks, prepared_paths


def differences(reference, actual):
    a, b = reference.toarray(), actual.toarray()
    return {"relative_l1": float(np.abs(a-b).sum() / max(np.abs(a).sum(), 1e-12)),
            "max_absolute": float(np.abs(a-b).max()),
            "support_disagreements": int(np.count_nonzero((a > 0) != (b > 0)))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--long-documents", action="store_true")
    args = p.parse_args()
    config = load_config(args.config)
    config["output_root"] = args.output_root
    torch.set_num_threads(9)
    torch.backends.cuda.matmul.allow_tf32 = False
    texts, sample_ids = [], []
    for dataset, count in [("msmarco_dev", 32), ("miracl_hi", 8), ("miracl_te", 8)]:
        _, corpus = prepared_paths(config, dataset)
        records = islice(jsonl(corpus / "documents.jsonl"), 2000)
        documents = (Document(**r) for r in records)
        if args.long_documents:
            documents = (d for d in documents if len(d.content) >= 2000)
        for document in islice(documents, count):
            texts.append(document.content)
            sample_ids.append({"dataset": dataset, "doc_id": document.doc_id})
    bases = {}
    for spec in config["bases"]:
        if spec.get("secondary"):
            continue
        for granularity in config["granularities"]:
            key = spec["name"], granularity
            b = Basis.load(basis_path(config, *key, 0))
            bases[key] = b
    encoder = make_encoder(config["encoder"])
    encoder.encode(texts[:8])
    torch.cuda.synchronize()
    reference, tick = {}, time.perf_counter()
    for key, basis in bases.items():
        print("Reference", key, flush=True)
        blocks = []
        for batch in chunks(texts, 8):
            encoded = encoder.encode(batch)
            blocks.append(basis.aggregate(encoded.pooled, encoded.tokens, key[1]))
        reference[key] = sparse.vstack(blocks)
    torch.cuda.synchronize()
    reference_seconds = time.perf_counter()-tick
    adapters = {key: TensorBasis(basis) for key, basis in bases.items()}
    # Isolate GPU transform differences from E5 batching and pooling differences.
    encoded = encoder.encode(texts[:8])
    transform_differences = {}
    for key, basis in bases.items():
        a = basis.aggregate(encoded.pooled, encoded.tokens, key[1])
        b = adapters[key].aggregate(encoded.pooled, encoded.tokens, key[1])
        transform_differences["__".join(key)] = differences(a, b)
    outcomes = []
    for batch_size in [8, 16, 32, 48]:
        print("Shared batch size", batch_size, flush=True)
        torch.cuda.reset_peak_memory_stats()
        results = {key: [] for key in bases}
        tick = time.perf_counter()
        try:
            for batch in chunks(texts, batch_size):
                encoded = encoder.encode(batch, device_output=True)
                for key, adapter in adapters.items():
                    results[key].append(adapter.aggregate(encoded.pooled, encoded.tokens, key[1]))
            torch.cuda.synchronize()
            seconds = time.perf_counter()-tick
            delta = {"__".join(key): differences(reference[key], sparse.vstack(results[key])) for key in bases}
            outcomes.append({"batch_size": batch_size, "seconds": seconds,
                             "speedup": reference_seconds/seconds, "differences": delta,
                             "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                             "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        except torch.cuda.OutOfMemoryError:
            outcomes.append({"batch_size": batch_size, "error": "CUDA OOM"})
            torch.cuda.empty_cache()
    # Gates are explicitly numerical, not a claim of bitwise CPU/CUDA equality.
    transform_passed = all(d["relative_l1"] <= 0.001 for d in transform_differences.values())
    eligible = [r for r in outcomes if "error" not in r and
                max(d["relative_l1"] for d in r["differences"].values()) <= 0.005 and
                r["peak_reserved_bytes"] < 8 * 1024**3]
    chosen = min(eligible, key=lambda r: r["seconds"])["batch_size"] if eligible and transform_passed else None
    report = {"documents": len(texts), "representations": len(bases), "reference_seconds": reference_seconds,
              "samples": sample_ids, "encoder": config["encoder"], "provenance": provenance(),
              "long_documents": args.long_documents,
              "transform_differences": transform_differences, "transform_gate_passed": transform_passed,
              "candidates": outcomes, "selected_batch_size": chosen,
              "gates": {"transform_relative_l1": 0.001, "end_to_end_relative_l1": 0.005,
                        "max_reserved_GiB": 8},
              "note": "Single small mixed-language sample; excludes disk writes and is not a full-corpus ETA."}
    atomic_json(args.output, report)
    print(report, flush=True)
    if chosen is None:
        raise SystemExit("No batch configuration passed numerical/memory gates")


if __name__ == "__main__":
    main()
