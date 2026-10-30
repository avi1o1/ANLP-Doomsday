"""Export stored SPLADE-v3 features in the external-evaluator format.

The SPLADE baseline already wrote document shards (vectors.npz, ids.json) under
baselines/splade/<corpus>/. This script encodes the judged queries once and writes the
manifest that `csx evaluate-external` reads, so SPLADE-v3 is scored under all eight
IDF / saturation / length combinations, the conceptual replication of the Appendix C
reversal.

    python -m scripts.export_splade_features CONFIG OUTPUT_ROOT DATASET MODEL_FILE [--device cpu]
"""

import argparse
from pathlib import Path

from scipy import sparse

from src.artifacts import atomic_json, file_hash, jsonl, read_json
from src.config import load_config, root
from src.retrieval.data import corpus_key
from src.retrieval.encoding import SpladeEncoder
from src.retrieval.index import qrels_mapping
from src.retrieval.pipeline import chunks, prepared_paths, spec_for


def export(config, dataset, model_file, device):
    collection, _ = prepared_paths(config, dataset)
    shards = [Path(p) for p in read_json(root(config) / "baselines" / "splade" / corpus_key(spec_for(config, dataset)) / "shards.json")]
    out = root(config) / "external_export" / dataset
    out.mkdir(parents=True, exist_ok=True)
    judged = qrels_mapping(jsonl(collection / "qrels.jsonl"))
    queries = [q for q in jsonl(collection / "queries.jsonl") if q["query_id"] in judged]
    spec = dict(config["external"]["splade"], device=device, precision="float32" if device == "cpu" else "float16")
    encoder = SpladeEncoder(**spec)
    blocks = [encoder.encode([q["text"] for q in batch], "query")
              for batch in chunks(queries, config["encoder"].get("batch_size", 8))]
    sparse.save_npz(out / "queries.npz", sparse.vstack(blocks).tocsr())
    atomic_json(out / "queries.ids.json", [q["query_id"] for q in queries])
    manifest = {
        "model": f"{spec['model']}@{encoder.revision}",
        "checkpoint_sha256": file_hash(model_file),
        "preprocessing": {"prefixes": "none", "max_length": spec["max_length"],
                          "aggregation": "max over tokens of log(1+relu(MLM logit)), vocabulary 30522",
                          "documents": "float16 on GPU, from the SPLADE baseline shards",
                          "queries": f"{spec['precision']} on {device}"},
        "reproduction_claim": "conceptual_replication",
        "documents": [{"matrix": str(s / "vectors.npz"), "ids": str(s / "ids.json")} for s in shards],
        "queries": [{"matrix": "queries.npz", "ids": "queries.ids.json"}],
    }
    atomic_json(out / "export.json", manifest)
    return out / "export.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("output_root")
    parser.add_argument("dataset")
    parser.add_argument("model_file")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    config["output_root"] = args.output_root
    print(export(config, args.dataset, args.model_file, args.device))


if __name__ == "__main__":
    main()
