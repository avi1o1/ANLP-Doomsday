"""Explicit CLI stages; no expensive work runs merely by importing the package."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src import __version__
from src.artifacts import StopFlag, StopRequested, atomic_json, provenance
from src.config import load_config, public_config

EXPERIMENT_FAMILIES = ["primary", "retrieval", "setting-a", "sparsity", "parameters", "robustness",
                       "controls", "shared_sae", "external", "attention", "setting-b", "routing",
                       "setting-c", "smoke", "all"]


def parser():
    command = argparse.ArgumentParser(prog="csx", description="Collection-statistics research workflow")
    command.add_argument("--version", action="version", version=__version__)
    command.add_argument("--config", default=os.environ.get("CONFIG", "configs/research.yaml"))
    command.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT"))
    command.add_argument("--device", choices=["cpu", "cuda"])
    sub = command.add_subparsers(dest="stage", required=True)
    for stage in ("encode-shared", "join-encoded"):
        p = sub.add_parser(stage)
        p.add_argument("--dataset", required=True)
        p.add_argument("--side", choices=["documents", "queries"], required=True)
        p.add_argument("--representations", type=json.loads, required=True)
        p.add_argument("--shard-id", type=int, default=0)
        p.add_argument("--num-shards", type=int, default=1)
    for stage in ("prepare", "encode-sample", "fit-bases", "encode-corpus", "index", "evaluate", "baseline"):
        p = sub.add_parser(stage)
        p.add_argument("--dataset")
        if stage in {"fit-bases", "encode-corpus", "index", "evaluate"}:
            p.add_argument("--basis")
            p.add_argument("--granularity", choices=["pooled", "token"])
            p.add_argument("--seed", type=int, default=0)
        if stage in {"index", "evaluate"}:
            p.add_argument("--budget")
        if stage == "encode-corpus":
            p.add_argument("--shard-id", type=int)
            p.add_argument("--num-shards", type=int, default=1)
        if stage == "evaluate":
            p.add_argument("--k1", type=float, default=1.2)
            p.add_argument("--b", type=float, default=0.75)
            p.add_argument("--control", choices=["shuffled_idf"])
        if stage == "baseline":
            p.add_argument("--method", choices=["bm25", "dense", "splade"], required=True)
    for stage in ("diagnose", "fit-predictor", "run-retrieval", "validate-datasets", "controls", "external-status"):
        sub.add_parser(stage)
    p = sub.add_parser("report")
    p.add_argument("--report-dir")
    p = sub.add_parser("manifest")
    p.add_argument("--family", choices=EXPERIMENT_FAMILIES, default="primary")
    p.add_argument("--directory", default="results/manifests/primary")
    p = sub.add_parser("launch", help="Run selected experiment stages within the current allocation")
    p.add_argument("--family", choices=EXPERIMENT_FAMILIES, default="primary")
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cpus", type=int, default=36)
    p.add_argument("--stages", nargs="+", help="Only run these named manifest stages")
    p = sub.add_parser("task")
    p.add_argument("--manifest", required=True)
    p.add_argument("--task-index", type=int, required=True)
    p.add_argument("--task-config", required=True)
    p = sub.add_parser("smoke")
    p.add_argument("--model", action="store_true", help="Include configured encoder throughput and peak GPU memory")
    p = sub.add_parser("freeze-models")
    p.add_argument("--destination", required=True)
    p = sub.add_parser("prepare-benchmarks")
    p.add_argument("--benchmark", choices=["ruler", "wikitext", "mmlu", "gsm8k"], required=True)
    p = sub.add_parser("transfer")
    p.add_argument("--setting", choices=["attention", "routing"], required=True)
    p.add_argument("--task")
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--token-budget", type=int, default=512)
    p.add_argument("--sink", action="store_true")
    p.add_argument("--benchmark", choices=["wikitext", "mmlu", "gsm8k"], default="wikitext")
    p.add_argument("--subject")
    p = sub.add_parser("evaluate-external")
    p.add_argument("--dataset", required=True)
    p.add_argument("--export-manifest", required=True)
    return command


def validate_datasets(config):
    import ir_datasets

    records = []
    for name, spec in config["datasets"].items():
        if not spec.get("enabled", True):
            records.append({"name": name, "status": "unavailable", "reason": spec.get("unavailable_reason")})
            continue
        if spec["source"] == "ir_datasets":
            dataset = ir_datasets.load(spec["id"])
            if not all(getattr(dataset, f"has_{part}")() for part in ("docs", "queries", "qrels")):
                raise ValueError(f"Dataset {name} lacks required interfaces")
            records.append({"name": name, "id": spec["id"], "status": "registered", "corpus_id": spec["corpus_id"]})
        elif spec["source"] == "hf_json":
            records.append({"name": name, "status": "released_jsonl_adapter", "id": spec["id"]})
        else:
            records.append({"name": name, "status": "local", "exists": Path(spec["path"]).is_dir()})
    return records


def smoke(config, include_model):
    import numpy as np
    import torch
    import torch.nn.functional as F

    result = {"provenance": provenance(), "cuda_available": torch.cuda.is_available(), "devices": []}
    for i in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(i)
        q = torch.randn(1, 4, 128, 64, device=f"cuda:{i}", dtype=torch.float16)
        actual = F.scaled_dot_product_attention(q, q, q, is_causal=True)
        if not actual.isfinite().all():
            raise ValueError("FP16 SDPA smoke produced nonfinite outputs")
        result["devices"].append({"name": properties.name, "bytes": properties.total_memory,
                                  "capability": [properties.major, properties.minor], "fp16_sdpa": "passed"})
    if include_model:
        from src.retrieval.encoding import make_encoder

        encoder = make_encoder(config["encoder"])
        texts = ["A fixed engineering smoke passage about information retrieval.",
                 "यह सूचना पुनर्प्राप्ति का परीक्षण है।", "এটি তথ্য অনুসন্ধানের পরীক্ষা।", "ఇది సమాచార శోధన పరీక్ష."] * 4
        encoder.encode(texts[:4])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        encoded = encoder.encode(texts)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - tick
        result["encoder"] = {"resolved_revision": encoder.revision, "documents_per_second": len(texts)/elapsed,
                             "pooled_bytes_per_document": encoded.pooled.nbytes/len(texts),
                             "sample_content_tokens": encoded.token_counts,
                             "truncated_fraction": float(np.mean(encoded.truncated)),
                             "peak_cuda_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                             "caveat": "Short engineering inputs; size production jobs using representative corpus shards"}
    atomic_json(Path(config["output_root"]) / "smoke.json", result)
    return result


def freeze_models(config, destination):
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    frozen = deepcopy(public_config(config))
    for role, spec in [("encoder", frozen["encoder"]),
                       ("external_splade", frozen.get("external", {}).get("splade", {})),
                       ("attention", frozen.get("attention", {})), ("routing", frozen.get("routing", {}))]:
        if spec.get("model"):
            try:
                spec["revision"] = api.model_info(spec["model"], revision=spec.get("revision", "main")).sha
                hf_hub_download(spec["model"], "config.json", revision=spec["revision"])
            except Exception as error:
                status = getattr(getattr(error, "response", None), "status_code", None)
                if role == "external_splade" and status in {401, 403}:
                    # An optional comparator must not block the controlled E5 run.
                    # Retain its resolved revision and explicitly record missing access.
                    frozen["external"]["splade_access"] = {
                        "status": "requires_checkpoint_access",
                        "reason": f"Hugging Face checkpoint access denied (HTTP {status})",
                    }
                    continue
                if not spec.get("fallback_model") or status not in {401, 403}:
                    raise
                spec["requested_model"] = spec["model"]
                spec["model"] = spec["fallback_model"]
                spec["revision"] = api.model_info(spec["model"]).sha
        if spec.get("fallback_model"):
            spec["fallback_revision"] = api.model_info(spec["fallback_model"]).sha
    routing = frozen.get("routing", {})
    routing["dataset_revisions"] = {name: api.dataset_info(repository).sha for name, repository in
        [("wikitext", "Salesforce/wikitext"), ("mmlu", "cais/mmlu"), ("gsm8k", "openai/gsm8k")]}
    for spec in frozen["datasets"].values():
        if spec["source"] == "hf_json":
            spec["revision"] = api.dataset_info(spec["id"], revision=spec.get("revision", "main")).sha
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    Path(destination).write_text(yaml.safe_dump(frozen, sort_keys=False))
    return str(destination)


def main(argv=None):
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    args = parser().parse_args(argv)
    if args.stage == "task":
        from src.experiments import run_task
        return run_task(args.manifest, args.task_index, args.task_config)
    config = load_config(args.config)
    if args.output_root:
        config["output_root"] = args.output_root
    if args.device:
        config["encoder"]["device"] = args.device
    from src.retrieval import pipeline

    with StopFlag() as stop:
        if args.stage == "manifest":
            from src.experiments import write_manifest
            result = write_manifest(config, args.directory, args.family)
        elif args.stage == "launch":
            from src.launch import launch
            return launch(config, args.family, args.gpus, args.workers, args.cpus, stop, args.stages)
        elif args.stage == "run-retrieval":
            result = {"completed_rows": len(pipeline.run_retrieval(config, stop))}
        elif args.stage == "encode-sample":
            result = pipeline.encode_sample(config, stop)
        elif args.stage == "diagnose":
            result = {"rows": len(pipeline.diagnose(config))}
        elif args.stage == "fit-predictor":
            result = pipeline.fit_predictor(config)
        elif args.stage == "report":
            from src.reporting import report
            result = report(config, args.report_dir)
        elif args.stage == "validate-datasets":
            result = validate_datasets(config)
        elif args.stage == "smoke":
            result = smoke(config, args.model)
        elif args.stage == "freeze-models":
            result = freeze_models(config, args.destination)
        elif args.stage in {"encode-shared", "join-encoded"}:
            from src.retrieval.shared_encoding import encode_shared, join_shared
            if args.stage == "encode-shared":
                result = encode_shared(config, args.dataset, args.side, args.representations, stop,
                                       args.shard_id, args.num_shards)
            else:
                result = join_shared(config, args.dataset, args.side, args.representations, stop)
        elif args.stage == "controls":
            from src.retrieval.controls import run_controls
            result = run_controls(config)
        elif args.stage == "external-status":
            from src.retrieval.external import reproduction_status
            result = reproduction_status(config)
        elif args.stage == "evaluate-external":
            from src.retrieval.external import evaluate_external
            result = evaluate_external(config, args.dataset, args.export_manifest, stop)
        elif args.stage == "prepare-benchmarks":
            from src.attention.benchmarks import prepare_ruler
            from src.routing.benchmarks import prepare_routing
            result = prepare_ruler(config, stop) if args.benchmark == "ruler" else prepare_routing(config, args.benchmark, stop)
        elif args.stage == "transfer":
            from src.attention.pipeline import attention_transfer
            from src.routing.pipeline import routing_transfer
            if args.setting == "attention":
                tasks = [args.task] if args.task else config["attention"]["tasks"]
                result = [attention_transfer(config, task, args.context, args.token_budget, stop, args.sink) for task in tasks]
            else:
                result = routing_transfer(config, args.benchmark, stop, args.subject)
        else:
            datasets = [args.dataset] if args.dataset else [n for n, s in config["datasets"].items()
                if s.get("enabled", True) and (args.stage == "prepare" or s["role"] == "evaluate")]
            result = []
            if args.stage in {"prepare", "baseline"}:
                for dataset in datasets:
                    if args.stage == "prepare":
                        result.append(pipeline.prepare(config, dataset, stop))
                    else:
                        from src.retrieval.baselines import baseline
                        result.append(baseline(config, dataset, args.method, stop))
            else:
                names = [args.basis] if args.basis else [b["name"] for b in config["bases"] if not b.get("secondary")]
                granularities = [args.granularity] if args.granularity else config.get("granularities", ["pooled", "token"])
                for name in names:
                    for granularity in granularities:
                        if args.stage == "fit-bases":
                            result.append(pipeline.fit_basis(config, name, granularity, args.seed, stop))
                            continue
                        for dataset in datasets:
                            if args.stage == "encode-corpus":
                                result.append(pipeline.encode_corpus(config, dataset, name, granularity, args.seed,
                                                                     stop, args.shard_id, args.num_shards))
                                continue
                            budgets = [args.budget] if args.budget else [b["name"] for b in config["budgets"]
                                if not b.get("collections") or dataset in b["collections"]]
                            for budget in budgets:
                                if args.stage == "index":
                                    result.append(pipeline.build_index(config, dataset, name, granularity, args.seed, budget, stop))
                                else:
                                    result.append(pipeline.evaluate(config, dataset, name, granularity, args.seed, budget,
                                                                    stop, args.k1, args.b, args.control))
        print(json.dumps(result, default=str, indent=2))
    return 0


def entrypoint():
    try:
        raise SystemExit(main())
    except StopRequested as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(99) from error


if __name__ == "__main__":
    entrypoint()
