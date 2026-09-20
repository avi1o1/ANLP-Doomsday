"""Hypothesis-driven experiment expansion and reproducible task manifests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

from src.artifacts import atomic_json, jsonl, write_jsonl
from src.config import digest, public_config
from src.retrieval.pipeline import fit_dataset

SETTING_FAMILIES = {
    "setting-a": "primary",
    "setting-b": "attention",
    "setting-c": "routing",
    "retrieval": "primary",
}


def canonical_family(family: str) -> str:
    """Resolve user-facing setting names to the existing experiment families."""
    return SETTING_FAMILIES.get(family, family)


def experiment_rows(config, family="primary"):
    family = canonical_family(family)
    if family == "all":
        combined = {}
        for value in ("primary", "sparsity", "robustness", "parameters", "controls", "shared_sae"):
            for row in experiment_rows(config, value):
                combined[row["row_id"]] = row
        return list(combined.values())
    spec = config.get("experiments", {})
    rows = []
    for collection, dataset in config["datasets"].items():
        if dataset["role"] != "evaluate" or not dataset.get("enabled", True):
            continue
        allowed = {"robustness": "robustness_collections", "parameters": "parameter_collections", "controls": "control_collections"}
        if family in allowed and collection not in spec.get(allowed[family], []):
            continue
        for basis in config["bases"]:
            if bool(basis.get("secondary")) != (family == "shared_sae"):
                continue
            if family == "robustness" and basis["name"] not in spec.get("stochastic_bases", []):
                continue
            for granularity in config.get("granularities", ["pooled", "token"]):
                seeds = spec.get("robustness_seeds", [1, 2]) if family == "robustness" else [0]
                for seed in seeds:
                    for budget in config["budgets"]:
                        if not budget.get("primary") and family != "sparsity":
                            continue
                        if budget.get("collections") and collection not in budget["collections"]:
                            continue
                        k_values = spec.get("k1", [0.6, 1.2, 2.4]) if family == "parameters" else [1.2]
                        b_values = spec.get("b", [0, 0.5, 0.75]) if family == "parameters" else [0.75]
                        for k1 in k_values:
                            for b in b_values:
                                row = {"collection": collection, "corpus_id": dataset["corpus_id"],
                                       "basis": basis["name"], "granularity": granularity, "seed": seed,
                                       "budget": budget["name"], "k1": k1, "b": b,
                                       "control": "shuffled_idf" if family == "controls" else None}
                                row["row_id"] = digest(row)
                                rows.append(row)
    return rows


def build_tasks(config, rows, include_baselines=True):
    stages = {name: [] for name in ["prepare", "baselines-cpu", "baselines-gpu", "encode-sample", "fit-cpu",
                                  "fit-gpu", "fit-shared", "encode-corpus", "index", "evaluate", "analysis"]}
    datasets = sorted({r["collection"] for r in rows} | {fit_dataset(config)})
    stages["prepare"] = [["prepare", "--dataset", d] for d in datasets]
    stages["encode-sample"] = [["encode-sample"]]
    fit = sorted({(r["basis"], r["granularity"], r["seed"]) for r in rows})
    for basis in config["bases"]:
        if basis.get("dictionary_basis") and any(r["basis"] == basis["name"] for r in rows):
            fit.append((basis["dictionary_basis"], basis["dictionary_granularity"], 0))
    fit = sorted(set(fit))
    kinds = {b["name"]: b["kind"] for b in config["bases"]}
    for name, granularity, seed in fit:
        stage = "fit-gpu" if kinds[name] == "sae" else "fit-cpu"
        if next(b for b in config["bases"] if b["name"] == name).get("dictionary_basis"):
            stage = "fit-shared"
        stages[stage].append(["fit-bases", "--basis", name, "--granularity", granularity, "--seed", str(seed)])
    encoded = sorted({(r["collection"], r["basis"], r["granularity"], r["seed"]) for r in rows})
    for dataset, name, granularity, seed in encoded:
        stages["encode-corpus"].append(["encode-corpus", "--dataset", dataset, "--basis", name,
                                       "--granularity", granularity, "--seed", str(seed)])
    indexes = sorted({(r["collection"], r["basis"], r["granularity"], r["seed"], r["budget"]) for r in rows})
    for dataset, name, granularity, seed, budget in indexes:
        stages["index"].append(["index", "--dataset", dataset, "--basis", name, "--granularity", granularity,
                                "--seed", str(seed), "--budget", budget])
    for row in rows:
        command = ["evaluate", "--dataset", row["collection"], "--basis", row["basis"],
                   "--granularity", row["granularity"], "--seed", str(row["seed"]), "--budget", row["budget"],
                   "--k1", str(row["k1"]), "--b", str(row["b"])]
        if row["control"]:
            command.extend(["--control", row["control"]])
        stages["evaluate"].append(command)
    if include_baselines:
        for dataset in sorted({r["collection"] for r in rows}):
            stages["baselines-cpu"].append(["baseline", "--dataset", dataset, "--method", "bm25"])
            stages["baselines-gpu"].append(["baseline", "--dataset", dataset, "--method", "dense"])
    stages["analysis"] = [["diagnose"]]
    # Predictor is a separate final stage, with a clear failure if outcomes have no spread.
    stages["fit-predictor"] = [["fit-predictor"]]
    stages["report"] = [["report"]]
    return {key: value for key, value in stages.items() if value}


def write_manifest(config, directory, family="primary"):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    requested_family = family
    family = canonical_family(family)
    special = family in {"smoke", "attention", "routing", "external"}
    rows = [] if special else experiment_rows(config, family)
    if family == "smoke":
        stages = {"smoke": [["smoke", "--model"]]}
    elif family == "attention":
        stages = {"prepare-benchmarks": [["prepare-benchmarks", "--benchmark", "ruler"]],
                  "attention": [["transfer", "--setting", "attention", "--task", task,
                                 "--context", str(context), "--token-budget", str(budget)]
                    for task in config["attention"]["tasks"] for context in config["attention"]["contexts"]
                    for budget in config["attention"]["budgets"]], "report": [["report"]]}
    elif family == "routing":
        stages = {"prepare-benchmarks": [["prepare-benchmarks", "--benchmark", b] for b in ("wikitext", "mmlu")],
                  "routing": [["transfer", "--setting", "routing", "--benchmark", b] for b in ("wikitext", "mmlu")],
                  "report": [["report"]]}
    elif family == "external":
        datasets = [n for n, d in config["datasets"].items() if d["role"] == "evaluate" and d.get("enabled", True)]
        stages = {"prepare": [["prepare", "--dataset", n] for n in datasets],
                  "baselines-gpu": [["baseline", "--dataset", n, "--method", "splade"] for n in datasets],
                  "external-status": [["external-status"]], "report": [["report"]]}
    else:
        stages = build_tasks(config, rows)
    resolved = public_config(config)
    (directory / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    write_jsonl(directory / "rows.jsonl", rows)
    stage_records = []
    for name, tasks in stages.items():
        path = directory / f"{name}.jsonl"
        write_jsonl(path, ({"argv": command} for command in tasks))
        gpu = 1 if name in {"smoke", "attention", "encode-sample", "fit-gpu", "encode-corpus", "baselines-gpu"} else 0
        if name == "routing":
            gpu = 2
        stage_records.append({"name": name, "tasks": str(path), "count": len(tasks), "gpus": gpu,
                              "dependency": "afterany" if name == "report" else "afterok"})
    manifest = {"schema_version": 1, "family": family, "requested_family": requested_family,
                "config_hash": digest(resolved),
                "config": str(directory / "config.yaml"), "rows": len(rows) if not special else None,
                "scoring_configurations": 8 * len(rows) if not special else None,
                "planned_attention_rows": len(stages["attention"]) if family == "attention" else None,
                "routing_rows_resolved_from_prepared_subjects": family == "routing",
                "corpus_groups": sorted({r["corpus_id"] for r in rows}), "stages": stage_records,
                "unavailable": {n: d.get("unavailable_reason") for n, d in config["datasets"].items() if not d.get("enabled", True)}}
    atomic_json(directory / "manifest.json", manifest)
    return manifest



def run_task(manifest_path, index, config_path):
    tasks = list(jsonl(manifest_path))
    if not 0 <= index < len(tasks):
        raise ValueError("Task index outside manifest")
    # exec replaces the launcher, so Slurm signals reach checkpoint-aware Python directly.
    environment = os.environ.copy()
    environment.pop("OUTPUT_ROOT", None)
    os.execve(sys.executable, [sys.executable, "-m", "src", "--config", config_path] + tasks[index]["argv"], environment)
