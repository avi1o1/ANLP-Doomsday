"""Regenerate tables, macro averages, figures and incomplete-work inventories."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.artifacts import atomic_json, read_json
from src.config import root
from src.experiments import experiment_rows
from src.retrieval.pipeline import collect_results


def report(config, output=None):
    directory = Path(output) if output else root(config) / "report"
    directory.mkdir(parents=True, exist_ok=True)
    rows = collect_results(config)
    columns = ["row_id", "collection", "corpus_id", "language", "basis", "granularity", "seed", "budget",
               "k1", "b", "control", "ndcg_off", "ndcg_on", "raw_delta", "margin", "holm_p", "fixture"]
    with (directory / "retrieval.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            record = {key: row.get(key) for key in columns}
            record.update(ndcg_off=row["quality"]["000"], ndcg_on=row["quality"]["111"])
            writer.writerow(record)
    indicative = defaultdict(list)
    for row in rows:
        if row["language"] in {"hi", "bn", "te"}:
            key = (row["representation_id"], row["seed"], row["budget"], row["k1"], row["b"], row["control"])
            indicative[key].append(row)
    macros = []
    for key, group in indicative.items():
        if {r["language"] for r in group} == {"hi", "bn", "te"} and len(group) == 3:
            macros.append({"representation": key[0], "seed": key[1], "budget": key[2], "k1": key[3], "b": key[4],
                           "control": key[5], "languages": ["hi", "bn", "te"],
                           "metrics": {s: {metric: float(np.mean([r["metrics"][s][metric] for r in group]))
                                for metric in ("ndcg@10", "mrr@10", "recall@100", "recall@1000")}
                                for s in group[0]["quality"]}})
    atomic_json(directory / "indic_macro.json", macros)
    with (directory / "resources.csv").open("w") as handle:
        names = ["collection", "basis", "granularity", "seed", "budget", "index_bytes", "document_nonzeros",
                 "query_nonzeros", "document_truncation", "query_truncation", "peak_rss_kib", "off_seconds", "on_seconds"]
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            encoding = row["encoding"]
            writer.writerow({**{k: row[k] for k in names[:5]}, "index_bytes": row["index"]["index_bytes"],
                "document_nonzeros": encoding["mean_retained_document_nonzeros"],
                "query_nonzeros": encoding["mean_retained_query_nonzeros"],
                "document_truncation": encoding["documents"]["truncation_fraction"],
                "query_truncation": encoding["queries"]["truncation_fraction"], "peak_rss_kib": row["peak_process_rss_kib"],
                "off_seconds": row["metrics"]["000"]["amortized_search_seconds"],
                "on_seconds": row["metrics"]["111"]["amortized_search_seconds"]})
    validation = root(config) / "predictor" / "validation.json"
    if validation.exists():
        names = ["model", "axis", "weighted", "n", "r2", "mae", "rmse", "sign_accuracy", "auc", "majority_sign_accuracy"]
        with (directory / "predictor.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=names)
            writer.writeheader()
            for record in read_json(validation):
                row = {**record, **record["summary"]}
                writer.writerow({k: row.get(k) for k in names})
    transfer_rows = []
    for setting in ("attention", "routing"):
        paths = sorted((root(config) / "transfer" / setting).glob("*/result.json"))
        for path in paths:
            if read_json(path.parent / "manifest.json")["status"] == "complete":
                transfer_rows.append(read_json(path))
        if paths and validation.exists():
            from src.transfer import transfer_report
            atomic_json(directory / f"{setting}_summary.json", transfer_report(config, setting))
    atomic_json(directory / "transfer_rows.json", transfer_rows)
    baseline_rows = [read_json(p) for p in sorted((root(config) / "baseline_evaluations").glob("*/*/result.json"))
                     if read_json(p.parent / "manifest.json")["status"] == "complete"]
    atomic_json(directory / "baselines.json", baseline_rows)
    statuses = []
    for path in sorted(root(config).rglob("manifest.json")):
        manifest = read_json(path)
        if "status" in manifest and manifest["status"] != "complete":
            statuses.append({"path": str(path), "status": manifest["status"], "error": manifest.get("error")})
    planned = experiment_rows(config, "all")
    completed = {(r["collection"], r["basis"], r["granularity"], r["seed"], r["budget"], r["k1"], r["b"],
                  None if r["control"] == "shared_dictionary" else r["control"]) for r in rows}
    missing = [r for r in planned if tuple(r[k] for k in ("collection", "basis", "granularity", "seed", "budget", "k1", "b", "control")) not in completed]
    inventory = {"completed_retrieval_rows": len(rows), "planned_retrieval_rows": len(planned),
                 "completed_transfer_rows": {setting: sum(r["setting"] == setting for r in transfer_rows)
                                             for setting in ("attention", "routing")},
                 "missing_rows": missing, "incomplete_artifacts": statuses,
                 "unavailable_datasets": {n: d.get("unavailable_reason") for n, d in config["datasets"].items() if not d.get("enabled", True)},
                 "fixture": config.get("fixture", False)}
    status = root(config) / "external" / "latent_terms_status.json"
    if status.exists():
        inventory["latent_terms"] = read_json(status)
    atomic_json(directory / "inventory.json", inventory)
    lines = ["# Experiment report", "", "Engineering fixture; no research conclusions." if config.get("fixture") else "Generated from completed artifacts only.",
             "", f"Completed retrieval rows: {len(rows)}. Planned rows in the current expanded manifest: {len(planned)}.",
             f"Incomplete or failed artifacts: {len(statuses)}. Unrun retrieval rows: {len(missing)}.", "",
             "| Collection | Basis | Granularity | Budget | Seed | nDCG off | nDCG on | Margin |",
             "|---|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        margin = "undefined" if row["margin"] is None else f"{row['margin']:.3f}"
        lines.append(f"| {row['collection']} | {row['basis']} | {row['granularity']} | {row['budget']} | {row['seed']} | "
                     f"{row['quality']['000']:.4f} | {row['quality']['111']:.4f} | {margin} |")
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    if rows:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pass
        else:
            primary = [r for r in rows if r["primary_budget"] and r["primary_parameters"] and r["seed"] == 0 and not r["control"]]
            fig, ax = plt.subplots(figsize=(7, 4))
            for language in sorted({r["language"] for r in primary}):
                group = [r for r in primary if r["language"] == language]
                ax.scatter([r["quality"]["000"] for r in group], [r["raw_delta"] for r in group], label=language, alpha=0.7)
            ax.axhline(0, color="black", linewidth=0.7)
            ax.set(xlabel="Uncorrected nDCG@10", ylabel="All-on minus all-off nDCG@10")
            if primary:
                ax.legend()
            fig.tight_layout()
            fig.savefig(directory / "retrieval_effects.pdf")
            plt.close(fig)
    return directory
