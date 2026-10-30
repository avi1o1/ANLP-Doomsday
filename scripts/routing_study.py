"""The registered Setting C analysis (`results/registration-2026-10-20.md`, Sections 1, 3 and 7).

Reads the routing transfer directory: each row's `result.json`, the per-example scores of
the native router and the frequency-bias baseline, and the example weights (WikiText token
counts). The frozen predictor is scored on the MMLU subject rows, against its pass marks,
the retrieval majority sign and the better constant sign. WikiText is one row and is
reported on its own. Holm correction runs over all routing rows.

    python -m scripts.routing_study ROUTING_DIRECTORY PREDICTOR RETRIEVAL_ROWS OUTPUT
"""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.shapley_attribution import is_primary, shapley
from scripts.transfer_study import SWITCHES, constants
from src.analysis import SWITCH_KEYS, FrozenPredictor, holm, metrics
from src.artifacts import atomic_json, file_hash, read_json

ROW_FIELDS = ["row_id", "benchmark", "subject", "model", "diagnostics", "quality", "quality_measure", "raw_delta",
              "raw_ci95", "spread", "margin", "margin_ci95", "p_value", "p_value_floor", "n_examples", "perplexity"]


def collect(row):
    result = read_json(row / "result.json")
    entry = {k: result.get(k) for k in ROW_FIELDS}
    weights = read_json(row / "example_weights.json")
    examples = sorted(p for p in (row / "examples").iterdir() if p.is_dir())
    baseline = {}
    for mode in ("native", "frequency_bias"):
        scores = [read_json(e / f"{mode}.json")["score"] for e in examples]
        baseline[mode] = float(np.average(scores, weights=weights)) if len(scores) == len(weights) else None
    entry["baselines"] = baseline
    entry["examples"] = len(examples)
    return entry


def subject_bootstrap(y, prediction, samples=4000, seed=0):
    rng = np.random.default_rng(seed)
    signs, maes = [], []
    for _ in range(samples):
        i = rng.integers(len(y), size=len(y))
        nonzero = y[i] != 0
        signs.append(np.mean(np.sign(y[i][nonzero]) == np.sign(prediction[i][nonzero])))
        maes.append(np.mean(np.abs(y[i] - prediction[i])))
    return ([float(v) for v in np.percentile(signs, [2.5, 97.5])],
            [float(v) for v in np.percentile(maes, [2.5, 97.5])])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("predictor", type=Path)
    parser.add_argument("retrieval_rows", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = []
    for row in sorted(p for p in args.directory.iterdir() if p.is_dir()):
        if (row / "manifest.json").exists() and read_json(row / "manifest.json")["status"] == "complete":
            rows.append(collect(row))
    predictor = FrozenPredictor.load(args.predictor)
    training = set(predictor.metadata["training_row_ids"])
    retrieval = [r for r in json.loads(args.retrieval_rows.read_text()) if not r.get("control") and is_primary(r)]
    report = {"rows": len(rows), "predictor_sha256": file_hash(args.predictor),
              "retrieval_m_positive": float(np.mean([r["margin"] > 0 for r in retrieval if r["row_id"] in training]))}
    mmlu = [r for r in rows if r["benchmark"] == "mmlu" and r["margin"] is not None]
    y = np.array([r["margin"] for r in mmlu])
    prediction = predictor.predict(mmlu) if mmlu else np.array([])
    summary = metrics(y, prediction) if mmlu else {"n": 0}
    if mmlu:
        threshold = predictor.metadata["transfer_mae_multiplier"] * predictor.metadata["retrieval_oof_rmse"]
        sign_ci, mae_ci = subject_bootstrap(y, prediction)
        summary.update({
            "subjects": len(mmlu), "undefined_subjects": sum(r["benchmark"] == "mmlu" and r["margin"] is None for r in rows),
            "mae_threshold": threshold, "sign_accuracy_ci95": sign_ci, "mae_ci95": mae_ci,
            "passes_point_thresholds": bool(summary["sign_accuracy"] is not None and summary["sign_accuracy"] >= 0.70
                                            and summary["mae"] <= threshold),
            "constant_retrieval_majority_accuracy": constants(y)["positive"],
            "better_constant_sign": constants(y)["better_constant"],
            "predicted_positive": int((prediction > 0).sum())})
    report["pre_registered_predictor_mmlu"] = summary
    wikitext = [r for r in rows if r["benchmark"] == "wikitext"]
    if wikitext:
        w = wikitext[0]
        report["wikitext"] = {"quality": w["quality"], "perplexity": w["perplexity"], "margin": w["margin"],
                              "raw_delta": w["raw_delta"], "raw_ci95": w["raw_ci95"], "p_value": w["p_value"],
                              "baselines": w["baselines"],
                              "prediction": float(predictor.predict([w])[0]) if w["margin"] is not None else None}
    adjusted = holm([r["p_value"] if r["p_value"] is not None else 1.0 for r in rows])
    report["holm_111_vs_000"] = {
        "significant_positive": int(sum(p < 0.05 and r["raw_delta"] > 0 for p, r in zip(adjusted, rows))),
        "significant_negative": int(sum(p < 0.05 and r["raw_delta"] < 0 for p, r in zip(adjusted, rows))),
        "rows": len(rows)}
    values = np.array([[shapley(r["quality"])[s] for s in SWITCHES] for r in mmlu]) if mmlu else np.zeros((0, 3))
    report["shapley_mmlu"] = dict(zip(SWITCHES, values.mean(0).tolist())) if mmlu else None
    report["mean_accuracy_mmlu"] = {k: float(np.mean([r["quality"][k] for r in mmlu])) for k in SWITCH_KEYS} | {
        m: float(np.mean([r["baselines"][m] for r in mmlu])) for m in ("native", "frequency_bias")} if mmlu else None
    report["per_row"] = [{k: r[k] for k in ("benchmark", "subject", "quality", "raw_delta", "margin", "p_value",
                                            "n_examples", "baselines")} | {"holm_p": p}
                         for r, p in zip(rows, adjusted)]
    report["row_data"] = rows
    atomic_json(args.output, report)
    s = report["pre_registered_predictor_mmlu"]
    if mmlu:
        print(f"MMLU predictor: sign {s['sign_accuracy']:.3f} (better constant {s['better_constant_sign']:.3f}, "
              f"constant + {s['constant_retrieval_majority_accuracy']:.3f}) mae {s['mae']:.2f}/{s['mae_threshold']:.2f} "
              f"pass={s['passes_point_thresholds']} subjects={s['subjects']}")
    print("holm", report["holm_111_vs_000"])


if __name__ == "__main__":
    main()
