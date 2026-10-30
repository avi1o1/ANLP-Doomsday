"""The registered Setting B analysis (`results/registration-2026-10-20.md`).

Section 1: the frozen diagnostic predictor against each row's margin, with its pass marks,
the retrieval majority sign and the better constant sign on the transfer rows.
Section 2: the label-free rule G~ > 0, with exact attention mass as the reference
(primary) and token F1 to the full-attention output (secondary).
Section 7: Holm-corrected 111 against 000, Shapley shares, and the baselines.

Sign accuracy counts only rows whose target is non-zero, as `src.analysis.metrics` does;
rows with a zero target are counted separately. A zero estimate counts as wrong.

    python -m scripts.transfer_study ROWS MASS_DIRECTORY PREDICTOR RETRIEVAL_ROWS OUTPUT
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from scripts.shapley_attribution import is_primary, shapley
from src.analysis import SWITCH_KEYS, FrozenPredictor, holm, metrics
from src.artifacts import atomic_json, file_hash

SWITCHES = ["idf", "saturation", "length"]


def gain(quality):
    return quality["111"] - quality["010"]


def sign_accuracy(y, estimate):
    y, estimate = np.asarray(y, float), np.asarray(estimate, float)
    nonzero = y != 0
    return float(np.mean(np.sign(estimate[nonzero]) == np.sign(y[nonzero]))), int((~nonzero).sum())


def constants(y):
    y = np.asarray(y, float)[np.asarray(y, float) != 0]
    positive = float(np.mean(y > 0))
    return {"positive": positive, "better_constant": max(positive, 1 - positive)}


def task_bootstrap(rows, values, samples=4000, seed=0):
    """95% interval of the mean of `values`, resampling whole tasks (both budgets together)."""
    tasks = sorted({r["task"] for r in rows})
    members = [np.flatnonzero([r["task"] == t for r in rows]) for t in tasks]
    rng = np.random.default_rng(seed)
    draws = [np.mean(np.concatenate([values[members[i]] for i in rng.integers(len(tasks), size=len(tasks))]))
             for _ in range(samples)]
    return [float(v) for v in np.percentile(draws, [2.5, 97.5])]


def decision(rows, estimate):
    on = np.array([r["quality"]["111"] for r in rows])
    off = np.array([r["quality"]["010"] for r in rows])
    chosen = np.where(np.asarray(estimate) > 0, on, off)
    oracle = np.maximum(on, off)
    headroom = oracle.mean() - on.mean()
    return {"always_on": float(on.mean()), "always_off": float(off.mean()), "rule": float(chosen.mean()),
            "oracle": float(oracle.mean()),
            "headroom_recovered": float((chosen.mean() - on.mean()) / headroom) if headroom > 0 else None,
            "rule_minus_always_on_ci95": task_bootstrap(rows, chosen - on),
            "rows_choosing_on": int((np.asarray(estimate) > 0).sum())}


def label_free(rows, pseudo, retrieval_positive):
    # A row whose generations end at the first token has no decode step to probe and no reference.
    undefined = sum(p is None for p in pseudo)
    rows, pseudo = zip(*[(r, p) for r, p in zip(rows, pseudo) if p is not None])
    y = [gain(r["quality"]) for r in rows]
    estimate = [gain(p) for p in pseudo]
    accuracy, zero = sign_accuracy(y, estimate)
    base = constants(y)
    result = {"sign_accuracy": accuracy, "zero_target_rows": zero, "undefined_reference_rows": undefined,
              "constant_retrieval_majority": {"prediction": "G > 0", "retrieval_positive_share": retrieval_positive,
                                              "accuracy": base["positive"]},
              "better_constant_sign": base["better_constant"],
              "spearman": float(spearmanr(estimate, y).statistic), "decision": decision(rows, estimate),
              "rows": [{"task": r["task"], "budget": r["budget"], "G": g, "G_estimate": e}
                       for r, g, e in zip(rows, y, estimate)]}
    result["passes"] = bool(accuracy > base["better_constant"]
                            and result["decision"]["rule"] >= result["decision"]["always_on"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rows", type=Path)
    parser.add_argument("mass", type=Path)
    parser.add_argument("predictor", type=Path)
    parser.add_argument("retrieval_rows", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = sorted(json.loads(args.rows.read_text()), key=lambda r: (r["task"], r["budget"]))
    rows = [r for r in rows if not r["sink"]]
    predictor = FrozenPredictor.load(args.predictor)
    training = set(predictor.metadata["training_row_ids"])
    retrieval = [r for r in json.loads(args.retrieval_rows.read_text()) if not r.get("control") and is_primary(r)]
    m_positive = float(np.mean([r["margin"] > 0 for r in retrieval if r["row_id"] in training]))
    g_positive = float(np.mean([gain(r["quality"]) > 0 for r in retrieval]))
    report = {"rows": len(rows), "predictor_sha256": file_hash(args.predictor),
              "retrieval_majority": {"m_positive": m_positive, "G_positive": g_positive}}

    # Section 1.
    defined = [r for r in rows if r["margin"] is not None]
    y = np.array([r["margin"] for r in defined])
    prediction = predictor.predict(defined)
    summary = metrics(y, prediction)
    threshold = predictor.metadata["transfer_mae_multiplier"] * predictor.metadata["retrieval_oof_rmse"]
    summary.update({
        "undefined_rows": len(rows) - len(defined), "mae_threshold": threshold,
        "passes_point_thresholds": bool(summary["sign_accuracy"] is not None and summary["sign_accuracy"] >= 0.70
                                        and summary["mae"] <= threshold),
        "constant_retrieval_majority": {"prediction": "m > 0", "accuracy": constants(y)["positive"]},
        "better_constant_sign": constants(y)["better_constant"],
        "predicted_positive": int((prediction > 0).sum()),
        "rows": [{"task": r["task"], "budget": r["budget"], "margin": r["margin"], "prediction": float(p)}
                 for r, p in zip(defined, prediction)]})
    report["pre_registered_predictor"] = summary

    # Section 2.
    mass = {}
    for r in rows:
        path = args.mass / f"{r['task']}__c{r['context']}__b{r['budget']}__sink0.json"
        mass[id(r)] = json.loads(path.read_text())
    report["mass_reference_checks"] = {
        "examples_probed": sum(mass[id(r)]["examples_probed"] for r in rows),
        "full_text_identical": sum(mass[id(r)]["full_text_identical"] for r in rows),
        "examples": sum(mass[id(r)]["examples"] for r in rows)}
    report["label_free_attention_mass"] = label_free(rows, [mass[id(r)]["pseudo_quality"] for r in rows], g_positive)
    f1 = [{k: float(np.mean([e["f1_to_full"][k] for e in r["per_example"]])) for k in SWITCH_KEYS} for r in rows]
    report["label_free_output_agreement"] = label_free(rows, f1, g_positive)

    # Section 7.
    adjusted = holm([r["p_value"] for r in rows])
    report["holm_111_vs_000"] = {
        "significant": int(sum(p < 0.05 for p in adjusted)),
        "significant_positive": int(sum(p < 0.05 and r["raw_delta"] > 0 for p, r in zip(adjusted, rows))),
        "significant_negative": int(sum(p < 0.05 and r["raw_delta"] < 0 for p, r in zip(adjusted, rows))),
        "p_value_floor": rows[0]["p_value_floor"]}
    values = np.array([[shapley(r["quality"])[s] for s in SWITCHES] for r in rows])
    report["shapley"] = {"mean": dict(zip(SWITCHES, values.mean(0).tolist())),
                         "mean_absolute": dict(zip(SWITCHES, np.abs(values).mean(0).tolist()))}
    modes = ["full", "quest", "h2o"] + SWITCH_KEYS
    means = {m: float(np.mean([np.mean([e["scores"][m] for e in r["per_example"]]) for r in rows])) for m in modes}
    report["mean_accuracy"] = {"all": means, **{f"budget_{b}": {
        m: float(np.mean([np.mean([e["scores"][m] for e in r["per_example"]]) for r in rows if r["budget"] == b]))
        for m in modes} for b in sorted({r["budget"] for r in rows})}}
    report["quest_equals_000"] = all(e["scores"]["quest"] == e["scores"]["000"] for r in rows for e in r["per_example"])
    report["per_row"] = [{"task": r["task"], "budget": r["budget"], "quality": r["quality"],
                          "full": float(np.mean([e["scores"]["full"] for e in r["per_example"]])),
                          "h2o": float(np.mean([e["scores"]["h2o"] for e in r["per_example"]])),
                          "raw_delta": r["raw_delta"], "margin": r["margin"], "holm_p": p}
                         for r, p in zip(rows, adjusted)]
    atomic_json(args.output, report)
    s, a, o = report["pre_registered_predictor"], report["label_free_attention_mass"], report["label_free_output_agreement"]
    print(f"predictor: sign {s['sign_accuracy']:.3f} (constant + {s['constant_retrieval_majority']['accuracy']:.3f}, "
          f"better constant {s['better_constant_sign']:.3f}) mae {s['mae']:.2f}/{threshold:.2f} pass={s['passes_point_thresholds']}")
    for name, x in (("attention mass", a), ("output agreement", o)):
        d = x["decision"]
        print(f"{name}: sign {x['sign_accuracy']:.3f} vs better constant {x['better_constant_sign']:.3f}, "
              f"spearman {x['spearman']:.2f}, rule {d['rule']:.4f} on {d['always_on']:.4f} off {d['always_off']:.4f} "
              f"oracle {d['oracle']:.4f}, pass={x['passes']}")


if __name__ == "__main__":
    main()
