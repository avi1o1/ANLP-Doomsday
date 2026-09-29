"""Exploratory: how well label-free index diagnostics predict each switch's Shapley value.

Primary rows are the training data. Two kinds of evidence are reported:

- grouped cross-validation, holding out whole basis families and whole corpora, with
  ridge regularisation chosen by an inner leave-one-group-out loop;
- out-of-sample prediction, fitting on the primary rows only and predicting rows with
  other document/query budgets and other basis seeds, whose indexes and diagnostics the
  fit never saw.

These analyses were chosen after looking at the data and are exploratory. The
confirmatory tests are in scripts/test_hypotheses.py.

Usage: python -m scripts.saturation_predictor ROWS_JSON DIAGNOSTICS_JSON OUT_JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from scripts.shapley_attribution import is_primary, shapley

FEATURE_SETS = {
    "pre-registered": ["frequency_slope", "idf_variance", "frequent_activation_share"],
    "value_cv": ["value_cv"],
    "length_log_sd": ["length_log_sd"],
    "three mechanism": ["value_cv", "peak_ratio", "length_log_sd"],
}
ALPHAS = (0.1, 1.0, 10.0, 100.0)


def features(row, diagnostics):
    d = diagnostics.get((row["representation_id"], row["seed"], row["budget"], row["collection"]))
    if d is None:
        return None
    f = dict(row["diagnostics"])
    f.update({k: v for k, v in d.items() if k not in ("representation", "seed", "budget", "collection")})
    return f


def fit(x, y, alpha):
    scaler = StandardScaler().fit(x)
    return scaler, Ridge(alpha=alpha).fit(scaler.transform(x), y)


def grouped_cv(x, y, groups):
    """Held-out R2 and Spearman over whole groups, alpha chosen by inner leave-one-group-out."""
    prediction = np.zeros(len(y))
    for g in sorted(set(groups)):
        train = np.array([v != g for v in groups])
        inner = sorted(set(np.array(groups)[train]))
        scores = []
        for alpha in ALPHAS:
            errors = []
            for h in inner:
                a = train & np.array([v != h for v in groups])
                b = train & np.array([v == h for v in groups])
                scaler, model = fit(x[a], y[a], alpha)
                errors.append(np.mean((model.predict(scaler.transform(x[b])) - y[b]) ** 2))
            scores.append((np.mean(errors), alpha))
        scaler, model = fit(x[train], y[train], min(scores)[1])
        prediction[~train] = model.predict(scaler.transform(x[~train]))
    r2 = 1 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2)
    return {"r2": round(float(r2), 4), "spearman": round(float(spearmanr(y, prediction).statistic), 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rows", type=Path)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.rows.read_text())
    diagnostics = {(d["representation"], d["seed"], d["budget"], d["collection"]): d
                   for d in json.loads(args.diagnostics.read_text())}
    usable = [r for r in rows if r["k1"] == 1.2 and r["b"] == 0.75 and not r.get("control")
              and features(r, diagnostics) is not None]
    primary = [r for r in usable if is_primary(r)]
    held = {"other budgets": [r for r in usable if r["seed"] == 0 and r["budget"] != "q32_d128"],
            "other seeds": [r for r in usable if r["seed"] != 0]}
    report = {"training_rows": len(primary), "held_rows": {k: len(v) for k, v in held.items()}, "targets": {}}
    for target in ("saturation", "length", "idf"):
        y = np.array([shapley(r["quality"])[target] for r in primary])
        out = {}
        for name, names in FEATURE_SETS.items():
            x = np.array([[features(r, diagnostics)[n] for n in names] for r in primary])
            entry = {"family": grouped_cv(x, y, [r["family"] for r in primary]),
                     "corpus": grouped_cv(x, y, [r["corpus_id"] for r in primary])}
            scaler, model = fit(x, y, 10.0)
            for label, rs in held.items():
                truth = np.array([shapley(r["quality"])[target] for r in rs])
                guess = model.predict(scaler.transform(np.array([[features(r, diagnostics)[n] for n in names]
                                                                 for r in rs])))
                entry[label] = {"r2": round(float(1 - np.sum((truth - guess) ** 2)
                                                  / np.sum((truth - truth.mean()) ** 2)), 4),
                                "spearman": round(float(spearmanr(truth, guess).statistic), 4)}
            out[name] = entry
        report["targets"][target] = out
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for target, out in report["targets"].items():
        print(f"{target}")
        for name, e in out.items():
            print(f"   {name:16s} family R2 {e['family']['r2']:+.2f} rho {e['family']['spearman']:+.2f} | "
                  f"corpus R2 {e['corpus']['r2']:+.2f} rho {e['corpus']['spearman']:+.2f} | "
                  f"budgets R2 {e['other budgets']['r2']:+.2f} rho {e['other budgets']['spearman']:+.2f} | "
                  f"seeds R2 {e['other seeds']['r2']:+.2f} rho {e['other seeds']['spearman']:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
