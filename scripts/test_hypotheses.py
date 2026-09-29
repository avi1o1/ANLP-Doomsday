"""Run the confirmatory tests fixed in results/.../exploratory/HYPOTHESES.md (commit 9df6d58).

Everything is computed from the committed analysis rows and index diagnostics. The
tests, directions and thresholds are the committed ones; nothing here was tuned after
the results were seen. H2-H6 are Holm-corrected as one family (H5 is an exact identity
and carries no p-value).

Usage: python -m scripts.test_hypotheses ROWS_JSON DIAGNOSTICS_JSON OUT_JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from scripts.shapley_attribution import is_primary, shapley

PARAMETER_COLLECTIONS = {"scifact", "fiqa", "miracl_hi", "miracl_bn", "miracl_te"}


def holm(p):
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def within_corpus_spearman(x, y, groups):
    x, y = np.array(x, float), np.array(y, float)
    for g in set(groups):
        m = [i for i, v in enumerate(groups) if v == g]
        x[m] -= x[m].mean()
        y[m] -= y[m].mean()
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rows", type=Path)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.rows.read_text())
    diagnostics = {(d["representation"], d["seed"], d["budget"], d["collection"]): d
                   for d in json.loads(args.diagnostics.read_text())}
    grid = [r for r in rows if r["collection"] in PARAMETER_COLLECTIONS and r["seed"] == 0
            and r["budget"] == "q32_d128" and not r.get("control")]

    def value_cv(r):
        return diagnostics[(r["representation_id"], r["seed"], r["budget"], r["collection"])]["value_cv"]

    def cell(k1, b):
        return {(r["representation_id"], r["collection"]): r for r in grid if r["k1"] == k1 and r["b"] == b}

    def sv(r):
        return shapley(r["quality"])

    results = {}

    # H1: collection-statistics share at the default cell lies in 36-42%.
    default = list(cell(1.2, 0.75).values())
    values = np.array([[sv(r)[s] for s in ("idf", "saturation", "length")] for r in default])
    total = sum(r["quality"]["111"] - r["quality"]["000"] for r in default)
    share = float((values[:, 0].sum() + values[:, 2].sum()) / total)
    results["H1"] = {"rows": len(default), "collection_statistics_share": round(share, 4),
                     "interval": [0.36, 0.42], "passes": 0.36 <= share <= 0.42,
                     "note": "the default cell is the primary row for these collections, so this data overlaps "
                             "the exploratory analysis; a consistency check, not independent confirmation"}

    # H2: saturation larger at k1 = 0.6 than at k1 = 2.4, b = 0.75, paired, one-sided.
    low, high = cell(0.6, 0.75), cell(2.4, 0.75)
    keys = sorted(set(low) & set(high))
    a = np.array([sv(low[k])["saturation"] for k in keys])
    c = np.array([sv(high[k])["saturation"] for k in keys])
    p2 = float(wilcoxon(a, c, alternative="greater").pvalue)
    results["H2"] = {"pairs": len(keys), "mean_k1_0.6": round(float(a.mean()), 5), "mean_k1_2.4": round(float(c.mean()), 5),
                     "larger_at_0.6": round(float(np.mean(a > c)), 4), "p": p2}

    # H3: within-corpus Spearman(value_cv, saturation) >= 0.3 at k1 0.6 and 2.4.
    h3 = {}
    for k1, rows_k in (("0.6", low), ("2.4", high)):
        rs = list(rows_k.values())
        rho, p = within_corpus_spearman([value_cv(r) for r in rs], [sv(r)["saturation"] for r in rs],
                                        [r["collection"] for r in rs])
        h3[k1] = {"rows": len(rs), "rho": round(rho, 4), "p": p}
    results["H3"] = h3

    # H4: value_cv ridge fitted on primary rows ranks saturation at k1 0.6 and 2.4 with Spearman >= 0.5.
    primary = [r for r in rows if is_primary(r)]
    x = np.array([[value_cv(r)] for r in primary])
    y = np.array([sv(r)["saturation"] for r in primary])
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=10).fit(scaler.transform(x), y)
    h4 = {}
    for k1, rows_k in (("0.6", low), ("2.4", high)):
        rs = list(rows_k.values())
        prediction = model.predict(scaler.transform(np.array([[value_cv(r)] for r in rs])))
        truth = np.array([sv(r)["saturation"] for r in rs])
        result = spearmanr(prediction, truth)
        r2 = 1 - np.sum((truth - prediction) ** 2) / np.sum((truth - truth.mean()) ** 2)
        h4[k1] = {"rows": len(rs), "spearman": round(float(result.statistic), 4), "p": float(result.pvalue),
                  "r2": round(float(r2), 4)}
    results["H4"] = h4

    # H5: length's Shapley value is exactly zero at b = 0.
    zero_b = [r for r in grid if r["b"] == 0.0]
    worst = max(abs(sv(r)["length"]) for r in zero_b)
    results["H5"] = {"rows": len(zero_b), "max_abs_length_shapley": worst, "passes": worst < 1e-12}

    # H6: length larger at b = 0.75 than at b = 0.5, k1 = 1.2, paired, one-sided.
    mid, full = cell(1.2, 0.5), cell(1.2, 0.75)
    keys = sorted(set(mid) & set(full))
    m = np.array([sv(mid[k])["length"] for k in keys])
    f = np.array([sv(full[k])["length"] for k in keys])
    p6 = float(wilcoxon(f, m, alternative="greater").pvalue)
    results["H6"] = {"pairs": len(keys), "mean_b_0.5": round(float(m.mean()), 5), "mean_b_0.75": round(float(f.mean()), 5),
                     "larger_at_0.75": round(float(np.mean(f > m)), 4), "p": p6}

    # Holm over H2, H3 (two), H4 (two), H6; then apply each hypothesis's own criterion.
    labels = ["H2", "H3@0.6", "H3@2.4", "H4@0.6", "H4@2.4", "H6"]
    raw = np.array([p2, h3["0.6"]["p"], h3["2.4"]["p"], h4["0.6"]["p"], h4["2.4"]["p"], p6])
    adjusted = dict(zip(labels, holm(raw)))
    results["H2"].update(holm_p=adjusted["H2"], passes=bool(adjusted["H2"] < 0.05))
    for k1 in ("0.6", "2.4"):
        h3[k1].update(holm_p=adjusted[f"H3@{k1}"],
                      passes=bool(h3[k1]["rho"] >= 0.3 and adjusted[f"H3@{k1}"] < 0.05))
        h4[k1].update(holm_p=adjusted[f"H4@{k1}"],
                      passes=bool(h4[k1]["spearman"] >= 0.5 and adjusted[f"H4@{k1}"] < 0.05))
    results["H6"].update(holm_p=adjusted["H6"], passes=bool(adjusted["H6"] < 0.05))
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    for name in ("H1", "H2", "H3", "H4", "H5", "H6"):
        r = results[name]
        verdict = (r["passes"] if "passes" in r else all(v["passes"] for v in r.values()))
        print(f"{name}: {'PASS' if verdict else 'FAIL'}  {json.dumps(r)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
