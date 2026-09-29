"""Exploratory: how far label-free features predict what the switches contribute.

Targets, per row:
- the Shapley value of each switch (idf, saturation, length);
- collection_gain = nDCG@10(all on) - nDCG@10(saturation only), the value of the two
  collection statistics once saturation is applied; its sign is the decision whether to
  use them.

Feature tiers, each adding to the one before; none reads relevance labels:
- pre-registered: frequency_slope, idf_variance, frequent_activation_share;
- index: the index diagnostics of scripts/mechanism_diagnostics.py;
- rankings: how much each switch changes the top 10, and score-shape statistics
  (scripts/ranking_features.py);
- pseudo-labels: the same targets computed against the top 10 of a reference retriever
  (text BM25 by default, the dense encoder, or the two fused) in place of relevance labels.

The pseudo-label targets are also reported unfitted, as direct estimates. Models are
ridge regression (alpha by inner leave-one-group-out) and gradient-boosted trees with
fixed settings; evaluation holds out whole basis families, whole corpora, and, fitting on
primary rows only, rows at other budgets, seeds and scoring parameters. For
collection_gain the study also reports the nDCG@10 obtained by using collection
statistics only where the prediction is positive.

Usage: python -m scripts.predictor_study ROWS_JSON DIAGNOSTICS_JSON RANKING_FEATURES_JSON OUT_JSON [--reference dense]
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from scripts.shapley_attribution import is_primary, shapley

CONFIGURATIONS = [f"{i:03b}" for i in range(8)]
PRE_REGISTERED = ["frequency_slope", "idf_variance", "frequent_activation_share"]
INDEX = ["value_cv", "peak_ratio", "top_share", "length_cv", "length_log_sd", "nnz_cv"]
RANKINGS = (["disruption_idf", "disruption_saturation", "disruption_length"]
            + [f"nqc_{c}" for c in CONFIGURATIONS] + [f"gap_{c}" for c in CONFIGURATIONS])
PSEUDO = ["pseudo_idf", "pseudo_saturation", "pseudo_length", "pseudo_collection_gain", "pseudo_000", "pseudo_111"]
TIERS = {"pre-registered": PRE_REGISTERED,
         "index": PRE_REGISTERED + INDEX,
         "rankings": PRE_REGISTERED + INDEX + RANKINGS,
         "pseudo-labels": PRE_REGISTERED + INDEX + RANKINGS + PSEUDO}
TARGETS = ("saturation", "length", "idf", "collection_gain")
ALPHAS = (0.1, 1.0, 10.0, 100.0)


def targets(quality):
    values = shapley(quality)
    values["collection_gain"] = quality["111"] - quality["010"]
    return values


def ranking_key(row):
    tag = f"k{row['k1']}_b{row['b']}_{row.get('control') or 'primary'}"
    return row["collection"], f"{row['representation_id']}__s{row['seed']}", row["budget"], tag


def load_rankings(path: Path):
    return {(e["collection"], e["representation"], e["budget"], e["tag"]): e for e in json.loads(path.read_text())}


def features(row, diagnostics, rankings, reference="bm25"):
    d = diagnostics.get((row["representation_id"], row["seed"], row["budget"], row["collection"]))
    ranking = rankings.get(ranking_key(row))
    if d is None or ranking is None or f"pseudo_{reference}_000" not in ranking:
        return None
    f = {n: row["diagnostics"][n] for n in PRE_REGISTERED}
    f.update({n: d[n] for n in INDEX})
    f.update({n: ranking[n] for n in RANKINGS})
    pseudo = {c: ranking[f"pseudo_{reference}_{c}"] for c in CONFIGURATIONS}
    f.update({f"pseudo_{k}": v for k, v in targets(pseudo).items()})
    f["pseudo_000"], f["pseudo_111"] = pseudo["000"], pseudo["111"]
    return f


def ridge(alpha):
    return Ridge(alpha=alpha)


def boosted():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=10, l2_regularization=1.0, random_state=0)


def fit(kind, x, y, alpha=10.0):
    scaler = StandardScaler().fit(x)
    model = (ridge(alpha) if kind == "ridge" else boosted()).fit(scaler.transform(x), y)
    return lambda z: model.predict(scaler.transform(z))


def choose_alpha(x, y, groups, train):
    inner = sorted(set(groups[train]))
    if len(inner) < 3:
        return 10.0
    scores = []
    for alpha in ALPHAS:
        errors = []
        for h in inner:
            a, b = train & (groups != h), train & (groups == h)
            errors.append(np.mean((fit("ridge", x[a], y[a], alpha)(x[b]) - y[b]) ** 2))
        scores.append((np.mean(errors), alpha))
    return min(scores)[1]


def score(y, prediction):
    r2 = 1 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2)
    return {"r2": round(float(r2), 4), "spearman": round(float(spearmanr(y, prediction).statistic), 4),
            "sign_accuracy": round(float(np.mean(np.sign(prediction) == np.sign(y))), 4)}


def grouped(kind, x, y, groups):
    prediction = np.zeros(len(y))
    for g in sorted(set(groups)):
        train = groups != g
        alpha = choose_alpha(x, y, groups, train) if kind == "ridge" else None
        prediction[~train] = fit(kind, x[train], y[train], alpha or 10.0)(x[~train])
    return prediction


def decision(rows, prediction, samples=2000):
    """nDCG@10 when collection statistics are used only where the prediction is positive.

    The interval is for predicted minus always-on, resampling whole collections.
    """
    on = np.array([r["quality"]["111"] for r in rows])
    off = np.array([r["quality"]["010"] for r in rows])
    chosen = np.where(prediction > 0, on, off)
    collections = np.array([r["collection"] for r in rows])
    names = sorted(set(collections))
    members = [np.flatnonzero(collections == c) for c in names]
    rng = np.random.default_rng(0)
    gains = []
    for _ in range(samples):
        index = np.concatenate([members[i] for i in rng.integers(len(names), size=len(names))])
        gains.append(chosen[index].mean() - on[index].mean())
    return {"always_on": round(float(on.mean()), 5), "never": round(float(off.mean()), 5),
            "predicted": round(float(chosen.mean()), 5), "oracle": round(float(np.maximum(on, off).mean()), 5),
            "headroom_recovered": round(float((chosen.mean() - on.mean()) / (np.maximum(on, off).mean() - on.mean())), 4),
            "gain_over_always_on_ci95": [round(float(v), 5) for v in np.percentile(gains, [2.5, 97.5])]}


def configuration_agreement(rows, rankings, reference):
    """Within-row Spearman between true and pseudo-label nDCG@10 over the eight configurations."""
    by_collection = {}
    for r in rows:
        f = rankings.get(ranking_key(r))
        if f is None or f"pseudo_{reference}_000" not in f:
            continue
        true = [r["quality"][c] for c in CONFIGURATIONS]
        pseudo = [f[f"pseudo_{reference}_{c}"] for c in CONFIGURATIONS]
        if np.std(true) > 0 and np.std(pseudo) > 0:
            by_collection.setdefault(r["collection"], []).append(spearmanr(true, pseudo).statistic)
    every = [v for values in by_collection.values() for v in values]
    summary = {"rows": len(every), "median": round(float(np.median(every)), 4)}
    summary["by_collection"] = {c: round(float(np.median(v)), 4) for c, v in sorted(by_collection.items())}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rows", type=Path)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("ranking_features", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", default="bm25", choices=["bm25", "dense", "fused"])
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    rows = json.loads(args.rows.read_text())
    diagnostics = {(d["representation"], d["seed"], d["budget"], d["collection"]): d
                   for d in json.loads(args.diagnostics.read_text())}
    rankings = load_rankings(args.ranking_features)
    usable = []
    for r in rows:
        if r.get("control"):
            continue
        f = features(r, diagnostics, rankings, args.reference)
        if f is not None and np.isfinite(list(f.values())).all():
            usable.append((r, f))
    primary = [(r, f) for r, f in usable if is_primary(r)]
    default = [(r, f) for r, f in usable if r["k1"] == 1.2 and r["b"] == 0.75 and not is_primary(r)]
    held = {"other budgets": [(r, f) for r, f in default if r["seed"] == 0],
            "other seeds": [(r, f) for r, f in default if r["seed"] != 0],
            "other parameters": [(r, f) for r, f in usable if (r["k1"], r["b"]) != (1.2, 0.75)]}
    report = {"reference": args.reference, "rows": {"primary": len(primary), **{k: len(v) for k, v in held.items()}},
              "configuration_agreement": configuration_agreement(
                  [r for r in rows if not r.get("control")], rankings, args.reference),
              "targets": {}}
    for target in TARGETS:
        y = np.array([targets(r["quality"])[target] for r, _ in primary])
        entry = {"unfitted pseudo-label estimate": {}}
        for axis, rs in (("primary", primary), *held.items()):
            truth = np.array([targets(r["quality"])[target] for r, _ in rs])
            estimate = np.array([f[f"pseudo_{target}"] for _, f in rs])
            entry["unfitted pseudo-label estimate"][axis] = score(truth, estimate)
            if target == "collection_gain":
                entry["unfitted pseudo-label estimate"][axis]["decision"] = decision([r for r, _ in rs], estimate)
        for tier, names in TIERS.items():
            x = np.array([[f[n] for n in names] for _, f in primary])
            for kind in ("ridge", "boosted"):
                result = {}
                for axis, key in (("family", "family"), ("corpus", "corpus_id")):
                    groups = np.array([r[key] for r, _ in primary])
                    prediction = grouped(kind, x, y, groups)
                    result[axis] = score(y, prediction)
                    if target == "collection_gain":
                        result[axis]["decision"] = decision([r for r, _ in primary], prediction)
                model = fit(kind, x, y)
                for axis, rs in held.items():
                    truth = np.array([targets(r["quality"])[target] for r, _ in rs])
                    prediction = model(np.array([[f[n] for n in names] for _, f in rs]))
                    result[axis] = score(truth, prediction)
                    if target == "collection_gain":
                        result[axis]["decision"] = decision([r for r, _ in rs], prediction)
                entry[f"{tier} / {kind}"] = result
        report["targets"][target] = entry
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["rows"]), json.dumps(report["configuration_agreement"]))
    for target, entry in report["targets"].items():
        print(target)
        for name, result in entry.items():
            cells = " | ".join(f"{axis} {v['r2']:+.2f}/{v['spearman']:+.2f}/{v['sign_accuracy']:.2f}"
                               for axis, v in result.items())
            print(f"  {name:32s} {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
