"""Order-free attribution of the all-on gain to IDF, saturation and length normalisation.

The primary margin compares all switches on against all off. The all-off scorer is a
raw inner product of non-negative features, which is close to useless for many
representations, so that contrast mostly measures recovery from a broken baseline and
credits every switch with the recovery. The Shapley value of a switch averages its
marginal effect over every order in which the three switches can be turned on, using
all eight configurations each row already has. The three values sum exactly to the
all-on minus all-off gain, and a switch that never changes the outcome gets zero.

IDF and length normalisation read collection statistics (document frequencies and the
average length); saturation reads only the document itself. The collection-statistics
share of a gain is therefore the IDF value plus the length value over the total.

Usage: python -m scripts.shapley_attribution ROWS_JSON OUTPUT_DIR [--label NAME]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

SWITCHES = ("idf", "saturation", "length")  # bit order of the configuration keys


def configuration(members):
    return "".join("1" if name in members else "0" for name in SWITCHES)


def shapley(quality):
    """Shapley value of each switch from a mapping of the eight configuration keys to quality."""
    n, values = len(SWITCHES), {}
    for switch in SWITCHES:
        others = [s for s in SWITCHES if s != switch]
        total = 0.0
        for k in range(len(others) + 1):
            weight = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            for subset in itertools.combinations(others, k):
                total += weight * (quality[configuration({*subset, switch})] - quality[configuration(subset)])
        values[switch] = total
    return values


def is_primary(row):
    return (row["seed"] == 0 and row["budget"] == "q32_d128" and row["k1"] == 1.2 and row["b"] == 0.75
            and not row.get("control"))


def collection_share(values, totals):
    return (values[:, 0].sum() + values[:, 2].sum()) / totals.sum()


def corpus_bootstrap(rows, values, totals, samples=4000, seed=0):
    """95% interval of the collection-statistics share, resampling whole collections."""
    rng = np.random.default_rng(seed)
    collections = sorted({r["collection"] for r in rows})
    members = {c: [i for i, r in enumerate(rows) if r["collection"] == c] for c in collections}
    draws = []
    for _ in range(samples):
        index = [i for c in rng.choice(collections, len(collections)) for i in members[c]]
        draws.append(collection_share(values[index], totals[index]))
    return [float(q) for q in np.quantile(draws, [0.025, 0.975])]


def summarise(rows):
    values = np.array([[shapley(r["quality"])[s] for s in SWITCHES] for r in rows])
    totals = np.array([r["quality"]["111"] - r["quality"]["000"] for r in rows])
    given_saturation = np.array([r["quality"]["111"] - r["quality"]["010"] for r in rows])
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["basis"]].append(i)
    shares = values.sum(0) / totals.sum()
    return {
        "rows": len(rows), "collections": len({r["collection"] for r in rows}),
        "mean_shapley": dict(zip(SWITCHES, values.mean(0).round(6).tolist())),
        "share_of_gain": dict(zip(SWITCHES, shares.round(4).tolist())),
        "collection_statistics_share": round(float(shares[0] + shares[2]), 4),
        "collection_statistics_share_ci95": [round(x, 4) for x in corpus_bootstrap(rows, values, totals)],
        "collection_statistics_given_saturation": {
            "mean": round(float(given_saturation.mean()), 6),
            "positive_fraction": round(float(np.mean(given_saturation > 0)), 4),
            "share_of_gain": round(float(given_saturation.sum() / totals.sum()), 4)},
        "collection_statistics_share_by_basis": {
            b: round(float(collection_share(values[i], totals[i])), 4) for b, i in sorted(by.items())},
    }, values


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rows", type=Path, help="an analysis/rows.json")
    parser.add_argument("output", type=Path, help="directory for shapley_rows.csv and shapley_summary.json")
    parser.add_argument("--label", default="run")
    args = parser.parse_args()
    rows = json.loads(args.rows.read_text())
    primary = [r for r in rows if is_primary(r)]
    summary, _ = summarise(primary)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "shapley_rows.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "collection", "basis", "granularity", "seed", "budget", "k1", "b", "control",
                         "gain", *[f"shapley_{s}" for s in SWITCHES]])
        for r in rows:
            v = shapley(r["quality"])
            writer.writerow([r["row_id"], r["collection"], r["basis"], r["granularity"], r["seed"], r["budget"],
                             r["k1"], r["b"], r.get("control") or "", r["quality"]["111"] - r["quality"]["000"],
                             *[v[s] for s in SWITCHES]])
    (args.output / "shapley_summary.json").write_text(json.dumps({"label": args.label, "primary": summary},
                                                                 indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
