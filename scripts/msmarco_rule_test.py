"""The mid report's unfitted decision rule on MS MARCO rows produced after it was published.

The rule uses collection statistics (configuration 111 rather than 010) where the
pseudo-label estimate G~ is positive. It has no fitted parameter and was fixed in the mid
report (tag mid-submission) before the 84 MS MARCO budget rows existed. The 42 primary
MS MARCO rows are reported separately; they existed earlier. The three collections share
one corpus, so their rows are not independent.

    python -m scripts.msmarco_rule_test MSMARCO_ROWS RANKING_FEATURES OUTPUT
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from scripts.predictor_study import load_rankings, ranking_key, targets
from scripts.ranking_features import CONFIGURATIONS, REFERENCES
from src.artifacts import atomic_json


def evaluate(entries):
    estimate, gain, on, off = (np.array(v) for v in zip(*entries))
    chosen = np.where(estimate > 0, on, off)
    oracle = np.maximum(on, off)
    nonzero = gain != 0
    positive = float(np.mean(gain[nonzero] > 0))
    return {"rows": len(entries), "sign_accuracy": float(np.mean(np.sign(estimate[nonzero]) == np.sign(gain[nonzero]))),
            "constant_positive": positive, "better_constant": max(positive, 1 - positive),
            "spearman": float(spearmanr(estimate, gain).statistic),
            "always_on": float(on.mean()), "always_off": float(off.mean()), "rule": float(chosen.mean()),
            "oracle": float(oracle.mean()),
            "headroom_recovered": float((chosen.mean() - on.mean()) / (oracle.mean() - on.mean()))
            if oracle.mean() > on.mean() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rows", type=Path)
    parser.add_argument("ranking_features", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = [r for r in json.loads(args.rows.read_text()) if not r.get("control")]
    rankings = load_rankings(args.ranking_features)
    report = {}
    for name in REFERENCES:
        groups = {}
        for r in rows:
            f = rankings.get(ranking_key(r))
            if f is None or f"pseudo_{name}_000" not in f:
                continue
            pseudo = {c: f[f"pseudo_{name}_{c}"] for c in CONFIGURATIONS}
            group = "primary_q32_d128" if r["budget"] == "q32_d128" else "new_budget_rows"
            groups.setdefault(group, []).append((targets(pseudo)["collection_gain"], targets(r["quality"])["collection_gain"],
                                                 r["quality"]["111"], r["quality"]["010"]))
        report[name] = {group: evaluate(entries) for group, entries in sorted(groups.items())}
    atomic_json(args.output, report)
    for name, groups in report.items():
        for group, x in groups.items():
            print(f"{name:5s} {group:17s} n={x['rows']} sign {x['sign_accuracy']:.3f} vs {x['better_constant']:.3f} "
                  f"spearman {x['spearman']:.2f} recovered {x['headroom_recovered']}")


if __name__ == "__main__":
    main()
