"""Registered SPLADE-v3 test: does the label-free rule detect where the correction hurts?

`results/registration-2026-10-20.md`, Section 4. SPLADE-v3's stored rankings under the
eight configurations are scored against the top 10 of the BM25, dense and fused references
with `scripts.ranking_features.row_features`, the computation behind the mid report's
pseudo-labels. The external runs are linked into that script's directory layout, so the
code is unchanged. G~ = pseudo(111) - pseudo(010) is compared with G = Q111 - Q010.

    python -m scripts.splade_label_free EXTERNAL_DIRECTORY BASELINE_ROOT WORK OUTPUT

BASELINE_ROOT holds baseline_evaluations/<collection>/{bm25,dense}.
"""

import argparse
import json
from pathlib import Path

from scripts.ranking_features import CONFIGURATIONS, REFERENCES, row_features
from scripts.transfer_study import label_free
from src.artifacts import atomic_json

EXCLUDED = {"miracl_bn", "miracl_hi", "miracl_te"}


def run_directory(collection_directory):
    runs = [p for p in collection_directory.iterdir() if p.is_dir() and (p / "result.json").exists()]
    if len(runs) != 1:
        raise ValueError(f"Expected one SPLADE run under {collection_directory}, found {len(runs)}")
    return runs[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("external", type=Path)
    parser.add_argument("baseline_root", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows, pseudo = [], {name: [] for name in REFERENCES}
    for collection in sorted(p.name for p in args.external.iterdir() if p.is_dir()):
        run = run_directory(args.external / collection)
        view = args.work / "view" / collection / "splade-v3" / "full" / "primary"
        view.parent.mkdir(parents=True, exist_ok=True)
        if not view.exists():
            view.symlink_to(run.resolve(), target_is_directory=True)
        out = args.work / "features" / f"{collection}.json"
        if not out.exists():
            row_features((str(args.baseline_root), str(view), str(out)))
        features = json.loads(out.read_text())
        quality = json.loads((run / "result.json").read_text())["quality"]
        entry = {"collection": collection, "quality": quality, "features": features}
        print(collection, features["queries"], {n: features.get(f"{n}_queries") for n in REFERENCES}, flush=True)
        if collection in EXCLUDED:
            continue
        rows.append({"task": collection, "budget": None, "quality": quality, "entry": entry})
        for name in REFERENCES:
            pseudo[name].append({c: features[f"pseudo_{name}_{c}"] for c in CONFIGURATIONS})
    report = {"collections": [r["task"] for r in rows], "excluded": sorted(EXCLUDED),
              "references": {name: label_free(rows, pseudo[name], None) for name in REFERENCES},
              "features": {r["task"]: r["entry"]["features"] for r in rows}}
    atomic_json(args.output, report)
    for name in REFERENCES:
        x = report["references"][name]
        d = x["decision"]
        print(f"{name}: sign {x['sign_accuracy']:.2f} vs better constant {x['better_constant_sign']:.2f}, "
              f"spearman {x['spearman']:.2f}, rule {d['rule']:.4f} always-on {d['always_on']:.4f} "
              f"always-off {d['always_off']:.4f} oracle {d['oracle']:.4f}, pass={x['passes']}")


if __name__ == "__main__":
    main()
