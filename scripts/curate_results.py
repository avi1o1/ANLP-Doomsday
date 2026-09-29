"""Copy a finished run's summary results into a directory small enough to commit.

Keeps what a reader of the results needs and nothing that is regenerable or bulky:
the report directory, the analysis rows and component ablations, and the predictor
checkpoints. The predictor's validation.json is reduced to validation_summary.json,
which keeps fold metrics, aggregate metrics, ordering and uncertainty and drops the
per-example predictions. Per-query records, encodings and indexes stay behind.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPORT = ("report.md", "retrieval.csv", "baselines.json", "indic_macro.json", "resources.csv",
          "predictor.csv", "inventory.json", "transfer_rows.json", "retrieval_effects.pdf")
ANALYSIS = ("rows.json", "component_ablations.json")
PREDICTOR = ("predictor.json", "english_predictor.json", "excluded_rows.json", "manifest.json")


def copy(source: Path, target: Path, names):
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in names:
        if (source / name).exists():
            shutil.copy2(source / name, target / name)
            copied.append(name)
    return copied


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    root, dest = args.output_root, args.destination
    done = {"report": copy(root / "report", dest / "report", REPORT),
            "analysis": copy(root / "analysis", dest / "analysis", ANALYSIS),
            "predictor": copy(root / "predictor", dest / "predictor", PREDICTOR)}
    validation = root / "predictor" / "validation.json"
    if validation.exists():
        records = json.loads(validation.read_text())
        summary = [{k: v for k, v in r.items() if k != "predictions"} for r in records]
        (dest / "predictor" / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        done["predictor"].append("validation_summary.json")
    for part, names in done.items():
        print(f"{part}: {', '.join(names) or 'nothing'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
