"""Compile Setting B rows into one compact file for analysis away from the cluster.

For each completed row this keeps the row summary (`result.json`) and, per example, the
score of every mode, the decode measurements, and the token F1 between each switch
configuration's output and the full-attention output (the secondary label-free reference
of the transfer registration). Predictions themselves are not copied.

    python -m scripts.collect_attention_rows ATTENTION_DIRECTORY OUTPUT
"""

import argparse
import re
from collections import Counter
from pathlib import Path

from src.analysis import SWITCH_KEYS
from src.artifacts import atomic_json, read_json

MODES = ["full", "quest", "h2o"] + SWITCH_KEYS
ROW_FIELDS = ["row_id", "task", "context", "budget", "sink", "examples", "model", "diagnostics", "quality",
              "raw_delta", "raw_ci95", "spread", "margin", "margin_ci95", "margin_variance", "p_value",
              "p_value_floor", "bootstrap_samples", "n_examples"]


def tokens(text):
    # The cleaning of `ruler_score`, then whitespace tokens.
    return re.sub(r"[\x00-\x1f]", "\n", text.strip()).strip().casefold().split()


def token_f1(prediction, reference):
    p, r = tokens(prediction), tokens(reference)
    if not p and not r:
        return 1.0
    common = sum((Counter(p) & Counter(r)).values())
    if not common:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def collect(row):
    result = read_json(row / "result.json")
    entry = {k: result.get(k) for k in ROW_FIELDS}
    examples = []
    for item in sorted((row / "examples").iterdir(), key=lambda p: int(p.name)):
        records = {mode: read_json(item / f"{mode}.json") for mode in MODES}
        full = records["full"]["prediction"]
        examples.append({
            "id": int(item.name),
            "scores": {mode: records[mode]["score"] for mode in MODES},
            "f1_to_full": {key: token_f1(records[key]["prediction"], full) for key in SWITCH_KEYS},
            "generated_tokens": {mode: records[mode]["measurements"].get("generated_tokens") for mode in MODES},
            "decode_seconds": {mode: records[mode]["measurements"].get("decode_seconds") for mode in MODES},
            "selected_kv_tokens": {mode: records[mode]["measurements"].get("selected_kv_tokens") for mode in MODES}})
    entry["per_example"] = examples
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = []
    for row in sorted(p for p in args.directory.iterdir() if p.is_dir()):
        manifest = row / "manifest.json"
        if manifest.exists() and read_json(manifest)["status"] == "complete" and (row / "result.json").exists():
            rows.append(collect(row))
            print(row.name, len(rows[-1]["per_example"]), flush=True)
    atomic_json(args.output, rows, compact=True)


if __name__ == "__main__":
    main()
