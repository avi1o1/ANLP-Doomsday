"""Merge two diagnosed row sets and redo Holm correction over the union.

Each input comes from `csx diagnose` with the same significance sample count. Holm
adjusts within a family, so the primary all-on versus all-off family and each component's
main-effect family are corrected again across the merged rows, in the order of the inputs.

    python -m scripts.merge_analysis_rows OUT_DIR BASE_ROWS BASE_ABLATIONS NEW_ROWS NEW_ABLATIONS
"""

import argparse
import json
from pathlib import Path

from src.analysis import holm

PRIMARY = "primary_all_on_vs_all_off"


def merge(base_rows, base_ablations, new_rows, new_ablations):
    rows, ablations = base_rows + new_rows, base_ablations + new_ablations
    ids = [r["row_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("The two row sets share row IDs")
    if len({r["bootstrap_samples"] for r in rows}) != 1:
        raise ValueError("Row sets use different significance sample counts")
    primary = [r for r in rows if r.get("comparison_family") == PRIMARY and r.get("p_value") is not None]
    for record, p in zip(primary, holm([r["p_value"] for r in primary])):
        record["holm_p"] = p
    for component in ("idf", "saturation", "length"):
        group = [a for a in ablations if a["component"] == component and a["p_value"] is not None]
        for record, p in zip(group, holm([a["p_value"] for a in group])):
            record["holm_p"] = p
    return rows, ablations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    for name in ("base_rows", "base_ablations", "new_rows", "new_ablations"):
        parser.add_argument(name)
    args = parser.parse_args()
    load = lambda path: json.loads(Path(path).read_text())
    rows, ablations = merge(load(args.base_rows), load(args.base_ablations),
                            load(args.new_rows), load(args.new_ablations))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(rows, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (out / "component_ablations.json").write_text(json.dumps(ablations, indent=2, sort_keys=True, allow_nan=False) + "\n")
    primary = [r for r in rows if r.get("comparison_family") == PRIMARY]
    print(f"{len(rows)} rows, {len(ablations)} ablations, {len(primary)} primary; "
          f"{sum(r['holm_p'] < 0.05 for r in primary)} primary rows significant at 0.05 after Holm")


if __name__ == "__main__":
    main()
