"""How often calibration selects the page holding the needle, the sink page and the last pages.

Reads only the uncorrected calibration statistics (`reference.json.gz`) of a Setting B
row and the RULER inputs. For each example the selection rate of a page is its count over
its opportunities, averaged over layers and heads; the needle page is located by the
character offset of the first expected answer.

    python -m scripts.needle_page_rates ROW_DIRECTORY RULER_JSONL MODEL REVISION OUTPUT [--examples N]
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from src.artifacts import atomic_json, jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("row", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("model")
    parser.add_argument("revision")
    parser.add_argument("output", type=Path)
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument("--page-size", type=int, default=16)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    records = []
    for example in list(jsonl(args.data))[:args.examples]:
        with gzip.open(args.row / "examples" / str(example["index"]) / "reference.json.gz", "rt") as handle:
            reference = json.load(handle)
        encoded = tokenizer(example["input"], add_special_tokens=True, return_offsets_mapping=True)
        char = example["input"].find(example["outputs"][0])
        needle = next(i for i, (start, end) in enumerate(encoded["offset_mapping"]) if start <= char < end or start >= char)
        pages = -(-len(encoded["input_ids"]) // args.page_size)
        rate = np.zeros(pages)
        for statistics in reference.values():
            counts, opportunities = np.array(statistics["counts"]), np.array(statistics["opportunities"])
            rate[:len(counts)] += np.where(opportunities > 0, counts / np.maximum(opportunities, 1), 0)
        rate /= len(reference)
        order = list(np.argsort(-rate, kind="stable"))
        page = needle // args.page_size
        records.append({"id": example["index"], "pages": pages, "needle_page": int(page),
                        "needle_rank": order.index(page) + 1, "needle_rate": float(rate[page]),
                        "sink_rate": float(rate[0]), "last_page_rate": float(rate[-1]), "median_rate": float(np.median(rate))})
    summary = {k: [float(np.min([r[k] for r in records])), float(np.max([r[k] for r in records]))]
               for k in ("needle_rank", "needle_rate", "sink_rate", "last_page_rate", "median_rate")}
    atomic_json(args.output, {"row": args.row.name, "examples": len(records), "range": summary, "records": records})
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
