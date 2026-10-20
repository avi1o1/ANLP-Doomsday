"""Label-free pseudo-quality for one Setting B row: exact attention mass each configuration captures.

For every example of the row the uncorrected model regenerates its full-attention answer
with `src.attention.mass.MassProbe`, using the example's stored calibration reference. Only
`reference.json.gz` and `full.json` are read from the row; corrected outcomes are not. The
regenerated text is compared with the stored full-attention prediction as a parity check.

    python -m scripts.attention_mass_reference CONFIG ROW_DIRECTORY OUTPUT
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from src.analysis import SWITCH_KEYS
from src.artifacts import atomic_json, jsonl, read_json
from src.attention.benchmarks import RULER_GENERATION
from src.attention.mass import MassProbe
from src.config import load_config
from src.modeling import greedy_generate, load_language_model, model_inputs

ROW = re.compile(r"^(?P<task>.+)__c(?P<context>\d+)__b(?P<budget>\d+)__sink(?P<sink>[01])$")


class NoStop:
    def check(self):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("row", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    match = ROW.match(args.row.name)
    if match is None:
        sys.exit(f"Not a Setting B row directory: {args.row}")
    task, context, budget, sink = match["task"], int(match["context"]), int(match["budget"]), match["sink"] == "1"
    spec = load_config(args.config)["attention"]
    model, tokenizer, model_info = load_language_model(spec)
    examples = list(jsonl(Path(spec["ruler_data"]) / str(context) / task / "validation.jsonl"))
    records = []
    for example in examples:
        item = args.row / "examples" / str(example["index"])
        reference = read_json(item / "reference.json.gz")
        stored = read_json(item / "full.json")["prediction"]
        ids = model_inputs(model, tokenizer, example["input"])
        with MassProbe(model, reference, budget, spec["page_size"], sink) as probe:
            text, _ = greedy_generate(model, tokenizer, ids, RULER_GENERATION.get(task, 128), NoStop())
        records.append({"id": example["index"], "shares": probe.shares(), "probed_calls": probe.calls,
                        "full_text_identical": text == stored})
        print(f"{task} b{budget} example {example['index']}: identical={text == stored} calls={probe.calls}",
              flush=True)
    defined = [r for r in records if r["shares"] is not None]
    row = {"task": task, "context": context, "budget": budget, "sink": sink, "model": model_info,
           "examples": len(records), "examples_probed": len(defined),
           "full_text_identical": sum(r["full_text_identical"] for r in records),
           "pseudo_quality": {k: float(np.mean([r["shares"][k] for r in defined])) for k in SWITCH_KEYS}
           if defined else None,
           "records": records}
    atomic_json(args.output, row)


if __name__ == "__main__":
    main()
