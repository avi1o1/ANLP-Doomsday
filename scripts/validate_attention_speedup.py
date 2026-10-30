"""Compare the batched attention adapter with the per-head one it replaced, on real RULER examples.

For each example and each mode the same prompt is generated under both adapters. The text
must be identical and the reference statistics must agree. Wall-clock time per adapter is
recorded, which is the point of the rewrite.

    python -m scripts.validate_attention_speedup CONFIG TASK CONTEXT BUDGET EXAMPLES [--modes quest,h2o,111]
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

from src.artifacts import jsonl
from src.attention.adapter import AttentionController
from src.config import load_config
from src.modeling import greedy_generate, load_language_model, model_inputs
from src.scoring import Switches

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import attention_adapter_oracle as oracle  # noqa: E402


class NoStop:
    def check(self):
        pass


def controller_options(mode, spec, budget):
    if mode in {"full", "quest", "h2o"}:
        return {"mode": mode, "budget": budget, "page_size": spec["page_size"]}
    return {"mode": "corrected", "switches": Switches.parse(mode), "budget": budget, "page_size": spec["page_size"]}


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("task")
    parser.add_argument("context", type=int)
    parser.add_argument("budget", type=int)
    parser.add_argument("examples", type=int)
    parser.add_argument("--modes", default="full,quest,h2o,000,111")
    parser.add_argument("--skip-old", action="store_true", help="time the batched adapter only")
    args = parser.parse_args()
    config = load_config(args.config)
    spec = config["attention"]
    model, tokenizer, _ = load_language_model(spec)
    data = Path(spec["ruler_data"]) / str(args.context) / args.task / "validation.jsonl"
    examples = list(jsonl(data))[:args.examples]
    summary = {"model": spec["model"], "task": args.task, "context": args.context, "budget": args.budget,
               "examples": [], "all_identical": True}
    for example in examples:
        ids = model_inputs(model, tokenizer, example["input"])
        record = {"index": example["index"], "prompt_tokens": ids.shape[1], "calibration": {}, "modes": {}}
        references = {}
        for name, controller in (("new", AttentionController), ("old", oracle.AttentionController)):
            if name == "old" and args.skip_old:
                continue
            tick = time.perf_counter()
            with torch.inference_mode(), controller(model, mode="full", budget=args.budget, page_size=spec["page_size"],
                                                    calibration_samples=spec["calibration_samples"]) as ctrl:
                model(ids, use_cache=False)
                references[name] = ctrl.freeze()
            torch.cuda.synchronize()
            record["calibration"][name + "_seconds"] = time.perf_counter() - tick
        if "old" in references:
            old, new = references["old"], references["new"]
            record["calibration"]["counts_identical"] = all(old[k]["counts"] == new[k]["counts"] for k in old)
            record["calibration"]["max_relative_scale_difference"] = max(
                abs(old[k]["scale"] - new[k]["scale"]) / old[k]["scale"] for k in old)
        for mode in args.modes.split(","):
            options = controller_options(mode, spec, args.budget)
            results = {}
            for name, controller in (("new", AttentionController), ("old", oracle.AttentionController)):
                if name == "old" and args.skip_old:
                    continue
                with controller(model, reference=copy.deepcopy(references[name]), **options) as ctrl:
                    text, measures = greedy_generate(model, tokenizer, ids, 128, NoStop())
                results[name] = {"text": text, "seconds": measures["decode_seconds"],
                                 "selected_kv_tokens": ctrl.measurements["selected_kv_tokens"],
                                 "selection_seconds": ctrl.measurements["selection_seconds"]}
            entry = {k + "_seconds": v["seconds"] for k, v in results.items()}
            entry["new_selection_seconds"] = results["new"]["selection_seconds"]
            if "old" in results:
                entry["text_identical"] = results["old"]["text"] == results["new"]["text"]
                entry["selected_tokens_identical"] = results["old"]["selected_kv_tokens"] == results["new"]["selected_kv_tokens"]
                entry["speedup"] = results["old"]["seconds"] / results["new"]["seconds"]
                summary["all_identical"] &= entry["text_identical"] and entry["selected_tokens_identical"]
            record["modes"][mode] = entry
            print(json.dumps({"example": example["index"], "mode": mode, **entry}), flush=True)
        summary["examples"].append(record)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
