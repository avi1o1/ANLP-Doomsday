"""Attention-page experiments using a frozen retrieval predictor."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis import SWITCH_KEYS, paired_effect
from src.artifacts import Artifact, atomic_json, file_hash, jsonl, read_json
from src.attention.adapter import AttentionController
from src.attention.benchmarks import RULER_GENERATION, ruler_score, verify_ruler_metric
from src.config import digest, root
from src.diagnostics import ALL_FEATURES
from src.modeling import greedy_generate, load_language_model, model_inputs
from src.scoring import Switches
from src.transfer import frozen_parent, transfer_report


def attention_transfer(config, task, context, budget, stop, sink=False):
    import torch

    parent, predictor = frozen_parent(config)
    spec = config["attention"]
    data_dir = Path(spec["ruler_data"]).resolve() / str(context) / task
    examples = list(jsonl(data_dir / "validation.jsonl"))
    if len(examples) != spec["examples_per_task"]:
        raise ValueError("RULER example count differs from the frozen protocol")
    metric_info = verify_ruler_metric(spec["ruler_checkout"])
    model, tokenizer, model_info = load_language_model(spec)
    generated_for = read_json(data_dir / "manifest.json")["config"]
    if generated_for["model"] != model_info["model"] or generated_for["model_revision"] != model_info["revision"]:
        raise ValueError("RULER tokenizer/model mismatch; freeze the accessible model and regenerate RULER first")
    directory = root(config) / "transfer" / "attention" / f"{task}__c{context}__b{budget}__sink{int(sink)}"
    identity = {"spec": spec, "task": task, "context": context, "budget": budget, "sink": sink,
                "model": model_info, "data_sha256": file_hash(data_dir / "validation.jsonl"), **metric_info}
    with Artifact(directory, identity, [parent, data_dir]) as work:
        if work.reused:
            return directory
        outputs, diagnostics, outcomes = [], [], []
        for example in examples:
            stop.check()
            item = directory / "examples" / str(example["index"])
            ids = model_inputs(model, tokenizer, example["input"])
            maximum = RULER_GENERATION.get(task, 128)
            if ids.shape[1] + maximum > context:
                raise ValueError("Prepared RULER input exceeds declared total context; do not silently truncate")
            with Artifact(item, {"id": example["index"], "input_sha256": digest(ids.cpu().tolist()),
                                 "intervention": digest(identity)}, [parent, data_dir]) as part:
                if not part.reused:
                    reference_path = item / "reference.json.gz"
                    if reference_path.exists() and (item / "calibration.json").exists():
                        reference = read_json(reference_path)
                    else:
                        with torch.inference_mode(), AttentionController(model, mode="full", budget=budget,
                            page_size=spec["page_size"], calibration_samples=spec["calibration_samples"], sink=sink) as ctrl:
                            model(ids, use_cache=False)
                            reference = ctrl.freeze()
                            atomic_json(reference_path, reference)
                            atomic_json(item / "calibration.json", {"diagnostics": ctrl.diagnostics(),
                                        "measurements": ctrl.measurements})
                    records = []
                    for key in ["full", "quest", "h2o"] + SWITCH_KEYS:
                        path = item / f"{key}.json"
                        if not path.exists():
                            stop.check()
                            for device in range(torch.cuda.device_count()):
                                torch.cuda.reset_peak_memory_stats(device)
                            with AttentionController(model, mode=key if key in {"full", "quest", "h2o"} else "corrected",
                                switches=Switches.parse(key) if key in SWITCH_KEYS else Switches(), budget=budget,
                                page_size=spec["page_size"], sink=sink, reference=reference) as ctrl:
                                text, measures = greedy_generate(model, tokenizer, ids, maximum, stop)
                                record = {"id": example["index"], "prediction": text,
                                          "score": ruler_score(task, text, example["outputs"]),
                                          "measurements": {**measures, **ctrl.measurements}}
                            atomic_json(path, record)
                        records.append(read_json(path))
                    if records[1]["prediction"] != records[3]["prediction"]:
                        raise ValueError("Quest / all-off no-op generation parity failed")
                    part.complete(["reference.json.gz", "calibration.json"] + [f"{k}.json" for k in ["full", "quest", "h2o"] + SWITCH_KEYS])
            outcomes.append([read_json(item / f"{key}.json")["score"] for key in SWITCH_KEYS])
            diagnostics.append(read_json(item / "calibration.json")["diagnostics"])
            outputs.append(str(item / "manifest.json"))
        effect = paired_effect(outcomes, config["analysis"]["bootstrap_samples"], spec["seed"])
        row = {"row_id": digest(identity), "setting": "attention", "model": model_info, "task": task,
               "context": context, "budget": budget, "sink": sink, "examples": len(examples),
               "diagnostics": {k: float(np.mean([r[k] for r in diagnostics])) for k in ALL_FEATURES}, **effect}
        atomic_json(directory / "result.json", row)
        np.save(directory / "outcomes.npy", np.asarray(outcomes))
        atomic_json(directory / "example_artifacts.json", outputs)
        work.complete(["result.json", "outcomes.npy", "example_artifacts.json"])
    transfer_report(config, "attention", predictor)
    return directory
