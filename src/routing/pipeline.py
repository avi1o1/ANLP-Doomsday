"""Expert-routing experiments using a frozen retrieval predictor."""

from __future__ import annotations

import math
import re
import time
from pathlib import Path

import numpy as np

from src.analysis import SWITCH_KEYS, paired_effect
from src.artifacts import Artifact, atomic_json, file_hash, jsonl, read_json
from src.config import digest, root
from src.modeling import greedy_generate, load_language_model, model_inputs, synchronize
from src.routing.adapter import RoutingController
from src.scoring import Switches
from src.transfer import frozen_parent, transfer_report


def token_windows(tokenizer, records, size):
    # Standard disjoint WikiText stream windows; all but each window's initial token are scored.
    text = "\n\n".join(r["text"] for r in records)
    tokens = tokenizer(text, add_special_tokens=False).input_ids
    return [tokens[i:i+size] for i in range(0, len(tokens), size) if len(tokens[i:i+size]) >= 2]


def choice_likelihood(model, tokenizer, prompt, choices):
    import torch

    ids = model_inputs(model, tokenizer, prompt)
    result = []
    with torch.inference_mode():
        for choice in choices:
            suffix = tokenizer(choice, add_special_tokens=False).input_ids
            continuation = torch.tensor([suffix], device=ids.device)
            joined = torch.cat([ids, continuation], dim=-1)
            output = model(joined, use_cache=False).logits.float()
            start = ids.shape[1] - 1
            logp = torch.log_softmax(output[0, start:start+len(suffix)], dim=-1)
            result.append(float(logp.gather(1, continuation.T).sum()))
    return result


def routing_transfer(config, benchmark, stop, subject=None):
    import torch
    import torch.nn.functional as F

    parent, predictor = frozen_parent(config)
    spec = config["routing"]
    data_dir = Path(spec["data"]).resolve() / benchmark
    records = list(jsonl(data_dir / "test.jsonl"))
    if benchmark == "mmlu":
        subjects = sorted({r["subject"] for r in records})
        if subject is None:
            return [routing_transfer(config, benchmark, stop, name) for name in subjects]
        records = [r for r in records if r["subject"] == subject]
        if not records:
            raise ValueError("Unknown MMLU subject")
    model, tokenizer, model_info = load_language_model(spec, routing=True)
    label = benchmark + (f"__{subject}" if subject else "")
    directory = root(config) / "transfer" / "routing" / label
    identity = {"benchmark": benchmark, "subject": subject, "spec": spec, "model": model_info,
                "data_sha256": file_hash(data_dir / "test.jsonl"),
                "quality": "negative_mean_token_nll" if benchmark == "wikitext" else "accuracy"}
    with Artifact(directory, identity, [parent, data_dir]) as work:
        if work.reused:
            return directory
        if benchmark == "wikitext":
            windows = token_windows(tokenizer, records, spec["sequence_length"])
            examples = [{"id": str(i), "tokens": ids} for i, ids in enumerate(windows)]
            calibration = token_windows(tokenizer, jsonl(data_dir / "train.jsonl"), spec["sequence_length"])
            calibration = calibration[:spec["calibration_documents"]]
        elif benchmark == "mmlu":
            examples = records
            # Only prompts, no answer continuations or labels, contribute to subject statistics.
            calibration = [tokenizer(r["prompt"], add_special_tokens=True).input_ids for r in records]
        elif benchmark == "gsm8k":
            examples = records
            calibration = [tokenizer(r["prompt"], add_special_tokens=True).input_ids
                           for r in list(jsonl(data_dir / "train.jsonl"))[:spec["calibration_documents"]]]
        else:
            raise ValueError("Routing benchmark must be wikitext, mmlu, or gsm8k")
        reference_path = directory / "reference.json"
        if reference_path.exists() and (directory / "diagnostics.json").exists():
            reference = read_json(reference_path)
        else:
            with torch.inference_mode(), RoutingController(model, preserve_experts=spec.get("preserve_experts", 0)) as ctrl:
                for ids in calibration:
                    stop.check()
                    if len(ids) > model.config.max_position_embeddings:
                        raise ValueError("Calibration context exceeds model capacity; do not silently truncate")
                    model(torch.tensor([ids], device=model.get_input_embeddings().weight.device), use_cache=False)
                reference = ctrl.freeze()
                atomic_json(reference_path, reference)
                atomic_json(directory / "diagnostics.json", ctrl.diagnostics())
        outcomes, counts, outputs = [], [], ["reference.json", "diagnostics.json"]
        for example in examples:
            row_scores = []
            for key in ["native", "frequency_bias"] + SWITCH_KEYS:
                stop.check()
                path = directory / "examples" / digest(example["id"])[:16] / f"{key}.json"
                if not path.exists():
                    with torch.inference_mode(), RoutingController(model,
                        switches=Switches.parse(key) if key in SWITCH_KEYS else Switches(), reference=reference,
                        mode=key if key in {"native", "frequency_bias"} else "corrected",
                        bias_strength=spec["bias_strength"], preserve_experts=spec.get("preserve_experts", 0)) as ctrl:
                        synchronize()
                        tick = time.perf_counter()
                        if benchmark == "wikitext":
                            ids = torch.tensor([example["tokens"]], device=model.get_input_embeddings().weight.device)
                            logits = model(ids, use_cache=False).logits[:, :-1].float()
                            losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum")
                            prediction = {"nll_sum": float(losses), "tokens": ids.shape[1]-1}
                            score = -float(losses) / prediction["tokens"]
                        elif benchmark == "mmlu":
                            scores = choice_likelihood(model, tokenizer, example["prompt"], example["choices"])
                            prediction = {"choice_scores": scores, "choice": int(np.argmax(scores))}
                            score = float(prediction["choice"] == example["answer"])
                        else:
                            text, measures = greedy_generate(model, tokenizer,
                                model_inputs(model, tokenizer, example["prompt"]), spec.get("max_new_tokens", 256), stop)
                            numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
                            answer = numbers[-1].replace(",", "") if numbers else ""
                            prediction = {"text": text, "extracted_answer": answer, "measurements": measures}
                            score = float(answer == example["answer"].replace(",", ""))
                        synchronize()
                        atomic_json(path, {"id": example["id"], "prediction": prediction, "score": score,
                                           "seconds": time.perf_counter()-tick})
                result = read_json(path)
                if key in SWITCH_KEYS:
                    row_scores.append(result["score"])
                outputs.append(str(path.relative_to(directory)))
            native = read_json(path.parent / "native.json")
            noop = read_json(path.parent / "000.json")
            comparable_native = {k: v for k, v in native["prediction"].items() if k != "measurements"}
            comparable_noop = {k: v for k, v in noop["prediction"].items() if k != "measurements"}
            if not spec.get("preserve_experts", 0) and comparable_native != comparable_noop:
                raise ValueError("Routing native / all-off no-op parity failed")
            outcomes.append(row_scores)
            counts.append(len(example["tokens"])-1 if benchmark == "wikitext" else 1)
        effect = paired_effect(outcomes, config["analysis"]["bootstrap_samples"], 0, weights=counts)
        row = {"row_id": digest(identity), "setting": "routing", "benchmark": benchmark, "subject": subject,
               "model": model_info, "diagnostics": read_json(directory / "diagnostics.json"),
               "quality_measure": identity["quality"], **effect}
        if benchmark == "wikitext":
            row["perplexity"] = {key: math.exp(-quality) for key, quality in effect["quality"].items()}
        atomic_json(directory / "result.json", row)
        np.save(directory / "outcomes.npy", np.asarray(outcomes))
        atomic_json(directory / "example_weights.json", counts)
        work.complete(outputs + ["result.json", "outcomes.npy", "example_weights.json"])
    transfer_report(config, "routing", predictor)
    return directory
