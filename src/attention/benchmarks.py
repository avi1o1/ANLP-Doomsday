"""Prepare and score RULER attention benchmarks with pinned source revisions."""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import yaml

from src.artifacts import Artifact, file_hash, jsonl

RULER_TASKS = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2",
               "niah_multikey_3", "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2"]
RULER_GENERATION = {"vt": 30, "cwe": 120, "fwe": 50, "qa_1": 32, "qa_2": 32}


def ruler_score(task, prediction, references):
    # Matches the upstream case-insensitive substring metric, with no per-example rounding.
    cleaned = re.sub(r"[\x00-\x1f]", "\n", prediction.strip()).strip().casefold()
    if not references or not all(isinstance(r, str) and r for r in references):
        raise ValueError("RULER requires nonempty string references")
    matches = [str(reference).casefold() in cleaned for reference in references]
    return float(any(matches)) if task.startswith("qa_") else sum(matches) / len(matches)


def prepare_ruler(config, stop):
    from huggingface_hub import snapshot_download

    spec = config["attention"]
    checkout = Path(spec["ruler_checkout"]).resolve()
    script = checkout / "scripts" / "data" / "prepare.py"
    if not script.exists():
        raise FileNotFoundError(f"Place a pinned NVIDIA/RULER checkout at {checkout}; see context/IMPLEMENTATION.md")
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    tokenizer_path = snapshot_download(spec["model"], revision=spec.get("revision", "main"),
        allow_patterns=["config.json", "tokenizer*", "special_tokens_map.json", "vocab*", "merges*", "*.model"])
    tasks = spec.get("tasks", RULER_TASKS)
    if set(tasks) != set(RULER_TASKS) and not config.get("fixture", False):
        raise ValueError("Primary attention evaluation requires all 13 RULER tasks")
    for context in spec["contexts"]:
        for task in tasks:
            stop.check()
            directory = Path(spec["ruler_data"]).resolve() / str(context) / task
            with Artifact(directory, {"task": task, "context": context, "seed": spec["seed"],
                                     "examples": spec["examples_per_task"], "model": spec["model"],
                                     "model_revision": spec.get("revision", "main"),
                                     "ruler_revision": revision, "script_sha256": file_hash(script)}) as work:
                if work.reused:
                    continue
                subprocess.run([sys.executable, str(script), "--save_dir", str(directory.parent),
                    "--benchmark", "synthetic", "--task", task, "--tokenizer_path", tokenizer_path,
                    "--tokenizer_type", "hf", "--max_seq_length", str(context), "--model_template_type", "base",
                    "--num_samples", str(spec["examples_per_task"]), "--random_seed", str(spec["seed"])],
                    check=True, cwd=script.parent)
                data = list(jsonl(directory / "validation.jsonl"))
                if len(data) != spec["examples_per_task"] or len({r["index"] for r in data}) != len(data):
                    raise ValueError("RULER generation failed count/ID validation")
                work.complete(["validation.jsonl"], ruler_revision=revision)


def verify_ruler_metric(checkout):
    """Compare our unrounded per-example aggregation against a pinned official scorer."""
    path = Path(checkout) / "scripts/eval/synthetic/constants.py"
    spec = importlib.util.spec_from_file_location("ruler_official_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    customized = yaml.safe_load((Path(checkout) / "scripts/synthetic.yaml").read_text())
    predictions, references = ["alpha beta", "nothing", "GAMMA"], [["alpha", "beta"], ["beta"], ["gamma", "delta"]]
    for task in RULER_TASKS:
        expected = module.TASKS[customized[task]["task"]]["metric_fn"](predictions, references)
        actual = round(100 * sum(ruler_score(task, p, r) for p, r in zip(predictions, references)) / len(predictions), 2)
        if actual != expected:
            raise ValueError(f"Official RULER scoring mismatch for {task}")
    return {"scorer_sha256": file_hash(path), "tasks_verified": RULER_TASKS}
