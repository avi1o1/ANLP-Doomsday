"""Configuration loading without implicit downloads or dataset expansion."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Configuration must be a mapping with schema_version: 1")
    for key in ("datasets", "bases", "encoder", "fitting", "budgets"):
        if key not in config:
            raise ValueError(f"Missing configuration section: {key}")
    if len({b["name"] for b in config["bases"]}) != len(config["bases"]):
        raise ValueError("Basis names must be unique")
    for name, dataset in config["datasets"].items():
        if not all(k in dataset for k in ("source", "corpus_id", "language", "role")):
            raise ValueError(f"Dataset {name} lacks source/corpus_id/language/role")
        if dataset["source"] == "ir_datasets" and not dataset.get("id"):
            raise ValueError(f"Dataset {name} lacks an ir_datasets ID")
        dataset.setdefault("exclude_identical_ids", dataset.get("id", "").startswith("beir/"))
    if config["encoder"].get("kind") == "fixture" and not config.get("fixture", False):
        raise ValueError("A fixture encoder must be labeled fixture: true")
    if not any(d["role"] == "fit" for d in config["datasets"].values()):
        raise ValueError("A fitting dataset is required")
    if len({b["name"] for b in config["budgets"]}) != len(config["budgets"]):
        raise ValueError("Budget names must be unique")
    for b in config["budgets"]:
        for side in ("query", "document"):
            if b[side] is not None and (not isinstance(b[side], int) or b[side] <= 0):
                raise ValueError("Budgets must be positive integers or null (native support)")
    config["_config_path"] = str(path)
    config["output_root"] = os.path.expandvars(config.get("output_root", "results/raw"))
    if "$" in config["output_root"]:
        raise ValueError("Unresolved environment variable in output_root")
    return config


def public_config(config: dict) -> dict:
    return {k: v for k, v in config.items() if not k.startswith("_")}


def named(items: list[dict], name: str) -> dict:
    for item in items:
        if item["name"] == name:
            return item
    raise ValueError(f"Unknown name {name!r}; available: {[i['name'] for i in items]}")


def root(config):
    return Path(config["output_root"]).resolve()
