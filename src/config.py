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


def merged(base: dict, override: dict) -> dict:
    """Mappings merge key by key; any other value, lists included, replaces the base."""
    result = dict(base)
    for key, value in override.items():
        result[key] = merged(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def _read(path: Path, seen=()) -> dict:
    """One configuration file, with an optional `extends:` base resolved beside it.

    A machine configuration can then state only what differs from the scientific
    defaults, and the scientific settings stay in one place.
    """
    if path in seen:
        raise ValueError(f"Configuration extends itself: {path}")
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if isinstance(config, dict) and "extends" in config:
        base = _read((path.parent / config.pop("extends")).resolve(), (*seen, path))
        config = merged(base, config)
    return config


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    config = _read(path)
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
