"""Frozen retrieval-predictor loading and analysis shared by transfer settings."""

from __future__ import annotations

from src.analysis import FrozenPredictor, evaluate_transfer
from src.artifacts import atomic_json, file_hash, read_json
from src.config import root


def frozen_parent(config):
    directory = root(config) / "predictor"
    if read_json(directory / "manifest.json")["status"] != "complete":
        raise ValueError("Complete and freeze retrieval prediction before corrected transfer outcomes")
    predictor = FrozenPredictor.load(directory / "predictor.json")
    return directory, predictor


def transfer_report(config, setting, predictor=None):
    if predictor is None:
        _, predictor = frozen_parent(config)
    rows = []
    for path in sorted((root(config) / "transfer" / setting).glob("*/result.json")):
        if read_json(path.parent / "manifest.json")["status"] == "complete":
            row = read_json(path)
            if not row.get("sink", False):
                rows.append(row)
    report = evaluate_transfer(predictor, rows)
    report["setting"] = setting
    report["predictor_sha256"] = file_hash(root(config) / "predictor" / "predictor.json")
    atomic_json(root(config) / "transfer" / setting / "summary.json", report)
    return report
