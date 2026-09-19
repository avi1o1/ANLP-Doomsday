import subprocess
import sys
from types import SimpleNamespace

from src.cli import parser
from src.config import load_config
from src.experiments import write_manifest
from src.launch import run_stage


def test_cli_storage_defaults_and_explicit_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("CONFIG", "configs/gpu-smoke.yaml")
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "default"))
    args = parser().parse_args(["smoke"])
    assert args.config == "configs/gpu-smoke.yaml" and args.output_root == str(tmp_path / "default")
    args = parser().parse_args(["--config", "configs/fixture.yaml", "--output-root", str(tmp_path / "explicit"), "smoke"])
    assert args.config == "configs/fixture.yaml" and args.output_root == str(tmp_path / "explicit")
    smoke = load_config("configs/gpu-smoke.yaml")
    assert smoke["fixture"] and smoke["encoder"]["kind"] == "e5" and smoke["encoder"]["device"] == "cuda"


def test_saved_manifest_output_wins_over_inherited_default(tmp_path, monkeypatch):
    config = load_config("configs/fixture.yaml")
    config["output_root"] = str(tmp_path / "explicit-run")
    manifest = write_manifest(config, tmp_path / "manifest")
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "wrong-run"))
    stage = next(s for s in manifest["stages"] if s["name"] == "prepare")
    assert run_stage(stage, manifest["config"], tmp_path / "logs", [], 1, 1, SimpleNamespace(requested=False)) == 0
    assert (tmp_path / "explicit-run/datasets/collections/train/manifest.json").is_file()
    assert not (tmp_path / "wrong-run").exists()
    # The standalone manifest task entry point must preserve the same resolved path.
    subprocess.run([sys.executable, "-m", "src", "task", "--manifest", stage["tasks"],
                    "--task-index", "0", "--task-config", manifest["config"]], check=True, capture_output=True)
    assert not (tmp_path / "wrong-run").exists()
