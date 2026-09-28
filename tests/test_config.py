import pytest

from src.config import load_config, merged


def test_merged_replaces_lists_and_merges_mappings():
    base = {"a": {"x": 1, "y": [1, 2]}, "b": 1}
    assert merged(base, {"a": {"y": [3]}, "c": 2}) == {"a": {"x": 1, "y": [3]}, "b": 1, "c": 2}
    assert base == {"a": {"x": 1, "y": [1, 2]}, "b": 1}


def test_machine_config_changes_execution_only(tmp_path):
    research, machine = load_config("configs/research.yaml"), load_config("configs/sustaind.yaml")
    changed = {k for k in set(research) | set(machine) if research.get(k) != machine.get(k)}
    assert changed <= {"shared_encoding", "bm25", "_config_path", "datasets"}
    assert machine["shared_encoding"]["batch_size"] == research["shared_encoding"]["batch_size"]
    for key in ("k1", "b", "method", "implementation", "version"):
        assert machine["bm25"][key] == research["bm25"][key]
    assert "extends" not in machine


def test_extends_cycle_is_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("schema_version: 1\nextends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("schema_version: 1\nextends: a.yaml\n")
    with pytest.raises(ValueError, match="extends itself"):
        load_config(tmp_path / "a.yaml")
