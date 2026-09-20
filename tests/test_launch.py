import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.artifacts import read_json, write_jsonl
from src.config import load_config
from src.launch import allocated_devices, launch, run_stage

WORKER = """
import json, os, signal, sys, time
def checkpoint(*_):
    print('checkpoint saved', flush=True)
    sys.exit(99)
signal.signal(signal.SIGUSR1, checkpoint)
print(json.dumps({'devices': os.environ['CUDA_VISIBLE_DEVICES'],
                  'threads': os.environ['OMP_NUM_THREADS'], 'start': time.monotonic()}), flush=True)
time.sleep(float(sys.argv[1]))
print(json.dumps({'end': time.monotonic()}), flush=True)
sys.exit(int(sys.argv[2]))
"""


def worker_tasks(tmp_path, monkeypatch, *, gpu=1, delay=0.05, codes=(0,) * 5):
    original = subprocess.Popen

    def spawn(command, **kwargs):
        return original([sys.executable, "-c", WORKER] + command[-2:], **kwargs)

    monkeypatch.setattr("src.launch.subprocess.Popen", spawn)
    path = tmp_path / "tasks.jsonl"
    write_jsonl(path, [{"argv": [str(delay), str(code)]} for code in codes])
    return {"name": "worker-test", "tasks": path, "gpus": gpu}


@pytest.mark.parametrize("gpu", [0, 1, 2])
def test_allocation_isolation_and_cpu_sharing(tmp_path, monkeypatch, gpu):
    stage = worker_tasks(tmp_path, monkeypatch, gpu=gpu)
    devices = ["GPU-a", "GPU-b", "GPU-c", "GPU-d"]
    assert run_stage(stage, "unused", tmp_path / "logs", devices, 4, 36, SimpleNamespace(requested=False)) == 0
    records = read_json(tmp_path / "logs/status.json")["tasks"]
    assert len(records) == 5 and all(r["status"] == "complete" for r in records)
    measurements = [list(map(json.loads, Path(r["log"]).read_text().splitlines())) for r in records]
    slots = 4 // gpu if gpu else 1
    for i, (start, end) in enumerate(measurements):
        assert start["devices"] == ",".join(records[i]["devices"])
        assert int(start["threads"]) == 36 // slots
        assert len(records[i]["devices"]) == gpu
        for j, (other_start, other_end) in enumerate(measurements[:i]):
            if start["start"] < other_end["end"] and other_start["start"] < end["end"]:
                assert gpu and not set(records[i]["devices"]) & set(records[j]["devices"])


def test_failure_stops_new_tasks_and_preserves_exit_status(tmp_path, monkeypatch):
    stage = worker_tasks(tmp_path, monkeypatch, codes=(7,) + (0,) * 8)
    code = run_stage(stage, "unused", tmp_path / "logs", ["0", "1"], 2, 4, SimpleNamespace(requested=False))
    assert code == 7
    status = read_json(tmp_path / "logs/status.json")
    assert status["unstarted"] >= 7
    assert status["tasks"][0]["status"] == "failed"


def test_timeout_signal_reaches_active_workers(tmp_path, monkeypatch):
    stage = worker_tasks(tmp_path, monkeypatch, delay=10)
    stop = SimpleNamespace(requested=False)
    timer = threading.Timer(0.5, setattr, args=(stop, "requested", True))
    timer.start()
    try:
        assert run_stage(stage, "unused", tmp_path / "logs", ["0", "1"], 2, 4, stop) == 99
    finally:
        timer.cancel()
        timer.join()
    status = read_json(tmp_path / "logs/status.json")
    assert status["unstarted"] == 3
    assert all(r["exit_code"] == 99 and "checkpoint saved" in Path(r["log"]).read_text() for r in status["tasks"])


def test_launch_reports_failure_without_running_dependents(tmp_path, monkeypatch):
    config = load_config("configs/fixture.yaml")
    config["output_root"] = str(tmp_path)
    monkeypatch.setattr("src.launch.allocated_devices", lambda _: ["0"])
    executed = []

    def stage_runner(stage, *_):
        executed.append(stage["name"])
        return 8 if stage["name"] == "encode-sample" else 0

    monkeypatch.setattr("src.launch.run_stage", stage_runner)
    assert launch(config, "primary", 1, 1, 1, SimpleNamespace(requested=False)) == 8
    assert executed[-1] == "report" and "fit-cpu" not in executed
    status = read_json(next((tmp_path / "jobs/primary").glob("*/status.json")))
    assert status["exit_code"] == 8
    assert any(r.get("status") == "skipped" for r in status["stages"])


def test_cuda_visibility_validation(monkeypatch):
    assert allocated_devices(0) == []
    monkeypatch.setattr("torch.cuda.device_count", lambda: 4)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c,GPU-d")
    assert allocated_devices(2) == ["GPU-a", "GPU-b"]
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.raises(ValueError, match="only 1 CUDA"):
        allocated_devices(4)


def test_launch_can_select_cpu_only_stages(tmp_path, monkeypatch):
    config = load_config("configs/fixture.yaml")
    config["output_root"] = str(tmp_path)
    executed = []
    monkeypatch.setattr("src.launch.run_stage", lambda stage, *_: executed.append(stage["name"]) or 0)
    assert launch(config, "setting-a", 0, 1, 2, SimpleNamespace(requested=False),
                  ["prepare", "baselines-cpu"]) == 0
    assert executed == ["prepare", "baselines-cpu"]
    with pytest.raises(ValueError, match="require more than 0"):
        launch(config, "setting-a", 0, 1, 2, SimpleNamespace(requested=False), ["encode-sample"])


@pytest.mark.parametrize("script", ["train-cpu.sbatch", "train-gpu.sbatch", "submit-setting-a.sh"])
def test_split_slurm_scripts_have_valid_shell(script):
    subprocess.run(["bash", "-n", script], check=True)


def batch_fixture(tmp_path, training_exit=0, sync_exit=0, checkpoint=False):
    script = Path("train.sbatch").resolve()
    project = tmp_path / "project with spaces"
    binary = project / ".venv/bin"
    binary.mkdir(parents=True)
    (binary / "activate").write_text('export PATH="$PWD/.venv/bin:$PATH"\n')
    programs = {
        "python": """import json, os, signal, sys
from pathlib import Path
Path('invocation.json').write_text(json.dumps(dict(argv=sys.argv[1:], cwd=os.getcwd(),
    ir=os.environ['IR_DATASETS_HOME'], hf=os.environ['HF_HOME'])))
if os.environ['TEST_CHECKPOINT'] == '1':
    def stop(*_):
        Path('checkpoint-complete').touch()
        sys.exit(99)
    signal.signal(signal.SIGUSR1, stop)
    Path('ready').touch()
    signal.pause()
sys.exit(int(os.environ['TEST_TRAIN_EXIT']))
""",
        "rsync": """import json, os, sys
from pathlib import Path
if os.environ['TEST_CHECKPOINT'] == '1' and not Path('checkpoint-complete').exists():
    sys.exit(88)
Path('rsync.json').write_text(json.dumps(sys.argv[1:]))
sys.exit(int(os.environ['TEST_SYNC_EXIT']))
""",
        # Record mkdir without touching the actual /scratch or /ssd_scratch mountpoints.
        "mkdir": """import json, sys
from pathlib import Path
Path('mkdir.json').write_text(json.dumps(sys.argv[1:]))
""",
    }
    for name, body in programs.items():
        executable = binary / name
        executable.write_text(f"#!{sys.executable}\n" + body)
        executable.chmod(0o755)
    environment = {**os.environ, "SLURM_JOB_ID": "123", "SLURM_SUBMIT_DIR": str(project),
                   "CONFIG": "configs/frozen config.yaml", "NGPU": "4", "WORKERS": "2",
                   "SLURM_CPUS_PER_TASK": "36", "TEST_TRAIN_EXIT": str(training_exit),
                   "TEST_SYNC_EXIT": str(sync_exit), "TEST_CHECKPOINT": str(int(checkpoint))}
    for name in ("PROJECT_DIR", "VENV_DIR", "IR_DATASETS_HOME", "HF_HOME", "OUTPUT_ROOT"):
        environment.pop(name, None)
    return script, project, environment


def test_sbatch_default_storage_and_archive_destination(tmp_path):
    script, project, environment = batch_fixture(tmp_path)
    subprocess.run(["bash", "-n", str(script)], check=True)
    subprocess.run(["bash", str(script), "routing"], env=environment, capture_output=True, text=True, check=True)
    actual = read_json(project / "invocation.json")
    user = environment["USER"]
    assert read_json(project / "mkdir.json") == ["-p", f"/ssd_scratch/{user}",
        f"/ssd_scratch/{user}/ir-datasets", f"/ssd_scratch/{user}/huggingface", f"/ssd_scratch/{user}/output"]
    assert actual["cwd"] == str(project)
    assert actual["ir"] == f"/ssd_scratch/{user}/ir-datasets"
    assert actual["hf"] == f"/ssd_scratch/{user}/huggingface"
    assert actual["argv"] == ["-m", "src", "--config", "configs/frozen config.yaml",
                              "--output-root", f"/ssd_scratch/{user}/output", "launch", "--family", "routing",
                              "--gpus", "4", "--workers", "2", "--cpus", "36"]
    sync = read_json(project / "rsync.json")
    assert sync == ["-a", "--partial", "-e", "ssh -o BatchMode=yes", "--", f"/ssd_scratch/{user}/output", f"ada:/share1/{user}/"]


@pytest.mark.parametrize("training_exit,sync_exit,expected", [(0, 0, 0), (7, 0, 7), (0, 23, 23), (7, 23, 7)])
def test_sbatch_overrides_and_exit_status(tmp_path, training_exit, sync_exit, expected):
    script, project, environment = batch_fixture(tmp_path, training_exit, sync_exit)
    environment.update(IR_DATASETS_HOME=str(tmp_path / "custom ir"), HF_HOME=str(tmp_path / "custom hf"),
                       OUTPUT_ROOT=str(tmp_path / "custom output"))
    result = subprocess.run(["bash", str(script)], env=environment, capture_output=True, text=True)
    assert result.returncode == expected, result.stderr
    actual = read_json(project / "invocation.json")
    assert actual["ir"] == environment["IR_DATASETS_HOME"] and actual["hf"] == environment["HF_HOME"]
    sync = read_json(project / "rsync.json")
    assert sync[-2] == environment["OUTPUT_ROOT"]
    if sync_exit:
        assert "Output sync failed" in result.stderr


def test_sbatch_checkpoint_signal_finishes_before_sync(tmp_path):
    script, project, environment = batch_fixture(tmp_path, checkpoint=True)
    process = subprocess.Popen(["bash", str(script)], env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while not (project / "ready").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (project / "ready").exists()
        process.send_signal(signal.SIGUSR1)
        _, stderr = process.communicate(timeout=5)
        assert process.returncode == 99, stderr
        assert (project / "checkpoint-complete").exists() and (project / "rsync.json").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
