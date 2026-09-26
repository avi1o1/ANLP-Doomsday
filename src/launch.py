"""Run manifest tasks inside one existing allocation, without submitting more jobs."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from src.artifacts import atomic_json, jsonl
from src.experiments import write_manifest


def allocated_devices(count):
    if not 0 <= count <= 4:
        raise ValueError("Request between zero and four allocated GPUs")
    if count == 0:
        return []
    import torch

    available = torch.cuda.device_count()
    if available < count:
        raise ValueError(f"Requested {count} GPUs, but only {available} CUDA devices are visible")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible is not None else [str(i) for i in range(available)]
    return devices[:count]


def run_stage(stage, config_path, directory, devices, workers, cpus, stop):
    """Bound concurrency, isolate CUDA devices and propagate checkpoint requests."""
    jobs = list(jsonl(stage["tasks"]))
    gpu = stage["gpus"]
    # Only evaluation tasks have independent writable artifact directories.
    # Keep shared-corpus indexing and other CPU stages serial.
    slots = (min(workers, len(devices) // gpu, cpus, len(jobs)) if gpu else
             min(workers, cpus, len(jobs)) if stage["name"] == "evaluate" else 1)
    if not jobs:
        return 0
    if slots < 1:
        raise ValueError(f"Stage {stage['name']} requires {gpu} GPUs per task")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    active, records = {}, []
    cursor, failure, interrupted = 0, 0, False
    try:
        while cursor < len(jobs) or active:
            interrupted = interrupted or stop.requested
            if interrupted or failure:
                for process, _, record in active.values():
                    if not record.get("stop_sent") and process.poll() is None:
                        process.send_signal(signal.SIGUSR1)
                        record["stop_sent"] = True
            else:
                for slot in range(slots):
                    if slot in active or cursor == len(jobs):
                        continue
                    environment = os.environ.copy()
                    # The saved task config already includes the parent's resolved paths.
                    # Do not let inherited defaults undo explicit CLI overrides.
                    environment.pop("OUTPUT_ROOT", None)
                    environment["CSX_EVALUATION_SLOTS"] = str(slots)
                    selected = devices[slot * gpu:(slot + 1) * gpu] if gpu else []
                    environment["CUDA_VISIBLE_DEVICES"] = ",".join(selected)
                    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                        environment[name] = str(max(1, cpus // slots))
                    command = [sys.executable, "-m", "src", "--config", str(config_path)] + jobs[cursor]["argv"]
                    log_path = directory / f"{cursor:05d}.log"
                    handle = log_path.open("w")
                    record = {"task": cursor, "argv": jobs[cursor]["argv"], "devices": selected,
                              "log": str(log_path), "status": "running",
                              "worker_slots": slots, "threads_per_worker": max(1, cpus // slots)}
                    try:
                        process = subprocess.Popen(command, env=environment, stdout=handle, stderr=subprocess.STDOUT)
                    except BaseException:
                        handle.close()
                        raise
                    active[slot] = (process, handle, record)
                    records.append(record)
                    print(f"{stage['name']} task {cursor + 1}/{len(jobs)}: {log_path}", flush=True)
                    cursor += 1
            for slot, (process, handle, record) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                record.update(exit_code=code, status="complete" if code == 0 else "interrupted" if code == 99 else "failed")
                if code:
                    failure = failure or (code if code > 0 else 1)
                    print(f"{stage['name']} task {record['task']} exited {code}; see {record['log']}", flush=True)
                del active[slot]
            atomic_json(directory / "status.json", {"tasks": records, "unstarted": len(jobs) - cursor})
            if (interrupted or failure) and not active:
                break
            if active:
                time.sleep(0.2)
    finally:
        for process, handle, _ in active.values():
            if process.poll() is None:
                process.terminate()
            process.wait()
            handle.close()
    return 99 if interrupted else failure


def launch(config, family, gpus, workers, cpus, stop, stages=None):
    if workers < 1 or cpus < 1:
        raise ValueError("Worker and CPU counts must be positive")
    devices = allocated_devices(gpus)
    job = os.environ.get("SLURM_JOB_ID", "local") + "-" + uuid.uuid4().hex[:8]
    directory = Path(config["output_root"]).resolve() / "jobs" / family / job
    manifest = write_manifest(config, directory, family)
    available = {stage["name"] for stage in manifest["stages"]}
    selected = set(stages or available)
    unknown = selected - available
    if unknown:
        raise ValueError(f"Unknown stages for {family}: {', '.join(sorted(unknown))}")
    manifest["stages"] = [stage for stage in manifest["stages"] if stage["name"] in selected]
    incompatible = [stage["name"] for stage in manifest["stages"] if stage["gpus"] > gpus]
    if incompatible:
        raise ValueError(f"Stages require more than {gpus} allocated GPUs: {', '.join(incompatible)}")
    records, failure = [], 0
    print(f"Run manifest and task logs: {directory}", flush=True)
    for stage in manifest["stages"]:
        if stage["name"] == "evaluate" and os.environ.get("CSX_EVALUATION_BACKEND") == "cuda":
            if gpus < 1:
                raise ValueError("CUDA evaluation requires at least one allocated GPU")
            stage["gpus"] = 1
        if stop.requested or failure == 99:
            break
        if failure and stage["name"] != "report":
            records.append({"stage": stage["name"], "status": "skipped"})
            continue
        code = run_stage(stage, manifest["config"], directory / stage["name"], devices, workers, cpus, stop)
        records.append({"stage": stage["name"], "exit_code": code})
        failure = failure or code
        atomic_json(directory / "status.json", {"stages": records, "exit_code": failure})
    code = 99 if stop.requested else failure
    atomic_json(directory / "status.json", {"stages": records, "exit_code": code})
    return code
