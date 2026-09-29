"""Run an explicit JSONL list of CLI tasks as one launcher stage, in bounded parallel slots.

A family manifest enumerates every row of its family. On an output root where most
rows already exist, made by an older revision of the code, relaunching the family
would re-enter those finished artifacts under a different source hash and stop. This
runs exactly the tasks given, each a {"argv": [...]} line such as the launcher writes,
through src.launch.run_stage: device isolation, per-worker thread limits, checkpoint
signals (SIGUSR1 or SIGTERM, exit 99) and status files behave as in a launched stage.

The stage name must be one run_stage parallelises across CPU workers ("index",
"evaluate") or any name with --gpus-per-task 1 for GPU work.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.artifacts import StopFlag
from src.launch import allocated_devices, run_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tasks", type=Path, help="JSONL of {\"argv\": [...]} task lines")
    parser.add_argument("--config", required=True, help="configuration each task is run with")
    parser.add_argument("--stage", required=True, help="stage name, which decides CPU parallelism")
    parser.add_argument("--directory", required=True, type=Path, help="where task logs and status.json go")
    parser.add_argument("--gpus", type=int, default=0, help="GPUs allocated to this run")
    parser.add_argument("--gpus-per-task", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cpus", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    devices = allocated_devices(args.gpus, int(os.environ.get("CSX_TASKS_PER_GPU", "1")))
    stage = {"name": args.stage, "tasks": str(args.tasks.resolve()), "gpus": args.gpus_per_task}
    with StopFlag() as stop:
        return run_stage(stage, args.config, args.directory, devices, args.workers, args.cpus, stop)


if __name__ == "__main__":
    sys.exit(main())
