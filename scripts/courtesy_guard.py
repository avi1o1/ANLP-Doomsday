"""Run experiment stages on a shared machine, stepping aside whenever someone else uses it.

The guard runs each planned launcher invocation as a child process. While another
user is logged in, owns a GPU process, or is using more CPU than a small threshold,
the child is asked to checkpoint with SIGUSR1, the same request Slurm sends before a
time limit. The launcher forwards it to every worker, each worker saves at its next
safe boundary and exits, and every GPU and CPU is released. Once nobody else has
been present for a grace period the invocation is started again on the same output
root: completed artifacts are reused and partial evaluations resume from their last
saved query batch.

Nothing here hides the work. Processes keep their ordinary names, the state file
records every pause, and anyone can ask the guard to yield by creating the pause file.

Example plan (JSON list, run in order):

    [{"name": "controls", "family": "controls", "stages": ["index", "evaluate"],
      "gpus": 4, "workers": 4, "cpus": 32, "env": {"CSX_EVALUATION_BACKEND": "cuda"}}]

An entry may give "argv" instead of family/stages to run any other resumable command.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import pwd
import signal
import subprocess
import sys
import time
from pathlib import Path

CHECKPOINT_EXIT = 99  # src.launch and every stage exit 99 after a cooperative stop


def log(handle, message):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    handle.write(line + "\n")
    handle.flush()


def owner(pid):
    try:
        return pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
    except (OSError, KeyError):
        return None


def logged_in_users():
    """Accounts with a login: utmp entries, plus every SSH session.

    A command run over SSH without a terminal leaves no utmp record, so sshd's
    per-session process titles ("sshd: alice@pts/0", "sshd-session: alice@notty") are read
    as well. Listener and pre-authentication titles are skipped.
    """
    users = set()
    try:
        output = subprocess.run(["who"], capture_output=True, text=True, timeout=10).stdout
        users |= {line.split()[0] for line in output.splitlines() if line.strip()}
    except (OSError, subprocess.TimeoutExpired):
        pass
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            title = Path(entry.path, "cmdline").read_bytes().split(b"\0")[0].decode(errors="ignore")
        except OSError:
            continue
        # OpenSSH 9.8 renamed the per-session process from sshd to sshd-session.
        for prefix in ("sshd: ", "sshd-session: "):
            if title.startswith(prefix) and "[" not in title and "@" in title:
                users.add(title[len(prefix):].split("@")[0].strip())
    return users


def gpu_users():
    try:
        output = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return set()
    users = set()
    for line in output.splitlines():
        if line.strip().isdigit():
            name = owner(int(line.strip()))
            # A process in another PID namespace has no /proc entry here; count it
            # as someone else, since it is not one of ours.
            users.add(name or "<unknown gpu process>")
    return users


class CpuSampler:
    """Per-user CPU use over the last poll interval, in cores, from /proc."""

    def __init__(self):
        self.previous = self._read()
        self.stamp = time.monotonic()

    @staticmethod
    def _read():
        ticks = {}
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            try:
                stat = Path(entry.path, "stat").read_text()
                uid = os.stat(entry.path).st_uid
            except OSError:
                continue
            fields = stat[stat.rindex(")") + 2:].split()
            ticks[int(entry.name)] = (uid, int(fields[11]) + int(fields[12]))
        return ticks

    def sample(self):
        current, now = self._read(), time.monotonic()
        elapsed = max(now - self.stamp, 1e-3) * os.sysconf("SC_CLK_TCK")
        cores = {}
        for pid, (uid, total) in current.items():
            before = self.previous.get(pid)
            used = total - before[1] if before and before[0] == uid else 0
            cores[uid] = cores.get(uid, 0.0) + max(used, 0) / elapsed
        self.previous, self.stamp = current, now
        return cores


def human_uids(ours):
    """Accounts that belong to people, excluding ours; system daemons are ignored."""
    minimum = 1000
    try:
        for line in Path("/etc/login.defs").read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == "UID_MIN":
                minimum = int(parts[1])
    except OSError:
        pass
    return lambda uid: uid >= minimum and uid != 65534 and _name(uid) not in ours


def _name(uid):
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def others_present(args, sampler, is_other):
    reasons = []
    if args.pause_file.exists():
        reasons.append(f"pause file {args.pause_file}")
    logins = sorted(u for u in logged_in_users() if u not in args.ours)
    if logins:
        reasons.append("logged in: " + ", ".join(logins))
    if not args.ignore_gpu:
        gpu = sorted(u for u in gpu_users() if u not in args.ours)
        if gpu:
            reasons.append("GPU in use by: " + ", ".join(gpu))
    busy = {uid: c for uid, c in sampler.sample().items() if is_other(uid) and c >= args.cpu_threshold}
    if busy:
        reasons.append("CPU in use by: " + ", ".join(f"{_name(u)} ({c:.1f} cores)" for u, c in busy.items()))
    return reasons


def command_for(entry, args):
    if "argv" in entry:
        return list(entry["argv"])
    command = [sys.executable, "-m", "src", "--config", str(args.config), "--output-root", str(args.output_root),
               "launch", "--family", entry["family"], "--gpus", str(entry.get("gpus", 0)),
               "--workers", str(entry.get("workers", 1)), "--cpus", str(entry.get("cpus", os.cpu_count() or 1))]
    if entry.get("stages"):
        command += ["--stages", *entry["stages"]]
    return command


def stop_child(process, args, handle):
    """Ask for a checkpoint, then escalate only if the child does not answer."""
    for sig, target, wait in ((signal.SIGUSR1, "launcher", args.stop_timeout),
                              (signal.SIGTERM, "process group", 180),
                              (signal.SIGKILL, "process group", 60)):
        if process.poll() is not None:
            break
        log(handle, f"sending {sig.name} to the {target}")
        try:
            if target == "launcher":
                process.send_signal(sig)
            else:
                os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            continue
    return process.wait()


def save_state(args, state):
    temporary = args.state_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    os.replace(temporary, args.state_file)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", type=Path, help="JSON list of launcher invocations, run in order")
    parser.add_argument("--config", type=Path, default=Path("configs/research.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("OUTPUT_ROOT", "output")))
    parser.add_argument("--ours", default=getpass.getuser(),
                        help="comma-separated accounts that do not trigger a pause (default: this user)")
    parser.add_argument("--poll", type=float, default=15, help="seconds between checks")
    parser.add_argument("--grace", type=float, default=600,
                        help="seconds with nobody else present before resuming")
    parser.add_argument("--cpu-threshold", type=float, default=1.0,
                        help="cores another user must be using to count as present")
    parser.add_argument("--stop-timeout", type=float, default=900,
                        help="seconds to wait for a checkpoint before escalating")
    parser.add_argument("--ignore-gpu", action="store_true", help="do not treat others' GPU processes as presence")
    parser.add_argument("--pause-file", type=Path, help="create this file to make the guard yield")
    parser.add_argument("--state-file", type=Path, help="progress and pause record")
    args = parser.parse_args(argv)
    args.ours = {name.strip() for name in args.ours.split(",") if name.strip()}
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.pause_file = args.pause_file or args.output_root / "PAUSE"
    args.state_file = args.state_file or args.output_root / "guard_state.json"
    plan = json.loads(args.plan.read_text())
    names = [entry.get("name", f"step{i}") for i, entry in enumerate(plan)]
    if len(set(names)) != len(names):
        raise SystemExit("Plan entries need unique names")

    state = json.loads(args.state_file.read_text()) if args.state_file.exists() else {}
    state.setdefault("completed", [])
    state.setdefault("pauses", [])
    is_other = human_uids(args.ours)
    sampler = CpuSampler()
    handle = (args.output_root / "guard.log").open("a")
    stop = {"requested": False}

    def request_exit(*_):
        stop["requested"] = True

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, request_exit)

    log(handle, f"guard started; ours={sorted(args.ours)}; plan={names}")
    # Grace applies only after a yield; step-to-step transitions start at once.
    process, running, clear_since, yielded = None, None, None, False
    try:
        while not stop["requested"]:
            pending = [e for e, n in zip(plan, names) if n not in state["completed"]]
            if not pending and process is None:
                log(handle, "plan complete")
                return 0
            reasons = others_present(args, sampler, is_other)
            if process is not None:
                code = process.poll()
                if code is not None:
                    log(handle, f"{running} exited {code}")
                    if code == 0:
                        state["completed"].append(running)
                        save_state(args, state)
                    elif code != CHECKPOINT_EXIT:
                        log(handle, f"{running} failed; stopping so the failure can be read")
                        return code
                    process = None
                elif reasons:
                    log(handle, f"yielding {running}: " + "; ".join(reasons))
                    started = time.time()
                    code = stop_child(process, args, handle)
                    state["pauses"].append({"step": running, "at": started, "reasons": reasons,
                                            "checkpoint_seconds": round(time.time() - started, 1),
                                            "exit_code": code})
                    save_state(args, state)
                    log(handle, f"{running} checkpointed with exit {code}")
                    if code not in (0, CHECKPOINT_EXIT):
                        log(handle, f"{running} did not checkpoint cleanly; it resumes from its last saved batch")
                    if code == 0:
                        state["completed"].append(running)
                        save_state(args, state)
                    process, clear_since, yielded = None, None, True
            pending = [e for e, n in zip(plan, names) if n not in state["completed"]]
            if process is None and pending:
                if reasons:
                    clear_since = None
                elif clear_since is None:
                    clear_since = time.monotonic()
                    if yielded:
                        log(handle, f"machine is clear; resuming after {args.grace:.0f}s without other users")
                if clear_since is not None and (not yielded or time.monotonic() - clear_since >= args.grace):
                    entry = pending[0]
                    running = names[plan.index(entry)]
                    environment = os.environ.copy()
                    environment.update({k: str(v) for k, v in entry.get("env", {}).items()})
                    command = command_for(entry, args)
                    step_log = (args.output_root / f"guard_{running}.log").open("a")
                    log(handle, f"starting {running}: {' '.join(command)}")
                    process = subprocess.Popen(command, env=environment, stdout=step_log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    step_log.close()
                    clear_since, yielded = None, False
            time.sleep(args.poll)
        log(handle, "guard asked to exit; checkpointing the running step")
        if process is not None:
            stop_child(process, args, handle)
        return CHECKPOINT_EXIT
    finally:
        handle.close()


if __name__ == "__main__":
    sys.exit(main())
