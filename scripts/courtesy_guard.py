"""Run experiment stages on a shared machine, stepping aside whenever someone else uses it.

The machine may be one account shared by many people, so presence is decided by
session and by process, not by account name. Ours is the guard's own process tree,
plus any session that connects from an IP listed as ours. Someone else is:

- an SSH connection from any other IP;
- a login at the machine itself (a console or desktop session);
- a GPU process outside our tree;
- more than a small amount of CPU used by processes outside our tree;
- the pause file, which anyone can create;
- less free disk than --min-free-gb, so the run never fills a shared disk.

Presence has to last --debounce seconds before it counts, so a quick scp or status
check does not cost a checkpoint. The pause file counts at once.

On presence the launcher gets SIGUSR1, the same checkpoint request Slurm sends. It
forwards it to every worker, each worker saves at its next safe boundary and exits
99, and every GPU and CPU is released. Once nobody else has been present for
--grace seconds the step starts again on the same output root: completed artifacts
are reused and partial evaluations resume from their last saved query batch.

Nothing here hides the work. Processes keep their ordinary names, the state file
records every pause, and the log says why each one happened.

Add your IP from inside your own SSH session with

    python3 scripts/courtesy_guard.py --register --output-root OUTPUT_ROOT

which appends it to OUTPUT_ROOT/our_ips; the file is re-read on every poll.

Example plan (JSON list, run in order):

    [{"name": "controls", "family": "controls", "stages": ["index", "evaluate"],
      "gpus": 1, "workers": 1, "cpus": 16, "env": {"CSX_EVALUATION_BACKEND": "cuda"}}]

An entry may give "argv" instead of family/stages to run any other resumable command.
"""

from __future__ import annotations

import argparse
import json
import os
import pwd
import shutil
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


def _uid_min():
    try:
        for line in Path("/etc/login.defs").read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == "UID_MIN":
                return int(parts[1])
    except OSError:
        pass
    return 1000


UID_MIN = _uid_min()


def _client_ip(address):
    """'10.1.38.157', '[::ffff:10.1.38.157]:22' or '10.1.38.157:22' -> '10.1.38.157'."""
    host = address.rsplit(":", 1)[0] if address.count(":") == 1 or address.startswith("[") else address
    host = host.strip("[]")
    return host[len("::ffff:"):] if host.startswith("::ffff:") else host


def process_table():
    """pid -> (ppid, uid, cpu ticks, SSH client IP or None), read from /proc."""
    table = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            stat = Path(entry.path, "stat").read_text()
            uid = os.stat(entry.path).st_uid
        except OSError:
            continue
        fields = stat[stat.rindex(")") + 2:].split()
        client = None
        try:
            # Readable for processes of this account, which is the shared case.
            for item in Path(entry.path, "environ").read_bytes().split(b"\0"):
                if item.startswith(b"SSH_CONNECTION="):
                    client = _client_ip(item.split(b"=", 1)[1].split()[0].decode())
                    break
        except OSError:
            pass
        table[int(entry.name)] = (int(fields[1]), uid, int(fields[11]) + int(fields[12]), client)
    return table


def our_processes(table, root_pid, our_ips, our_accounts):
    """Our tree, sessions from our IPs, and whole accounts listed as ours, with descendants."""
    children = {}
    for pid, (ppid, *_rest) in table.items():
        children.setdefault(ppid, []).append(pid)
    seeds = {root_pid} | {pid for pid, (_, uid, _t, ip) in table.items()
                          if (ip and ip in our_ips) or _name(uid) in our_accounts}
    ours, stack = set(), list(seeds)
    while stack:
        pid = stack.pop()
        if pid in ours:
            continue
        ours.add(pid)
        stack.extend(children.get(pid, ()))
    return ours


def ssh_peers():
    try:
        output = subprocess.run(["ss", "-Htn", "state", "established", "( sport = :22 )"],
                                capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [_client_ip(line.split()[-1]) for line in output.splitlines() if line.split()]


def console_logins():
    """Sessions at the machine itself: utmp entries with no remote host, or an X display."""
    try:
        output = subprocess.run(["who"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    found = []
    for line in output.splitlines():
        host = line[line.index("(") + 1:line.rindex(")")] if "(" in line and ")" in line else ""
        if not host or host.startswith(":"):
            found.append(" ".join(line.split()[:2]))
    return found


def gpu_pids():
    try:
        output = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [int(line) for line in (x.strip() for x in output.splitlines()) if line.isdigit()]


def _name(uid):
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


class Presence:
    """Decides whether someone else is using the machine, with a debounce."""

    def __init__(self, args):
        self.args = args
        self.previous = process_table()
        self.stamp = time.monotonic()
        self.since = None

    def our_ips(self):
        ips = set(self.args.our_ips)
        try:
            ips |= {line.strip() for line in self.args.ips_file.read_text().splitlines()
                    if line.strip() and not line.startswith("#")}
        except OSError:
            pass
        return ips

    def raw_reasons(self):
        table, now = process_table(), time.monotonic()
        ips = self.our_ips()
        ours = our_processes(table, os.getpid(), ips, self.args.ours)
        reasons = []
        others = sorted({ip for ip in ssh_peers() if ip not in ips})
        if others:
            reasons.append("SSH from " + ", ".join(others))
        consoles = console_logins()
        if consoles:
            reasons.append("console login: " + ", ".join(consoles))
        if not self.args.ignore_gpu:
            foreign = [pid for pid in gpu_pids() if pid not in ours]
            if foreign:
                reasons.append("GPU processes not ours: " + ", ".join(map(str, foreign)))
        # A sample taken moments after the last one turns a single clock tick into
        # several apparent cores; skip the CPU test until a real interval has passed.
        interval = now - self.stamp
        elapsed = max(interval, 1e-3) * os.sysconf("SC_CLK_TCK")
        cores = 0.0 if interval >= 0.5 else float("nan")
        for pid, (_, uid, ticks, _ip) in table.items():
            before = self.previous.get(pid)
            if pid in ours or uid < UID_MIN or uid == 65534 or not before or before[1] != uid:
                continue
            if interval >= 0.5:
                cores += max(ticks - before[2], 0) / elapsed
        if interval >= 0.5 and cores >= self.args.cpu_threshold:
            reasons.append(f"{cores:.1f} cores in use outside our processes")
        self.previous, self.stamp = table, now
        return reasons

    def reasons(self):
        """(lasting, current): presence past the debounce, and presence right now.

        The guard yields on lasting presence but will not start a step while anyone
        is present at all. The pause file and a nearly full disk count at once.
        """
        found = self.raw_reasons()
        if not found:
            self.since = None
        elif self.since is None:
            self.since = time.monotonic()
        lasting = found if found and time.monotonic() - self.since >= self.args.debounce else []
        immediate = []
        if self.args.pause_file.exists():
            immediate.append(f"pause file {self.args.pause_file}")
        free = shutil.disk_usage(self.args.output_root).free / 2**30
        if free < self.args.min_free_gb:
            immediate.append(f"only {free:.0f} GiB free on the output disk (floor {self.args.min_free_gb:.0f})")
        return immediate + lasting, immediate + found


def register(args):
    connection = os.environ.get("SSH_CONNECTION")
    if not connection:
        raise SystemExit("Run --register inside the SSH session you want treated as ours")
    ip = _client_ip(connection.split()[0])
    existing = args.ips_file.read_text().split() if args.ips_file.exists() else []
    if ip not in existing:
        with args.ips_file.open("a") as handle:
            handle.write(ip + "\n")
    print(f"{ip} is ours; recorded in {args.ips_file}")
    return 0


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
    parser.add_argument("plan", type=Path, nargs="?", help="JSON list of launcher invocations, run in order")
    parser.add_argument("--config", type=Path, default=Path("configs/research.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("OUTPUT_ROOT", "output")))
    parser.add_argument("--our-ips", default="",
                        help="comma-separated client IPs whose sessions are ours")
    parser.add_argument("--ours", default="",
                        help="comma-separated accounts used only by us; leave empty on a shared account")
    parser.add_argument("--register", action="store_true",
                        help="record this SSH session's client IP as ours, then exit")
    parser.add_argument("--debounce", type=float, default=60,
                        help="seconds presence must last before the guard yields")
    parser.add_argument("--poll", type=float, default=15, help="seconds between checks")
    parser.add_argument("--grace", type=float, default=600,
                        help="seconds with nobody else present before resuming")
    parser.add_argument("--cpu-threshold", type=float, default=1.0,
                        help="cores used outside our processes that count as someone present")
    parser.add_argument("--stop-timeout", type=float, default=900,
                        help="seconds to wait for a checkpoint before escalating")
    parser.add_argument("--ignore-gpu", action="store_true", help="do not treat foreign GPU processes as presence")
    parser.add_argument("--min-free-gb", type=float, default=40,
                        help="yield when the output disk has less free space than this, so it never fills")
    parser.add_argument("--pause-file", type=Path, help="create this file to make the guard yield")
    parser.add_argument("--state-file", type=Path, help="progress and pause record")
    args = parser.parse_args(argv)
    args.ours = {name.strip() for name in args.ours.split(",") if name.strip()}
    args.our_ips = {ip.strip() for ip in args.our_ips.split(",") if ip.strip()}
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.pause_file = args.pause_file or args.output_root / "PAUSE"
    args.state_file = args.state_file or args.output_root / "guard_state.json"
    args.ips_file = args.output_root / "our_ips"
    if args.register:
        return register(args)
    if args.plan is None:
        parser.error("a plan is required unless --register is given")
    plan = json.loads(args.plan.read_text())
    names = [entry.get("name", f"step{i}") for i, entry in enumerate(plan)]
    if len(set(names)) != len(names):
        raise SystemExit("Plan entries need unique names")

    state = json.loads(args.state_file.read_text()) if args.state_file.exists() else {}
    state.setdefault("completed", [])
    state.setdefault("pauses", [])
    presence = Presence(args)
    handle = (args.output_root / "guard.log").open("a")
    stop = {"requested": False}

    def request_exit(*_):
        stop["requested"] = True

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, request_exit)

    log(handle, f"guard started; our IPs={sorted(presence.our_ips())}; accounts={sorted(args.ours)}; plan={names}")
    # Grace applies only after a yield; step-to-step transitions start at once.
    process, running, clear_since, yielded = None, None, None, False
    try:
        while not stop["requested"]:
            pending = [e for e, n in zip(plan, names) if n not in state["completed"]]
            if not pending and process is None:
                log(handle, "plan complete")
                return 0
            reasons, present = presence.reasons()
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
                if present:
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
