"""Fill the ir_datasets cache with ranged, parallel downloads, verified against ir_datasets' MD5s.

The BEIR server and the MS MARCO blob serve one connection at a few tens of kB/s
from IIIT, so ir_datasets' own single-stream download takes hours for MS MARCO.
Both honour byte ranges. Each file is fetched as 2 MiB ranges over many
connections, reassembled, and accepted only when its MD5 equals the value
ir_datasets 0.5.11 expects; it is then placed where ir_datasets looks for it, so
the ordinary prepare stage finds it and does not download again.

configs/downloads.json lists the files: url, cache target relative to
IR_DATASETS_HOME, size hint and expected MD5.
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

CHUNK = 2 * 1024**2


def valid(path, record):
    if not path.exists():
        return False
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "md5").hexdigest() == record["expected_md5"]


def curl(record, out, options=()):
    command = ["curl", "--http1.1", "-fsSL", "--retry", "6", "--retry-all-errors",
               "--connect-timeout", "20", "--max-time", "300"]
    for key, value in record.get("download_args", {}).get("headers", {}).items():
        command += ["-H", f"{key}: {value}"]
    subprocess.run(command + list(options) + ["-o", str(out), record["url"]], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=Path("configs/downloads.json"))
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("IR_DATASETS_HOME", "~/.ir_datasets")))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--only", nargs="*", help="dataset name prefixes to fetch (default: all)")
    args = parser.parse_args()
    root = args.root.expanduser()
    records = json.loads(args.manifest.read_text())
    if args.only:
        records = [r for r in records if any(r["dataset"].startswith(p) for p in args.only)]
    pending, jobs = [], []
    for record in records:
        target = root / record["target"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if valid(target, record):
            print("already verified", record["dataset"], flush=True)
            continue
        if target.exists():
            target.unlink()  # a partial or foreign file; it is fetched again and verified
        if record.get("size_hint", 0) < CHUNK:
            temporary = target.with_name(target.name + ".range-download")
            curl(record, temporary)
            if not valid(temporary, record):
                raise SystemExit("MD5 mismatch: " + record["dataset"])
            temporary.replace(target)
            print("verified", record["dataset"], flush=True)
            continue
        parts = target.with_name(target.name + ".parts")
        parts.mkdir(exist_ok=True)
        headers = parts / "headers"
        curl(record, parts / "probe", ["--range", "0-0", "--max-filesize", str(CHUNK), "-D", str(headers)])
        sizes = re.findall(r"(?im)^content-range:\s*bytes\s+0-0/(\d+)", headers.read_text())
        if not sizes:
            raise SystemExit("server does not support byte ranges: " + record["dataset"])
        size = int(sizes[-1])
        pending.append((record, target, parts, size))
        jobs.extend((record, parts, first, min(first + CHUNK, size) - 1) for first in range(0, size, CHUNK))
        print(f"{record['dataset']}: {size / 1e6:.1f} MB in {-(-size // CHUNK)} ranges", flush=True)

    def fetch(job):
        record, parts, first, last = job
        part = parts / f"{first // CHUNK:06d}"
        if part.exists() and part.stat().st_size == last - first + 1:
            return 0
        temporary = part.with_suffix(".tmp")
        curl(record, temporary, ["--range", f"{first}-{last}", "--max-filesize", str(last - first + 1)])
        if temporary.stat().st_size != last - first + 1:
            raise SystemExit("wrong range size: " + str(part))
        temporary.replace(part)
        return part.stat().st_size

    # SIGUSR1 or SIGTERM is a checkpoint request, as for every stage: finished ranges
    # stay on disk, so a later run fetches only what is missing.
    stop = {"requested": False}

    def request(*_):
        stop["requested"] = True

    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, request)
    start, done, total = time.monotonic(), 0, sum(last - first + 1 for *_x, first, last in jobs)
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, future in enumerate(cf.as_completed([pool.submit(fetch, job) for job in jobs]), 1):
            if stop["requested"]:
                pool.shutdown(wait=True, cancel_futures=True)
                print("checkpoint: finished ranges kept", flush=True)
                sys.exit(99)
            done += future.result()
            if n % 20 == 0 or n == len(jobs):
                rate = done / max(1e-3, time.monotonic() - start) / 1e6
                print(f"{done / 1e6:.0f} / {total / 1e6:.0f} MB, {rate:.2f} MB/s", flush=True)
    for record, target, parts, size in pending:
        temporary = target.with_name(target.name + ".range-download")
        with temporary.open("wb") as out:
            for first in range(0, size, CHUNK):
                out.write((parts / f"{first // CHUNK:06d}").read_bytes())
        if not valid(temporary, record):
            raise SystemExit("final MD5 mismatch: " + record["dataset"])
        temporary.replace(target)
        for part in parts.iterdir():
            part.unlink()
        parts.rmdir()
        print("verified", record["dataset"], size, flush=True)
    print("all files verified", flush=True)


if __name__ == "__main__":
    main()
