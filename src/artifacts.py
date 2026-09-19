"""Atomic artifacts, explicit provenance, locking, and cooperative interruption."""

from __future__ import annotations

import contextlib
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterator

from src.config import digest


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb" if path.suffix == ".gz" else "w") as handle:
            if path.suffix == ".gz":
                payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
                handle.write(gzip.compress(payload.encode("utf8"), mtime=0))
            else:
                json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: str | Path):
    path = Path(path)
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as handle:
        return json.load(handle)


def jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, records) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def provenance() -> dict:
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = "unavailable"
    package = Path(__file__).resolve().parent
    source = {str(p.relative_to(package)): file_hash(p) for p in sorted(package.rglob("*.py"))}
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "torch", "transformers", "ir_datasets", "ir-measures",
                    "faiss-cpu", "huggingface-hub", "accelerate", "datasets"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[package] = importlib.metadata.version(package)
    return {"code_revision": revision, "source_hash": digest(source), "versions": versions}


class StopRequested(KeyboardInterrupt):
    pass


class StopFlag:
    """SIGTERM/SIGUSR1 request a checkpoint at the next safe boundary."""

    def __init__(self):
        self.requested = False
        self._old = {}

    def __enter__(self):
        for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
            self._old[sig] = signal.signal(sig, self._request)
        return self

    def _request(self, *_):
        self.requested = True

    def check(self):
        if self.requested:
            raise StopRequested("Interruption requested; completed shards/checkpoints are retained")

    def __exit__(self, *_):
        for sig, handler in self._old.items():
            signal.signal(sig, handler)


class Artifact:
    """A directory with a locked, content-addressed manifest and explicit status.

    Incomplete work is reused only with the same scientific configuration and
    parents. Completed outputs are verified before being reused.
    """

    def __init__(self, path: str | Path, config: dict, parents: list[str | Path] = ()):
        self.path = Path(path)
        self.config = config
        self.parent_records = []
        for parent in parents:
            manifest = read_json(Path(parent) / "manifest.json")
            if manifest["status"] != "complete":
                raise ValueError(f"Parent is incomplete: {parent}")
            self.parent_records.append({"path": str(parent), "fingerprint": manifest["fingerprint"],
                                        "outputs_hash": digest(manifest["outputs"])})
        self.provenance = provenance()
        self.fingerprint = digest({"config": config, "parents": self.parent_records,
                                   "source_hash": self.provenance["source_hash"],
                                   "versions": self.provenance["versions"]})
        self.reused = False
        self.outputs = []
        self.metadata = {}

    def __enter__(self):
        self.path.mkdir(parents=True, exist_ok=True)
        self._lock = (self.path / ".lock").open("w")
        fcntl.flock(self._lock, fcntl.LOCK_EX)
        self.started = time.time()
        manifest_path = self.path / "manifest.json"
        if manifest_path.exists():
            old = read_json(manifest_path)
            if old["fingerprint"] != self.fingerprint:
                self._unlock()
                raise ValueError(f"Artifact identity changed at {self.path}; use a new output_root")
            if old["status"] == "complete":
                for output in old["outputs"]:
                    target = self.path / output["path"]
                    if not target.exists() or file_hash(target) != output["sha256"]:
                        self._unlock()
                        raise ValueError(f"Missing or corrupted completed artifact: {target}")
                self.reused = True
                self.metadata = old.get("metadata", {})
                return self
        self._save("running")
        return self

    def complete(self, outputs: list[str], **metadata):
        self.outputs = [{"path": p, "sha256": file_hash(self.path / p),
                         "bytes": (self.path / p).stat().st_size} for p in outputs]
        self.metadata = metadata
        self._save("complete")

    def _save(self, status, error=None):
        atomic_json(self.path / "manifest.json", {
            "schema_version": 1, "fingerprint": self.fingerprint,
            "config_hash": digest(self.config),
            "status": status, "config": self.config, "parents": self.parent_records,
            "provenance": self.provenance, "outputs": self.outputs,
            "metadata": self.metadata, "elapsed_seconds": time.time() - self.started,
            "error": error,
        })

    def _unlock(self):
        fcntl.flock(self._lock, fcntl.LOCK_UN)
        self._lock.close()

    def __exit__(self, kind, value, traceback):
        try:
            if kind is not None and not self.reused:
                status = "interrupted" if issubclass(kind, KeyboardInterrupt) else "failed"
                self._save(status, str(value))
            elif not self.reused and read_json(self.path / "manifest.json")["status"] != "complete":
                self._save("incomplete", "Stage exited without a completion record")
        finally:
            self._unlock()
