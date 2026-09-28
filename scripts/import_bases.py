"""Import fitted bases from an earlier run into a new output root.

Bases are the expensive, stochastic part of Setting A. Reusing them makes a second
run's encodings match the first run's up to GPU rounding, so rows from both runs
describe the same representations. The import refuses anything it cannot check:

- every file must match the checksum in the basis's original manifest;
- the basis specification must equal this configuration's, key for key;
- it must have been fitted with the encoder revision this configuration pins;
- when the earlier sample record is given, the local fitting sample must select the
  same documents with the same encoder revision.

Each basis is written through the ordinary Artifact machinery with the local
fitting sample as its parent, so fit-bases later finds it complete and reuses it.
The new manifest records where it came from and the original fingerprint.

Run after encode-sample has completed in the new output root.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from src.artifacts import Artifact, file_hash, read_json
from src.config import load_config, named
from src.retrieval.pipeline import basis_path, root

NAME = re.compile(r"^(?P<name>.+)__(?P<granularity>pooled|token)__s(?P<seed>\d+)$")


def verify(directory, manifest):
    for output in manifest["outputs"]:
        path = directory / output["path"]
        if not path.exists() or file_hash(path) != output["sha256"]:
            raise SystemExit(f"Checksum mismatch in the source basis: {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="the earlier run's bases directory")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-sample", type=Path,
                        help="the earlier run's sample/sample.json, or its sample/manifest.json")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config["output_root"] = args.output_root
    sample = root(config) / "sample"
    if read_json(sample / "manifest.json")["status"] != "complete":
        raise SystemExit("Run encode-sample in the new output root first")
    local = read_json(sample / "sample.json")
    if args.source_sample:
        earlier = read_json(args.source_sample)
        # An archived run may keep only the sample's manifest, whose metadata holds
        # the same record that sample.json does.
        earlier = earlier.get("metadata", earlier) if "sample_ids_hash" not in earlier else earlier
        for key in ("sample_ids_hash", "model_revision"):
            if earlier[key] != local[key]:
                raise SystemExit(f"Fitting sample differs from the earlier run in {key}: "
                                 f"{earlier[key]} against {local[key]}")
        print(f"fitting sample matches: {local['sample_ids_hash'][:16]}, revision {local['model_revision']}")
    imported, skipped = [], []
    for directory in sorted(p for p in args.source.iterdir() if p.is_dir()):
        match = NAME.match(directory.name)
        manifest_path = directory / "manifest.json"
        if not match or not manifest_path.exists():
            skipped.append((directory.name, "not a basis directory"))
            continue
        manifest = read_json(manifest_path)
        if manifest["status"] != "complete":
            skipped.append((directory.name, f"status {manifest['status']}"))
            continue
        verify(directory, manifest)
        info = read_json(directory / "basis.json")
        name, granularity, seed = match["name"], match["granularity"], int(match["seed"])
        spec = named(config["bases"], name)
        if info["spec"] != spec:
            raise SystemExit(f"{directory.name}: specification differs from this configuration\n"
                             f"  earlier: {info['spec']}\n  here:    {spec}")
        revision = info["fit_metadata"].get("model_revision")
        if revision != local["model_revision"]:
            raise SystemExit(f"{directory.name}: fitted with encoder revision {revision}, "
                             f"but this sample used {local['model_revision']}")
        target = basis_path(config, name, granularity, seed)
        parents = [sample]
        if spec.get("dictionary_basis"):
            parents.append(basis_path(config, spec["dictionary_basis"], spec["dictionary_granularity"], seed))
        with Artifact(target, {"basis": spec, "granularity": granularity, "seed": seed}, parents) as work:
            if work.reused:
                skipped.append((directory.name, "already present"))
                continue
            for filename in ("basis.json", "weights.npz"):
                shutil.copy2(directory / filename, target / filename)
            work.complete(["basis.json", "weights.npz"], **info["fit_metadata"],
                          imported_from=str(directory.resolve()),
                          imported_fingerprint=manifest["fingerprint"])
        imported.append(directory.name)
    for name in imported:
        print(f"imported {name}")
    for name, reason in skipped:
        print(f"skipped  {name}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
