"""Pin immutable partial outputs and reuse foundation-typed predictions for media fixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def pin(source, destination):
    if destination.exists():
        raise ValueError("Do not replace a prior partial-run receipt")
    inventory = {"schema": "dense_partial_inventory.v1", "source": str(source.resolve()),
                 "files": {str(p.relative_to(source)): {"sha256": sha(p), "size_bytes": p.stat().st_size}
                           for p in sorted(source.rglob("*")) if p.is_file()}}
    destination.write_text(json.dumps(inventory, indent=2) + "\n")
    print(sha(destination))


def reuse(source, receipt_path, expected_sha256, destination):
    if sha(receipt_path) != expected_sha256:
        raise ValueError("Partial-run receipt checksum mismatch")
    receipt = json.loads(receipt_path.read_text())
    if receipt["schema"] != "dense_partial_inventory.v1" or receipt["source"] != str(source.resolve()):
        raise ValueError("Partial-run source identity mismatch")
    actual = {str(p.relative_to(source)) for p in source.rglob("*") if p.is_file()}
    if actual != set(receipt["files"]):
        raise ValueError("Partial run has changed its inventory")
    for relative, spec in receipt["files"].items():
        path = (source / relative).resolve()
        if not path.is_relative_to(source.resolve()) or path.stat().st_size != spec["size_bytes"] or sha(path) != spec["sha256"]:
            raise ValueError(f"Partial output changed: {relative}")
    for name in ("results", "tuning"):
        shutil.copytree(source / name, destination / name)
    for name in ("selection.json", "tuning-progress.json"):
        shutil.copyfile(source / name, destination / name)
    return {"source": str(source.resolve()), "inventory_sha256": expected_sha256, "verified_files": len(actual),
            "policy": "reuse only frozen selection and typed prediction arrays; rescore and regenerate all media"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    pin(args.source, args.destination)
