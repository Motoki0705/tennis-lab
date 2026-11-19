"""Copy only the frozen source/checkpoint snapshots needed by this campaign."""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from protocol import DRIVE, INPUTS, ROOT, write

from src.utils.checksum import dual_sha256

SOURCE = DRIVE / "outputs/ball_detection/train/capacity_scaling"


def copy_file(source, destination, expected):
    if destination.exists():
        if (
            destination.stat().st_size == expected["size_bytes"]
            and dual_sha256(destination) == expected["sha256"]
        ):
            return
        raise RuntimeError(
            "Existing input is corrupt; no automatic overwrite: " + str(destination)
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".fetching")
    shutil.copyfile(source, partial)
    if (
        partial.stat().st_size != expected["size_bytes"]
        or dual_sha256(partial) != expected["sha256"]
    ):
        raise RuntimeError("Input SHA-256/size differs: " + str(source))
    partial.replace(destination)


def copy_snapshot(source, destination, expected_manifest=None):
    marker = source / "_SNAPSHOT.json"
    if expected_manifest is not None and dual_sha256(marker) != expected_manifest:
        raise RuntimeError("Snapshot marker differs")
    raw = marker.read_bytes()
    manifest = json.loads(raw)
    if manifest["schema"] != "verified_artifact_snapshot.v1":
        raise RuntimeError("Unknown input snapshot schema")
    for name, expected in manifest["files"].items():
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError("Invalid snapshot path")
        copy_file(source / rel, destination / rel, expected)
    (destination / "_SNAPSHOT.json").write_bytes(raw)
    return manifest


def main():
    INPUTS.mkdir(parents=True, exist_ok=True)
    source = SOURCE / "s42-20261110-12k"
    receipt = json.loads((source / "inputs-receipt.json").read_text())
    copy_snapshot(
        source / "inputs/inputs-v1", INPUTS / "capacity12k", receipt["manifest_sha256"]
    )
    for name in ("common-dpt-T32", "query-d512-l6-T64"):
        parent = json.loads(
            (INPUTS / "capacity12k/parent-receipts" / f"{name}.json").read_text()
        )
        copy_snapshot(
            SOURCE / "s42-20261103-main" / name / parent["receipt"]["generation"],
            INPUTS / name,
            parent["receipt"]["manifest_sha256"],
        )
    summary = {}
    for name in ("plan.json", "theory.json"):
        source = SOURCE / "s42-20261103-main" / name
        expected = dict(size_bytes=source.stat().st_size, sha256=dual_sha256(source))
        copy_file(source, INPUTS / name, expected)
        summary[name] = expected
    write(
        ROOT / "restored-inputs.json",
        dict(source_snapshots_verified=True, metadata=summary),
    )


if __name__ == "__main__":
    main()
