"""Publish a self-contained source bundle and the frozen control/input provenance."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from control import HERE, REFERENCE, ROOT, WORKSPACE, read, snapshot_store, store, write


def main():
    stage = ROOT / "input-staging"
    if stage.exists():
        raise ValueError("Use the recorded receipt instead of republishing inputs")
    stage.mkdir(parents=True)
    for name in (
        "plan.json",
        "initialization-audit.json",
        "pretraining-audit.json",
        "scope.json",
    ):
        shutil.copyfile(ROOT / name, stage / name)
    shutil.copytree(
        HERE,
        stage / "reproduction",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copyfile(Path("uv.lock"), stage / "uv.lock")
    subprocess.run(
        ["git", "bundle", "create", str(stage / "source.bundle"), "HEAD"], check=True
    )
    subprocess.run(
        ["git", "bundle", "verify", str(stage / "source.bundle")], check=True
    )
    receipts = stage / "parent-receipts"
    receipts.mkdir()
    sources = [
        (
            "convnext-dpt",
            WORKSPACE
            / "outputs/ball_detection/train/sequential_length/s42-20261025-main/persistence/common-dpt-T32/completed-00010000-15c0438f.json",
        ),
        ("x3d-dpt", Path("/tmp/ball-12k-inputs/parent-receipts/common-dpt-T32.json")),
        (
            "eight-layer-4000",
            WORKSPACE
            / "outputs/ball_detection/train/capacity_scaling/s42-20261110-12k/persistence/query-d512-l8-T64-q06000-q12000/checkpoint-00004000-a7798c57.json",
        ),
    ]
    for name, path in sources:
        value = read(path)
        if name == "x3d-dpt":
            value = value["receipt"]
        write(receipts / f"{name}.json", value)
    from control import DRIVE, select_receipt

    for frames in (32, 64):
        _, value = select_receipt(
            DRIVE
            / REFERENCE.relative_to(WORKSPACE)
            / "persistence"
            / f"query-T{frames}"
        )
        write(receipts / f"baseline-query-T{frames}.json", value)
    for name in ("common-dpt-T32", "query-T32", "query-T64"):
        p = REFERENCE / name / "run.json"
        target = stage / "reference-recipes" / name / "run.json"
        target.parent.mkdir(parents=True)
        shutil.copyfile(p, target)
    receipt = snapshot_store(stage, ROOT.relative_to(WORKSPACE) / "inputs").publish(
        "inputs-v1"
    )
    write(ROOT / "inputs-receipt.json", receipt)
    store().publish_file(ROOT / "inputs-receipt.json")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
