"""Restore only the eight-layer prerequisite and started #1899 stages; never12/24 layers."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from control import (
    CAPACITY,
    DRIVE,
    OLD_ROOT,
    OLD_STAGE,
    ROOT,
    WORKSPACE,
    read,
    select_receipt,
    sha,
    snapshot_store,
    write,
)


def restore_stage(relative):
    source = DRIVE / relative
    path, receipt = select_receipt(source.parent / "persistence" / source.name)
    destination = WORKSPACE / relative
    if destination.exists():
        raise ValueError(
            f"Existing stage must be inspected before restoration: {destination}"
        )
    result = snapshot_store(destination, relative).restore(
        receipt["generation"], manifest_sha256=receipt["manifest_sha256"]
    )
    latest = read(destination / "latest.json")
    if sha(destination / latest["checkpoint"]) != latest["sha256"]:
        raise ValueError("Restored checkpoint differs")
    local = destination.parent / "persistence" / destination.name / path.name
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, local)
    return dict(stage=source.name, latest=latest, receipt=receipt, restore=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-prerequisite", action="store_true")
    args = parser.parse_args()
    import torch

    if torch.cuda.is_available() and torch.cuda.memory_allocated():
        raise ValueError("Stop the owning trainer before recovery")
    records = []
    if args.include_prerequisite:
        records.append(restore_stage(CAPACITY / OLD_STAGE))
        for name in ("d512-l6", "d512-l8"):
            src = DRIVE / CAPACITY / "job-status" / f"{name}.json"
            if src.exists():
                dst = OLD_ROOT / "job-status" / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
    plan = read(ROOT / "plan.json")
    for job in plan["jobs"]:
        for stage in job["stages"]:
            receipts = (
                DRIVE / ROOT.relative_to(WORKSPACE) / "persistence" / stage["name"]
            )
            if not receipts.exists():
                continue
            try:
                select_receipt(receipts)
            except ValueError:
                # Initialization-only snapshots cannot be called a full-state resume.
                if any(receipts.glob("*.json")):
                    raise
                continue
            records.append(restore_stage(ROOT.relative_to(WORKSPACE) / stage["name"]))
        src = (
            DRIVE / ROOT.relative_to(WORKSPACE) / "job-status" / f"{job['model']}.json"
        )
        if src.exists():
            dst = ROOT / "job-status" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    result = dict(complete=True, restored=records, held_models=[12, 24])
    write(ROOT / "recovery.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
