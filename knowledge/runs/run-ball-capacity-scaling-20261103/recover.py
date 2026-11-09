"""Restore verified #1675 input/stage snapshots without overwriting local work."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from src.utils.artifact_store import RcloneSnapshotStore
from src.utils.checksum import dual_sha256

spec = importlib.util.spec_from_file_location(
    "sequential_recovery",
    Path("knowledge/runs/run-ball-sequential-length-20261025/recover.py"),
)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
PREPARED = Path(
    "outputs/ball_detection/precompute/sequential_length/observed-packs-20261025"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace-root", type=Path, required=True)
    p.add_argument("--drive-root", type=Path, required=True)
    p.add_argument("--rclone-config", type=Path, required=True)
    p.add_argument("--run-id", default="s42-20261103-main")
    a = p.parse_args()
    if Path(a.run_id).name != a.run_id or a.run_id in {".", ".."}:
        raise ValueError("Unsafe run ID")
    relative = Path("outputs/ball_detection/train/capacity_scaling") / a.run_id
    remote = a.drive_root / relative
    local = a.workspace_root / relative
    receipt = json.loads((remote / "inputs-receipt.json").read_text())
    if (
        receipt.get("schema") != "verified_artifact_snapshot.v1"
        or receipt.get("status") != "verified"
        or receipt.get("generation") != "inputs-v1"
    ):
        raise ValueError("A verified immutable input snapshot is required")
    if receipt["remote"] != f"mounted_drive:{relative.as_posix()}/inputs/inputs-v1":
        raise ValueError("Input receipt points to another campaign")
    scratch = local.parent / (".restore-inputs-" + uuid.uuid4().hex)
    RcloneSnapshotStore(
        scratch,
        remote="mounted_drive",
        remote_root=relative.as_posix() + "/inputs",
        config_path=a.rclone_config,
    ).restore("inputs-v1", manifest_sha256=receipt["manifest_sha256"])
    try:
        plan = json.loads((scratch / "plan.json").read_text())
        if (
            plan.get("schema") != "ball_capacity_campaign.v1"
            or len(plan["stages"]) != 8
        ):
            raise ValueError("Unknown capacity plan")
        for name in (
            "plan.json",
            "benchmark.json",
            "download.json",
            "source.bundle",
            "theory.json",
        ):
            prior.copy_exact(scratch / name, local / name)
        prior.copy_exact(remote / "inputs-receipt.json", local / "inputs-receipt.json")
        for source in (scratch / "prepared").iterdir():
            if source.is_file():
                prior.copy_exact(source, a.workspace_root / PREPARED / source.name)
    finally:
        shutil.rmtree(scratch)
    stages = []
    for entry in plan["stages"]:
        name = entry["name"]
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Unsafe stage name")
        receipts = remote / "persistence" / name
        if not receipts.exists():
            if (remote / name).exists():
                raise ValueError("Stage exists without committed receipts: " + name)
            continue
        # Initial-generation-only stages have no resumable optimizer state.
        if not any(prior.GENERATION.fullmatch(p.stem) for p in receipts.glob("*.json")):
            raise ValueError(
                "Stage has initialization but no durable update; explicit inspection is required before restarting: "
                + name
            )
        source, saved = prior.select_snapshot(receipts)
        root = (relative / name).as_posix()
        if saved["remote"] != f"mounted_drive:{root}/{saved['generation']}":
            raise ValueError("Stage receipt belongs to another campaign")
        destination = local / name
        if destination.exists():
            prior.verify_existing(
                destination,
                remote / name / saved["generation"] / "_SNAPSHOT.json",
                saved,
            )
        else:
            RcloneSnapshotStore(
                destination,
                remote="mounted_drive",
                remote_root=root,
                config_path=a.rclone_config,
            ).restore(saved["generation"], manifest_sha256=saved["manifest_sha256"])
        prior.copy_exact(source, local / "persistence" / name / source.name)
        latest = json.loads((destination / "latest.json").read_text())
        if dual_sha256(destination / latest["checkpoint"]) != latest["sha256"]:
            raise ValueError("Restored checkpoint digest changed")
        stages.append(
            dict(stage=name, update=latest["update"], generation=saved["generation"])
        )
    for name in ("COMPLETED.json", "ANALYSIS_COMPLETED.json"):
        if (remote / name).is_file():
            prior.copy_exact(remote / name, local / name)
    print(
        json.dumps(dict(restored=True, stages=stages, training_started=False)),
        flush=True,
    )


if __name__ == "__main__":
    main()
