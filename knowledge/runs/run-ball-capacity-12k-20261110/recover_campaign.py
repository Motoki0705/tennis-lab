"""Restore committed stage snapshots with their original paths, without training."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from campaign import DRIVE, QUEUE, RELATIVE, ROOT, now, read, verify_plan_sources, write

from src.utils.artifact_store import RcloneSnapshotStore
from src.utils.checksum import dual_sha256


def main():
    pid_file = QUEUE / "worker.pid"
    if pid_file.exists():
        pid = pid_file.read_text().strip()
        if pid.isdecimal() and Path(f"/proc/{pid}").exists():
            raise ValueError(
                "Stop and verify the old service/process group before recovery"
            )
    plan = read(ROOT / "plan.json")
    verify_plan_sources(plan)
    spec = importlib.util.spec_from_file_location(
        "prior_snapshot_selection",
        Path("knowledge/runs/run-ball-sequential-length-20261025/recover.py"),
    )
    recovery = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recovery)
    restored = []
    for job in plan["jobs"]:
        for stage in job["stages"]:
            directory = DRIVE / RELATIVE / "persistence" / stage["name"]
            if not directory.exists():
                continue
            # Initialization-only, ambiguous or broken newest receipts fail explicitly.
            receipt_path, receipt = recovery.select_snapshot(directory)
            expected_remote = (
                f"mounted_drive:{RELATIVE}/{stage['name']}/{receipt['generation']}"
            )
            if receipt["remote"] != expected_remote:
                raise ValueError("Stage receipt points to another campaign")
            output = Path(stage["output"])
            if not output.exists():
                RcloneSnapshotStore(
                    output,
                    remote="mounted_drive",
                    remote_root=f"{RELATIVE}/{stage['name']}",
                    config_path=Path(plan["rclone_config"]),
                ).restore(
                    receipt["generation"], manifest_sha256=receipt["manifest_sha256"]
                )
            else:
                # The local active directory may contain uncommitted tail updates.
                # Its latest committed checkpoint/log prefix is checked by strict resume.
                marker = (
                    DRIVE
                    / RELATIVE
                    / stage["name"]
                    / receipt["generation"]
                    / "_SNAPSHOT.json"
                )
                if dual_sha256(marker) != receipt["manifest_sha256"]:
                    raise ValueError("Newest durable marker changed")
                latest = read(output / "latest.json")
                if latest["update"] != int(receipt["generation"].split("-")[1]):
                    raise ValueError(
                        "Local and Drive latest updates differ; inspect before recovery"
                    )
                if dual_sha256(output / latest["checkpoint"]) != latest["sha256"]:
                    raise ValueError("Local committed checkpoint changed")
            recovery.copy_exact(
                receipt_path, ROOT / "persistence" / stage["name"] / receipt_path.name
            )
            latest = read(output / "latest.json")
            if dual_sha256(output / latest["checkpoint"]) != latest["sha256"]:
                raise ValueError("Restored checkpoint hash differs")
            restored.append(
                dict(stage=stage["name"], update=latest["update"], receipt=receipt)
            )
    write(
        ROOT / "recovery-verified.json",
        dict(
            schema="ball_capacity_recovery.v1",
            complete=True,
            verified_at=now(),
            restored=restored,
            training_started=False,
        ),
    )


if __name__ == "__main__":
    main()
