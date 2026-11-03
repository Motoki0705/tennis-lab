"""Restore this experiment's committed Drive generations after loss of a VM.

Run before ``run.py train --resume``. This command does not start training,
change dataset paths, synthesize benchmark evidence, or overwrite local work.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.utils.artifact_store import RcloneSnapshotStore
from src.utils.checksum import dual_sha256

CAMPAIGNS = Path("outputs/ball_detection/train/sequential_length")
PREPARED = Path(
    "outputs/ball_detection/precompute/sequential_length/observed-packs-20261025"
)
GENERATION = re.compile(r"(completed|checkpoint)-(\d{8})-[0-9a-f]{8}")


def copy_exact(source: Path, destination: Path) -> None:
    """Copy immutable metadata atomically, or verify an existing identical copy."""
    if source.is_symlink() or not source.is_file() or destination.is_symlink():
        raise ValueError(f"Expected regular metadata files: {source}, {destination}")
    expected = dual_sha256(source)
    if destination.exists():
        if dual_sha256(destination) != expected:
            raise ValueError(f"Existing local metadata differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copyfile(source, temporary)
        if dual_sha256(temporary) != expected or dual_sha256(source) != expected:
            raise ValueError(f"Metadata changed while copying: {source}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def select_snapshot(directory: Path) -> tuple[Path, dict]:
    """Select the highest committed update; completed wins ties.

    A broken newest receipt is an error, never a reason to roll back silently.
    Initialization-only generations do not contain resumable optimizer state.
    """
    candidates = []
    for path in directory.glob("*.json"):
        match = GENERATION.fullmatch(path.stem)
        if match:
            candidates.append((int(match[2]), match[1] == "completed", path))
    if not candidates:
        raise ValueError(f"No committed checkpoint receipt in {directory}")
    highest = max((update, completed) for update, completed, _ in candidates)
    selected = [p for u, c, p in candidates if (u, c) == highest]
    if len(selected) != 1:
        raise ValueError(f"Ambiguous checkpoint receipts in {directory}")
    path = selected[0]
    saved = json.loads(path.read_text())
    if (
        saved.get("schema") != "verified_artifact_snapshot.v1"
        or saved.get("status") != "verified"
        or saved.get("generation") != path.stem
        or not re.fullmatch(r"[0-9a-f]{64}", saved.get("manifest_sha256", ""))
    ):
        raise ValueError(f"Invalid committed checkpoint receipt: {path}")
    return path, saved


def verify_existing(directory: Path, marker: Path, receipt: dict) -> None:
    """Make a repeated restore safe without overwriting an advanced local run."""
    if dual_sha256(marker) != receipt["manifest_sha256"]:
        raise ValueError("Snapshot completion marker SHA-256 differs")
    manifest = json.loads(marker.read_text())
    if (
        manifest.get("schema") != "verified_artifact_snapshot.v1"
        or manifest.get("generation") != receipt["generation"]
    ):
        raise ValueError("Snapshot marker schema or generation differs")
    files = manifest["files"]
    paths = list(directory.rglob("*"))
    if directory.is_symlink() or any(path.is_symlink() for path in paths):
        raise ValueError("Local recovery tree contains a symlink")
    if {p.relative_to(directory).as_posix() for p in paths if p.is_file()} != set(
        files
    ):
        raise ValueError("Local stage differs from snapshot; use its own resume state")
    for name, identity in files.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe snapshot path")
        local = directory / path
        if (
            local.stat().st_size != identity["size_bytes"]
            or dual_sha256(local) != identity["sha256"]
        ):
            raise ValueError(f"Existing local stage differs: {name}")


def restore(args: argparse.Namespace) -> dict:
    campaign = CAMPAIGNS / args.run_id
    remote = args.drive_root / campaign
    local = args.workspace_root / campaign
    plan = json.loads((remote / "plan.json").read_text())
    if plan["schema"] != "ball_sequential_campaign.v1":
        raise ValueError("Unknown campaign plan schema")
    copy_exact(remote / "plan.json", local / "plan.json")
    for name in ("progress.json", "COMPLETED.json"):
        if (remote / name).is_file():
            copy_exact(remote / name, local / name)
    stages = []
    for name, _, _, _ in plan["stages"]:
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Unsafe stage name")
        receipts = remote / "persistence" / name
        if not receipts.exists():
            if (remote / name).exists():
                raise ValueError(f"Stage exists without a committed receipt: {name}")
            continue
        path, receipt = select_snapshot(receipts)
        remote_root = (campaign / name).as_posix()
        if receipt["remote"] != f"mounted_drive:{remote_root}/{receipt['generation']}":
            raise ValueError("Snapshot receipt points to another campaign")
        destination = local / name
        if destination.exists():
            verify_existing(
                destination,
                remote / name / receipt["generation"] / "_SNAPSHOT.json",
                receipt,
            )
            result = dict(status="verified", reused=True)
        else:
            result = RcloneSnapshotStore(
                destination,
                remote="mounted_drive",
                remote_root=remote_root,
                config_path=args.rclone_config,
            ).restore(receipt["generation"], manifest_sha256=receipt["manifest_sha256"])
        copy_exact(path, local / "persistence" / name / path.name)
        latest = json.loads((destination / "latest.json").read_text())
        if dual_sha256(destination / latest["checkpoint"]) != latest["sha256"]:
            raise ValueError(f"Restored latest checkpoint differs: {name}")
        stages.append(
            dict(stage=name, update=latest["update"], receipt=receipt, restore=result)
        )
        print(json.dumps(stages[-1]), flush=True)
    if not stages:
        raise ValueError("No resumable campaign stages were restored")
    for name in (
        "audit.json",
        "experiments.json",
        "train_T32.json",
        "train_T64.json",
        "evaluation_T32.json",
        "evaluation_T64.json",
    ):
        copy_exact(
            args.drive_root / PREPARED / name, args.workspace_root / PREPARED / name
        )
    # Saved recipes pin the original absolute paths and exact manifest bytes.
    # Rewriting their paths would change the training recipe and is rejected.
    for stage in stages:
        recipe = json.loads((local / stage["stage"] / "run.json").read_text())["recipe"]
        frames = recipe["model"]["frames"]
        for kind, key in (
            ("train", "manifest_sha256"),
            ("evaluation", "evaluation_manifest_sha256"),
        ):
            manifest = args.workspace_root / PREPARED / f"{kind}_T{frames}.json"
            if dual_sha256(manifest) != recipe[key]:
                raise ValueError(
                    f"Prepared manifest differs from saved recipe: {manifest}"
                )
        expected_store = (
            args.workspace_root
            / "data/ball_detection/ball-mix1800-720p-all-frame-train-val-v1"
        )
        if Path(recipe["data"]["snapshot"]["directory"]) != expected_store:
            raise ValueError(
                "Restore the dataset at its original recorded absolute path"
            )
    result = dict(
        schema="ball_sequential_recovery.v1",
        complete=True,
        stages=stages,
        original_plan=plan,
        benchmark_summary_restored=False,
        benchmark_evidence="Original plan retains the selected batch sizes and benchmark SHA-256",
    )
    output = local / "recovery.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--rclone-config", type=Path, required=True)
    parser.add_argument("--run-id", default="s42-20261025-main")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
        parser.error("run-id must be a single directory name")
    restore(args)
