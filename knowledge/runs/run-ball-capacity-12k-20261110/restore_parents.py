"""Restore the eight immutable full-state inputs; never launch GPU work."""

from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.utils.artifact_store import RcloneSnapshotStore
from src.utils.checksum import dual_sha256

WORKSPACE = Path("/content/tennis-lab")
DRIVE = Path("/content/drive/MyDrive/tennis_lab")
RELATIVE = Path("outputs/ball_detection/train/capacity_scaling/s42-20261103-main")
NEW = WORKSPACE / RELATIVE.parent / "s42-20261110-12k"
INPUTS = Path("/tmp/ball-capacity-inputs-v1")
PREPARED = Path(
    "outputs/ball_detection/precompute/sequential_length/observed-packs-20261025"
)
CONFIG = Path("/tmp/ball-drive-mount.conf")


def recovery_module():
    spec = importlib.util.spec_from_file_location(
        "prior_recovery",
        Path("knowledge/runs/run-ball-sequential-length-20261025/recover.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def restore_one(entry):
    recovery = recovery_module()
    name = entry["name"]
    path, receipt = recovery.select_snapshot(DRIVE / RELATIVE / "persistence" / name)
    assert receipt["generation"].startswith("completed-"), name
    root = (RELATIVE / name).as_posix()
    assert receipt["remote"] == f"mounted_drive:{root}/{receipt['generation']}"
    destination = WORKSPACE / RELATIVE / name
    if destination.exists():
        recovery.verify_existing(
            destination,
            DRIVE / RELATIVE / name / receipt["generation"] / "_SNAPSHOT.json",
            receipt,
        )
        restored = dict(status="verified", reused=True)
    else:
        restored = RcloneSnapshotStore(
            destination, remote="mounted_drive", remote_root=root, config_path=CONFIG
        ).restore(receipt["generation"], manifest_sha256=receipt["manifest_sha256"])
    recovery.copy_exact(path, WORKSPACE / RELATIVE / "persistence" / name / path.name)
    latest = json.loads((destination / "latest.json").read_text())
    checkpoint = destination / latest["checkpoint"]
    assert latest["update"] == entry["budget"]["updates"]
    assert dual_sha256(checkpoint) == latest["sha256"]
    result = dict(
        name=name,
        checkpoint=str(checkpoint),
        sha256=latest["sha256"],
        update=latest["update"],
        receipt=receipt,
        restore=restored,
    )
    NEW.joinpath("parent-receipts").mkdir(exist_ok=True)
    NEW.joinpath("parent-receipts", name + ".json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)
    return result


def main():
    recovery = recovery_module()
    plan = json.loads((INPUTS / "plan.json").read_text())
    for name in (
        "plan.json",
        "benchmark.json",
        "download.json",
        "theory.json",
        "source.bundle",
    ):
        recovery.copy_exact(INPUTS / name, WORKSPACE / RELATIVE / name)
    for path in (INPUTS / "prepared").iterdir():
        recovery.copy_exact(path, WORKSPACE / PREPARED / path.name)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(restore_one, plan["stages"]))
    import torch

    for item, stage in zip(results, plan["stages"], strict=True):
        saved = torch.load(item["checkpoint"], map_location="cpu", weights_only=True)
        assert saved["schema"] == "ball_temporal_checkpoint.v2"
        assert saved["updates_completed"] == stage["budget"]["updates"]
        assert json.loads(json.dumps(saved["recipe"]["model"])) == stage["model"]
        assert saved["recipe"]["budget"] == stage["budget"]
        assert saved["code"]["source_sha256"] == plan["code"]["source_sha256"]
        assert saved["code"]["torch_version"] == str(torch.__version__)
        assert saved["validation"] is not None
        assert saved["optimizer"]["state"] and saved["rng"]["cuda"] is not None
        assert saved["sample_position"] == (
            10000 if stage["architecture"] == "dpt" else 16000
        )
        item["full_state_verified"] = True
        item["cumulative_updates"] = saved["model_artifact"]["training"][
            "cumulative_updates"
        ]
        item["sample_position"] = saved["sample_position"]
        item["optimizer_states"] = len(saved["optimizer"]["state"])
        del saved
    NEW.joinpath("parents-verified.json").write_text(
        json.dumps(
            dict(schema="ball_capacity_parents.v1", complete=True, parents=results),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
