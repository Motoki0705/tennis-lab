"""Read-only full-state qualification of a restored frozen campaign or snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import torch
import torchvision

from src.tasks.ball_detection.checkpoints.provenance import source_identity
from src.tasks.ball_detection.checkpoints.temporal import read_resume
from src.utils.artifact_store import RcloneSnapshotStore
from src.utils.checksum import dual_sha256

ROOT = Path("outputs/ball_detection/train/capacity_scaling/s42-20261103-main")
PREPARED = Path("outputs/ball_detection/precompute/sequential_length/observed-packs-20261025")


def inspect_stage(directory, entry, code):
    run = json.loads((directory / "run.json").read_text())
    latest = json.loads((directory / "latest.json").read_text())
    recipe = run["recipe"]
    saved = read_resume(directory / latest["checkpoint"], output=directory,
                        recipe=recipe, code=code)
    assert recipe["model"] == entry["model"]
    assert recipe["budget"] == entry["budget"]
    assert recipe["runtime"]["precision"] == "bf16"
    assert recipe["model"]["activation_checkpointing"] is False
    assert recipe["model"]["frame_chunk_size"] is None
    assert recipe["budget"]["seed"] == 42
    assert recipe["budget"]["input_frames"] == 128
    assert len(saved["optimizer"]["param_groups"]) == 2
    states = saved["optimizer"]["state"]
    assert states
    for state in states.values():
        assert {"step", "exp_avg", "exp_avg_sq"} <= state.keys()
        for tensor in state.values():
            if isinstance(tensor, torch.Tensor):
                assert torch.isfinite(tensor).all().item()
    assert {"python", "numpy", "torch", "cuda"} == saved["rng"].keys()
    assert saved["rng"]["torch"].dtype == torch.uint8
    assert len(saved["rng"]["cuda"]) == 1
    assert saved["rng"]["cuda"][0].dtype == torch.uint8
    identity = saved["train_log"]
    raw = (directory / "train.jsonl").read_bytes()
    prefix = raw[:identity["bytes"]]
    assert len(prefix) == identity["bytes"]
    assert hashlib.sha256(prefix).hexdigest() == identity["sha256"]
    rows = [json.loads(line) for line in prefix.splitlines()]
    assert [r["update"] for r in rows] == list(range(1, latest["update"] + 1))
    assert all(r["input_frames"] == 128 and math.isfinite(r["loss"])
               and math.isfinite(r["gradient_norm"]) for r in rows)
    assert rows[-1]["sample_position"] == saved["sample_position"]
    if recipe["phase"] == "finetune":
        assert recipe["initialization"]["update"] == 10000
        assert recipe["budget"]["schedule_updates"] == 12000
        assert recipe["budget"]["freeze_updates"] == 600
        assert recipe["budget"]["encoder_lr_ratio"] == 0.1
    return dict(stage=entry["name"], update=latest["update"],
                checkpoint=latest["checkpoint"], checkpoint_sha256=latest["sha256"],
                optimizer_parameter_states=len(states), rng_devices=1,
                sample_position=saved["sample_position"], train_log=identity,
                last_loss=rows[-1]["loss"], last_gradient_norm=rows[-1]["gradient_norm"],
                runtime=recipe["runtime"], validation_present=saved["validation"] is not None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=Path("/content/tennis-lab"))
    p.add_argument("--receipt", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--destination", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert sys.version_info[:2] == (3, 11)
    assert str(torch.__version__) == "2.14.1+cu130"
    assert str(torchvision.__version__) == "0.29.1+cu130"
    root = a.workspace / ROOT
    plan = json.loads((root / "plan.json").read_text())
    code = source_identity()
    assert code["directory"] == "/content/tennis-lab/.claude/worktrees/ball-capacity"
    assert code["source_sha256"] == plan["code"]["source_sha256"]
    assert code["torch_version"] == plan["code"]["torch_version"]
    checks = {
        "controller_sha256": Path(__file__).with_name("run.py"),
        "prior_recipe_sha256": Path("knowledge/runs/run-ball-sequential-length-20261025/run.py"),
        "theory_sha256": Path(__file__).with_name("theory.json"),
        "analysis_sha256": Path(__file__).with_name("report.py"),
        "benchmark_sha256": root / "benchmark.json",
    }
    for key, path in checks.items():
        assert dual_sha256(path) == plan[key], key
    for name in (root / "prepared").glob("*.json"):
        assert dual_sha256(name) == dual_sha256(a.workspace / PREPARED / name.name)
    download = json.loads((a.workspace / "outputs/ball_detection/precompute/sequential_length/drive-download-20261025/progress.json").read_text())
    assert download["complete"] and download["verified_files"] == download["expected_files"] == 2925
    assert download["verified_bytes"] == download["expected_bytes"] == 157028755547
    assert download["distribution_sha256"] == plan["download_distribution_sha256"]
    stages = []
    receipt = None
    if a.receipt:
        receipt = json.loads(a.receipt.read_text())
        assert receipt["schema"] == "verified_artifact_snapshot.v1"
        assert receipt["status"] == "verified"
        stage = a.receipt.parent.name
        entry = next(e for e in plan["stages"] if e["name"] == stage)
        expected = f"mounted_drive:{ROOT.as_posix()}/{stage}/{receipt['generation']}"
        assert receipt["remote"] == expected
        restored = RcloneSnapshotStore(
            a.destination, remote="mounted_drive", remote_root=(ROOT / stage).as_posix(),
            config_path=a.config, timeout_seconds=3600,
        ).restore(receipt["generation"], manifest_sha256=receipt["manifest_sha256"])
        stages.append(inspect_stage(a.destination, entry, code))
        assert restored["status"] == "verified"
    else:
        for entry in plan["stages"]:
            directory = root / entry["name"]
            if directory.exists():
                stages.append(inspect_stage(directory, entry, code))
    result = dict(schema="ball_capacity_cli_resume_verification.v1", verified=True,
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  source_files=len(code["source_sha256"]), frozen_digests=True,
                  torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                  dataset_verified=True, receipt=receipt, stages=stages)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
