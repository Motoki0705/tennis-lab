import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from src.utils.checksum import dual_sha256
from src.utils.models.state import state_dict_sha256
from tests.support.tasks.ball_detection.learning import (
    full_model_arguments,
    full_model_checkpoint,
    training_manifests,
)


def invoke(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "src.tasks.ball_detection.scripts.train", "--stage", "coordinates",
         "--initialize", "full_model", *arguments],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"),
        text=True, capture_output=True, timeout=180,
    )


def test_full_model_fresh_run_fixed_validation_and_exact_resume(tmp_path: Path) -> None:
    parent = full_model_checkpoint(tmp_path / "parent")
    train, validation = training_manifests(tmp_path)
    parent_hash, val_hash = dual_sha256(parent), dual_sha256(validation)
    initial = torch.load(parent, weights_only=True)
    output = tmp_path / "continuous"
    result = invoke(full_model_arguments(parent, train, validation, output))
    assert result.returncode == 0, result.stdout + result.stderr
    first = torch.load(output / "epoch-000.pt", weights_only=True)
    last = torch.load(output / "epoch-001.pt", weights_only=True)
    assert first["epoch"] == 0 and first["global_step"] == 1
    assert last["epoch"] == 1 and last["global_step"] == 2
    assert all(int(state["step"]) == 1 for state in first["optimizer"]["state"].values())
    assert first["best_epoch"] == 0 and first["best_score"] >= 0
    recipe = first["recipe"]
    assert recipe["parent"]["mode"] == "full_model"
    assert recipe["parent"]["sha256"] == parent_hash
    assert recipe["initial_state_sha256"] == state_dict_sha256(initial["state_dict"])
    assert recipe["manifest_sha256"] == dual_sha256(train) != initial["recipe"]["manifest_sha256"]
    assert recipe["validation_manifest_sha256"] == val_hash
    assert recipe["optimizer_initialization"] == "fresh"
    assert recipe["schedule_initialization"] == "step_zero"
    assert (output / "validation_manifest.json").read_bytes() == validation.read_bytes()
    encoder = [name for name in initial["state_dict"] if name.startswith("encoder.")]
    head = [name for name in initial["state_dict"] if not name.startswith("encoder.")]
    assert all(torch.equal(initial["state_dict"][name], first["state_dict"][name]) for name in encoder)
    assert any(not torch.equal(initial["state_dict"][name], first["state_dict"][name]) for name in head)
    assert any(not torch.equal(first["state_dict"][name], last["state_dict"][name]) for name in encoder)
    assert dual_sha256(parent) == parent_hash and dual_sha256(validation) == val_hash

    resumed = tmp_path / "resumed"
    shutil.copytree(output, resumed)
    for name in ("epoch-001.pt", "COMPLETED.json"):
        (resumed / name).unlink()
    for name in ("train.jsonl", "metrics.jsonl"):
        rows = [json.loads(line) for line in (resumed / name).read_text().splitlines()]
        (resumed / name).write_text("".join(json.dumps(row) + "\n" for row in rows if row["epoch"] == 0))

    # A new train/validation snapshot or different adopted parent starts a new
    # run. None may silently reuse the interrupted optimizer and schedule.
    changed_train, changed_val = tmp_path / "changed-train.json", tmp_path / "changed-val.json"
    changed_train.write_bytes(train.read_bytes() + b"\n")
    changed_val.write_bytes(validation.read_bytes() + b"\n")
    other_parent = tmp_path / "other-parent.pt"
    other_parent.write_bytes(parent.read_bytes())
    for source, manifest, fixed in ((parent, changed_train, validation),
                                    (parent, train, changed_val), (other_parent, train, validation)):
        args = full_model_arguments(source, manifest, fixed, resumed) + ["--resume", str(resumed / "epoch-000.pt")]
        rejected = invoke(args)
        assert rejected.returncode != 0 and "identical posttraining recipe" in rejected.stderr
        assert not (resumed / "epoch-001.pt").exists()

    result = invoke(full_model_arguments(parent, train, validation, resumed) + ["--resume", str(resumed / "epoch-000.pt")])
    assert result.returncode == 0, result.stdout + result.stderr
    replayed = torch.load(resumed / "epoch-001.pt", weights_only=True)
    assert state_dict_sha256(replayed["state_dict"]) == state_dict_sha256(last["state_dict"])
    assert replayed["validation"] == last["validation"]
    assert replayed["global_step"] == last["global_step"]
    assert json.loads((resumed / "COMPLETED.json").read_text()) == json.loads((output / "COMPLETED.json").read_text())
