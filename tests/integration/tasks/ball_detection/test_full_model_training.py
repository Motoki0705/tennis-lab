"""Full-model initialization and fixed validation through a tiny real restart."""

import json
from pathlib import Path

import pytest
import torch

from src.tasks.ball_detection.checkpoints.provenance import data_identity
from src.tasks.ball_detection.evaluation import coordinate_cli
from src.tasks.ball_detection.training.coordinates import full_model
from src.utils.checksum import dual_sha256
from src.utils.models.state import state_dict_sha256
from tests.integration.tasks.ball_detection.mdd_training_contracts import (
    assert_replay,
    invoke_main,
)
from tests.support.tasks.ball_detection.learning import (
    full_model_arguments,
    full_model_checkpoint,
    training_manifests,
)

pytestmark = pytest.mark.integration


def invoke(arguments: list[str]) -> None:
    invoke_main(full_model.main, [*arguments, "--cpu-threads", "1"])


def test_full_model_fresh_run_fixed_validation_and_exact_resume(tmp_path: Path) -> None:
    parent = full_model_checkpoint(tmp_path / "parent")
    train, validation = training_manifests(tmp_path)
    parent_hash, val_hash = dual_sha256(parent), dual_sha256(validation)
    initial = torch.load(parent, weights_only=True)
    output = tmp_path / "continuous"
    invoke(full_model_arguments(parent, train, validation, output))
    first = torch.load(output / "epoch-000.pt", weights_only=True)
    last = torch.load(output / "epoch-001.pt", weights_only=True)
    assert first["epoch"] == 0 and first["global_step"] == 1
    assert last["epoch"] == 1 and last["global_step"] == 2
    assert all(int(state["step"]) == 1 for state in first["optimizer"]["state"].values())
    assert first["best_epoch"] == 0 and first["best_score"] >= 0
    recipe = first["recipe"]
    assert recipe["runtime"]["input_verification"] == "lazy"
    assert recipe["parent"]["mode"] == "full_model"
    assert recipe["parent"]["sha256"] == parent_hash
    assert recipe["parent"]["ancestry"]["schema"] == initial["schema"]
    assert recipe["parent"]["ancestry"]["global_step"] == initial["global_step"]
    assert recipe["data_identity"] == data_identity(train)
    assert recipe["initialization"] == dict(mode="full_model", seed=42)
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

    report_path = tmp_path / "evaluation.json"
    evaluate = [
        "--checkpoint", str(output / "epoch-001.pt"), "--output", str(report_path),
        "--artifact-root", str(tmp_path), "--output-root", str(tmp_path),
        "--split", "val", "--device", "cpu", "--cpu-threads", "1",
    ]
    with pytest.raises(ValueError, match="frozen data/split identity"):
        invoke_main(coordinate_cli.main, [*evaluate, "--manifest", str(train)])
    assert not report_path.exists()
    invoke_main(coordinate_cli.main, [*evaluate, "--manifest", str(validation)])
    report = json.loads(report_path.read_text())
    assert report["manifest_sha256"] == val_hash
    assert report["split"] == "val" and report["scopes"] == last["validation"]["scopes"]

    resumed = tmp_path / "resumed"
    stopped_arguments = full_model_arguments(parent, train, validation, resumed)
    invoke([*stopped_arguments, "--stop-after-epoch", "0"])
    checkpoint = resumed / "epoch-000.pt"
    diagnostic = json.loads((resumed / "DIAGNOSTIC_STOP.json").read_text())
    assert diagnostic == dict(epoch=0, global_step=1, complete=False, checkpoint=checkpoint.name,
                              sha256=dual_sha256(checkpoint))
    assert not (resumed / "epoch-001.pt").exists() and not (resumed / "COMPLETED.json").exists()
    resume_arguments = [*stopped_arguments, "--resume", str(checkpoint)]
    with pytest.raises(ValueError, match="precedes the first resumed epoch"):
        invoke([*resume_arguments, "--stop-after-epoch", "0"])
    assert not (resumed / "epoch-001.pt").exists()

    # A changed snapshot or adopted parent starts a new run, never this optimizer.
    changed_train, changed_val = tmp_path / "changed-train.json", tmp_path / "changed-val.json"
    changed_train.write_bytes(train.read_bytes() + b"\n")
    changed_val.write_bytes(validation.read_bytes() + b"\n")
    other_parent = tmp_path / "other-parent.pt"
    other_parent.write_bytes(parent.read_bytes())
    for source, manifest, fixed in ((parent, changed_train, validation),
                                    (parent, train, changed_val), (other_parent, train, validation)):
        args = full_model_arguments(source, manifest, fixed, resumed) + ["--resume", str(checkpoint)]
        with pytest.raises(ValueError, match="identical posttraining recipe"):
            invoke(args)
        assert not (resumed / "epoch-001.pt").exists()

    invoke(resume_arguments)
    assert_replay(output / "epoch-001.pt", resumed / "epoch-001.pt")
    assert json.loads((resumed / "COMPLETED.json").read_text()) == json.loads((output / "COMPLETED.json").read_text())
