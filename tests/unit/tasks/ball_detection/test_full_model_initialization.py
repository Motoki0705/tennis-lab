import json
import sys
from pathlib import Path

import pytest
import torch

from src.tasks.ball_detection.checkpoints.initialization import initialize_full_model
from src.tasks.ball_detection.training.coordinates import full_model, joint
from src.utils.checksum import dual_sha256
from src.utils.models.state import state_dict_sha256
from tests.support.tasks.ball_detection.learning import (
    full_model_arguments,
    full_model_checkpoint,
    training_manifests,
)


def test_all_adopted_weights_and_input_contract_survive_initialization(tmp_path: Path) -> None:
    path = full_model_checkpoint(tmp_path / "parent")
    saved = torch.load(path, weights_only=True)
    digest = dual_sha256(path)
    model, provenance = initialize_full_model(path, expected_sha256=digest, jpeg_decoder="opencv")
    assert state_dict_sha256(model.state_dict()) == state_dict_sha256(saved["state_dict"])
    assert model.mdd.input_contract() == saved["recipe"]["input_contract"]
    assert provenance["sha256"] == digest
    assert provenance["mode"] == "full_model"
    assert provenance["optimizer_initialization"] == "fresh"
    assert provenance["schedule_initialization"] == "step_zero"
    assert provenance["parent_manifest_sha256"] == "0" * 64
    assert dual_sha256(path) == digest


@pytest.mark.parametrize("fault,match", [
    ("hash", "hash changed"), ("schema", "query posttraining"),
    ("decoder", "decoder must match"), ("nan", "nonfinite"),
    ("dtype", "changed tensor"), ("missing", "Missing key"),
])
def test_invalid_initial_checkpoint_fails_explicitly(tmp_path: Path, fault: str, match: str) -> None:
    path = full_model_checkpoint(tmp_path / "parent")
    saved = torch.load(path, weights_only=True)
    key = next(iter(saved["state_dict"]))
    if fault == "schema":
        saved["schema"] = "mdd_coordinates.v4"
    elif fault == "nan":
        saved["state_dict"][key].fill_(float("nan"))
    elif fault == "dtype":
        saved["state_dict"][key] = saved["state_dict"][key].double()
    elif fault == "missing":
        del saved["state_dict"][key]
    torch.save(saved, path)
    with pytest.raises((ValueError, RuntimeError), match=match):
        initialize_full_model(path, expected_sha256="f" * 64 if fault == "hash" else dual_sha256(path),
                              jpeg_decoder="nvjpeg" if fault == "decoder" else "opencv")


@pytest.mark.parametrize("fault,match", [
    ("output", "new output directory"), ("source", "overlap fixed validation"),
    ("clip", "overlap fixed validation"), ("common", "Common selection subset"),
])
def test_bad_training_inputs_fail_before_model_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str, match: str,
) -> None:
    path = full_model_checkpoint(tmp_path / "parent")
    train, validation = training_manifests(tmp_path)
    output = tmp_path / "output"
    val = json.loads(validation.read_text())
    if fault == "output":
        output.mkdir()
    elif fault == "source":
        val["clips"][0]["clip"]["group_id"] = "train"
    elif fault == "clip":
        val["clips"][0]["clip"]["clip_id"] = "train/clip"
    else:
        val["clips"][0]["common_evaluation"] = False
    validation.write_text(json.dumps(val))

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Initialization must not occur after invalid input")

    monkeypatch.setattr(full_model, "initialize_full_model", forbidden)
    monkeypatch.setattr(joint, "FrozenBallWindowDataset", forbidden)
    monkeypatch.setattr(sys, "argv", ["full_model", *full_model_arguments(path, train, validation, output)])
    with pytest.raises((ValueError, FileExistsError), match=match):
        full_model.main()


def test_parent_is_not_a_same_run_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = full_model_checkpoint(tmp_path / "parent")
    train, validation = training_manifests(tmp_path)
    monkeypatch.setattr(sys, "argv", ["full_model", *full_model_arguments(path, train, validation, tmp_path / "new"),
                                    "--resume", str(path)])
    with pytest.raises(SystemExit, match="2"):
        full_model.arguments()
