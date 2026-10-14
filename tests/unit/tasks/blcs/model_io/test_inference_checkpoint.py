"""Inference needs the recorded model and strict weights, not training objectives."""

from pathlib import Path
from typing import Any

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.base.generate_dataset import resolve_court_keypoint_contract
from src.tasks.base.model_io import write_model_artifact_court_keypoint_contract
from src.tasks.blcs.inference.predictor import BLCSPredictor
from src.tasks.blcs.model_io.checkpoints import load_checkpoint_model
from src.tasks.blcs.model_io.factory import compose_blcs_model_io
from src.utils.paths import PROJECT_ROOT
from src.utils.schema.court_normalization import court_coordinate_normalization_metadata
from tests.support.tasks.blcs_physics import physics_sample, prepared_samples


@pytest.fixture
def checkpoint_payload() -> dict[str, Any]:
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "src/tasks/blcs/configs"), version_base="1.3"
    ):
        config = compose(config_name="train")
    config.model.hidden_dim = 32
    config.model.num_heads = 4
    config.model.rope_dim = 8
    config.model.ffn_dim = 64
    config.model.num_layers = 1
    config.model.camera_layers_per_stage = [1]
    config.model.time_layers_per_stage = [1]
    config.model.time_global_stage_mask = [False]
    config.model.predict_velocity = False
    binding = compose_blcs_model_io(config)
    # Historical objective metadata remains historical; inference cannot require
    # fields added by the new training loss or metric schema.
    config.loss = {"position_weight": 1.0, "reprojection_weight": 0.1}
    config.metrics = {"position_threshold": 0.3}
    payload: dict[str, Any] = {
        "hyper_parameters": {"config": OmegaConf.to_container(config, resolve=True)},
        "state_dict": {f"model.{k}": v for k, v in binding.model.state_dict().items()},
        "court_coordinate_normalization": court_coordinate_normalization_metadata(),
    }
    write_model_artifact_court_keypoint_contract(
        payload, resolve_court_keypoint_contract("physical_v1")
    )
    return payload


def test_inference_preserves_outputs_with_historical_training_metadata(
    tmp_path: Path, checkpoint_payload: dict[str, Any]
) -> None:
    path = tmp_path / "model.ckpt"
    torch.save(checkpoint_payload, path)
    expected = compose_blcs_model_io(checkpoint_payload["hyper_parameters"]["config"])
    expected.model.load_state_dict(
        {
            k.removeprefix("model."): v
            for k, v in checkpoint_payload["state_dict"].items()
        },
        strict=True,
    )
    restored, runtime = load_checkpoint_model(path)
    before = BLCSPredictor(expected, torch.device("cpu"))
    after = BLCSPredictor(restored, torch.device("cpu"))
    batch, _ = prepared_samples([physics_sample(8)])
    expected_prediction = before.predict_batch(batch, denormalize=False)
    actual = after.predict_batch(batch, denormalize=False)
    torch.testing.assert_close(
        actual.position, expected_prediction.position, rtol=0, atol=0
    )
    assert actual.velocity is None
    assert runtime.config["loss"] == {
        "position_weight": 1.0,
        "reprojection_weight": 0.1,
    }


@pytest.mark.parametrize("corruption", ["missing", "extra", "shape", "unknown_prefix"])
def test_model_weights_are_strict(
    tmp_path: Path, checkpoint_payload: dict[str, Any], corruption: str
) -> None:
    state = checkpoint_payload["state_dict"]
    key = next(iter(state))
    if corruption == "missing":
        state.pop(key)
    elif corruption == "extra":
        state["model.unexpected_parameter"] = torch.zeros(1)
    elif corruption == "shape":
        state[key] = torch.empty(0)
    else:
        state["unrecognized.parameter"] = torch.zeros(1)
    path = tmp_path / "invalid.ckpt"
    torch.save(checkpoint_payload, path)
    with pytest.raises((RuntimeError, ValueError)):
        load_checkpoint_model(path)


def test_unmigrated_model_config_is_explicitly_rejected(
    tmp_path: Path, checkpoint_payload: dict[str, Any]
) -> None:
    del checkpoint_payload["hyper_parameters"]["config"]["model"]["physics_head"]
    path = tmp_path / "unmigrated.ckpt"
    torch.save(checkpoint_payload, path)
    with pytest.raises(ValueError, match="physics_head"):
        load_checkpoint_model(path)
