"""Tests for fail-closed BLCS checkpoint composition metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.base.generate_dataset import (
    CourtKeypointContractMismatchError,
    MissingCourtKeypointMetadataError,
    resolve_court_keypoint_contract,
)
from src.tasks.base.model_io import write_model_artifact_court_keypoint_contract
from src.tasks.blcs.model_io.checkpoints import (
    load_checkpoint_config,
    load_checkpoint_runtime,
)
from src.utils.schema.court_normalization import (
    court_coordinate_normalization_metadata,
)


def _axial_model_config() -> dict[str, Any]:
    with initialize_config_dir(
        config_dir=str(Path("src/tasks/blcs/configs").resolve()), version_base="1.3"
    ):
        config = compose(config_name="train")
    return cast(dict[str, Any], OmegaConf.to_container(config.model, resolve=True))


def _contract() -> dict[str, object]:
    return {"court_coordinate_normalization": court_coordinate_normalization_metadata()}


def test_checkpoint_requires_explicit_composition_config(tmp_path: Path) -> None:
    checkpoint = tmp_path / "missing_config.ckpt"
    torch.save({"state_dict": {}, **_contract()}, checkpoint)

    with pytest.raises(RuntimeError, match=r"hyper_parameters\.config is required"):
        load_checkpoint_config(checkpoint)


def test_checkpoint_returns_the_recorded_composition_config(tmp_path: Path) -> None:
    checkpoint = tmp_path / "configured.ckpt"
    config = {"model": _axial_model_config()}
    torch.save({"hyper_parameters": {"config": config}, **_contract()}, checkpoint)

    assert load_checkpoint_config(checkpoint) == config


def test_checkpoint_rejects_missing_normalization_contract(tmp_path: Path) -> None:
    checkpoint = tmp_path / "old.ckpt"
    torch.save({"hyper_parameters": {"config": {}}}, checkpoint)

    with pytest.raises(ValueError, match="missing.*court_coordinate_normalization"):
        load_checkpoint_config(checkpoint)


def test_metadata_free_court_checkpoint_is_explicit_physical_v1_only(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "legacy_court.ckpt"
    torch.save(
        {
            "hyper_parameters": {"config": {"model": _axial_model_config()}},
            **_contract(),
        },
        checkpoint,
    )
    checkpoint_bytes = checkpoint.read_bytes()

    with pytest.raises(MissingCourtKeypointMetadataError, match="metadata is absent"):
        load_checkpoint_runtime(checkpoint)
    with pytest.raises(MissingCourtKeypointMetadataError, match="physical_v1"):
        load_checkpoint_runtime(
            checkpoint,
            runtime_court_keypoints="camera_view_v2",
        )

    runtime = load_checkpoint_runtime(
        checkpoint,
        runtime_court_keypoints="physical_v1",
    )
    assert runtime.legacy_metadata_free is True
    assert runtime.court_keypoint_contract == resolve_court_keypoint_contract(
        "physical_v1"
    )
    assert runtime.config["court_keypoints"]["selector"] == "physical_v1"
    assert checkpoint.read_bytes() == checkpoint_bytes


def test_checkpoint_rejects_retired_court_keypoint_contract(tmp_path: Path) -> None:
    path = tmp_path / "retired.ckpt"
    contract = resolve_court_keypoint_contract("camera_view_v2")
    checkpoint: dict[str, object] = {
        "hyper_parameters": {
            "config": {
                "model": _axial_model_config(),
                "court_keypoints": {"selector": "camera_view_v2"},
            }
        },
        **_contract(),
    }
    write_model_artifact_court_keypoint_contract(checkpoint, contract)
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="physical_v1"):
        load_checkpoint_runtime(path)
    with pytest.raises(CourtKeypointContractMismatchError):
        load_checkpoint_runtime(path, runtime_court_keypoints="physical_v1")


def test_camera_view_checkpoint_never_accepts_missing_marker(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "missing_court_view.ckpt"
    checkpoint: dict[str, object] = {
        "hyper_parameters": {
            "config": {
                "court_keypoints": {"selector": "camera_view_v2"},
            }
        },
        **_contract(),
    }
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(MissingCourtKeypointMetadataError, match="metadata is absent"):
        load_checkpoint_runtime(checkpoint_path)
