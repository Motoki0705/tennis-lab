"""Tests for fail-closed BLCS checkpoint composition metadata."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.tasks.base.data import (
    CourtCoordinateContractMismatchError,
    MissingCourtCoordinateMetadataError,
)
from src.tasks.base.model_io import write_checkpoint_court_coordinate_contract
from src.tasks.blcs.model_io.checkpoints import (
    load_checkpoint_config,
    load_checkpoint_runtime,
)
from src.utils.schema.court_normalization import resolve_court_coordinate_normalization


def test_checkpoint_requires_explicit_composition_config(tmp_path: Path) -> None:
    checkpoint = tmp_path / "missing_config.ckpt"
    torch.save({"state_dict": {}}, checkpoint)

    with pytest.raises(RuntimeError, match=r"hyper_parameters\.config is required"):
        load_checkpoint_config(checkpoint)


def test_checkpoint_returns_the_recorded_composition_config(tmp_path: Path) -> None:
    checkpoint = tmp_path / "configured.ckpt"
    config = {"model": {"name": "blcs"}}
    torch.save({"hyper_parameters": {"config": config}}, checkpoint)

    assert load_checkpoint_config(checkpoint) == config


def test_metadata_free_blcs_checkpoint_is_explicit_v1_only(tmp_path: Path) -> None:
    checkpoint = tmp_path / "legacy.ckpt"
    torch.save(
        {"hyper_parameters": {"config": {"model": {"name": "blcs"}}}},
        checkpoint,
    )
    checkpoint_bytes = checkpoint.read_bytes()

    with pytest.raises(MissingCourtCoordinateMetadataError, match="metadata is absent"):
        load_checkpoint_runtime(checkpoint)
    with pytest.raises(MissingCourtCoordinateMetadataError, match="legacy v1 only"):
        load_checkpoint_runtime(checkpoint, runtime_normalization="v2")

    runtime = load_checkpoint_runtime(checkpoint, runtime_normalization="v1")
    assert runtime.legacy_metadata_free is True
    assert runtime.normalization.version == "v1"
    assert runtime.config["court_coordinate_normalization"]["version"] == "v1"
    assert runtime.config["training"] == {
        "position_huber_beta_v1": 1.0,
        "position_huber_transition_m_v2": 1.0,
    }
    assert checkpoint.read_bytes() == checkpoint_bytes


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("position_huber_beta_v1", 0.5),
        ("position_huber_transition_m_v2", 2.0),
    ],
)
def test_metadata_free_v1_checkpoint_rejects_conflicting_loss_overlay(
    tmp_path: Path,
    field: str,
    value: float,
) -> None:
    checkpoint = tmp_path / "legacy_conflict.ckpt"
    torch.save(
        {
            "hyper_parameters": {
                "config": {"training": {field: value}},
            }
        },
        checkpoint,
    )

    with pytest.raises(RuntimeError, match=rf"conflicting training\.{field}"):
        load_checkpoint_runtime(checkpoint, runtime_normalization="v1")


def test_checkpoint_metadata_saved_config_and_runtime_must_agree(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "versioned.ckpt"
    checkpoint: dict[str, object] = {
        "hyper_parameters": {
            "config": {"court_coordinate_normalization": {"version": "v1"}}
        }
    }
    write_checkpoint_court_coordinate_contract(
        checkpoint,
        resolve_court_coordinate_normalization("v2"),
    )
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(CourtCoordinateContractMismatchError, match="config normalization"):
        load_checkpoint_runtime(checkpoint_path)
