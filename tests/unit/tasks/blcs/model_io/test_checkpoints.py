"""Tests for fail-closed BLCS checkpoint composition metadata."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.tasks.blcs.model_io.checkpoints import load_checkpoint_config
from src.utils.schema.court_normalization import (
    court_coordinate_normalization_metadata,
)


def _contract() -> dict[str, object]:
    return {
        "court_coordinate_normalization": (court_coordinate_normalization_metadata())
    }


def test_checkpoint_requires_explicit_composition_config(tmp_path: Path) -> None:
    checkpoint = tmp_path / "missing_config.ckpt"
    torch.save({"state_dict": {}, **_contract()}, checkpoint)

    with pytest.raises(RuntimeError, match=r"hyper_parameters\.config is required"):
        load_checkpoint_config(checkpoint)


def test_checkpoint_returns_the_recorded_composition_config(tmp_path: Path) -> None:
    checkpoint = tmp_path / "configured.ckpt"
    config = {"model": {"name": "blcs"}}
    torch.save({"hyper_parameters": {"config": config}, **_contract()}, checkpoint)

    assert load_checkpoint_config(checkpoint) == config


def test_checkpoint_rejects_missing_normalization_contract(tmp_path: Path) -> None:
    checkpoint = tmp_path / "old.ckpt"
    torch.save({"hyper_parameters": {"config": {}}}, checkpoint)

    with pytest.raises(ValueError, match="missing.*court_coordinate_normalization"):
        load_checkpoint_config(checkpoint)
