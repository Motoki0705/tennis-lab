"""Tests for checkpoint-defined Court dense-head reconstruction."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.tasks.court_detection.models.checkpoint_dense_head import (
    CourtCheckpointDenseResidualHead,
    build_checkpoint_dense_heads,
)


def _residual_config() -> dict[str, object]:
    return {
        "name": "residual",
        "normalization_groups": 4,
        "kp": {"hidden_channels": 16, "depth": 2},
        "seg": {"hidden_channels": 12, "depth": 1},
        "line": {"hidden_channels": 8, "depth": 3},
    }


def test_residual_heads_restore_serialized_hierarchy_and_shapes() -> None:
    heads = build_checkpoint_dense_heads(
        _residual_config(),
        input_channels=32,
        output_channels={"kp": 14, "seg": 7, "line": 1},
    )

    features = torch.randn(2, 32, 9, 11)
    assert isinstance(heads["kp"], CourtCheckpointDenseResidualHead)
    assert heads["kp"](features).shape == (2, 14, 9, 11)
    assert heads["seg"](features).shape == (2, 7, 9, 11)
    assert heads["line"](features).shape == (2, 1, 9, 11)

    state_keys = set(heads.state_dict())
    assert "kp.stem.0.weight" in state_keys
    assert "kp.blocks.1.network.4.bias" in state_keys
    assert "kp.output.weight" in state_keys
    assert "seg.blocks.0.network.0.weight" in state_keys
    assert "line.blocks.2.network.3.weight" in state_keys


def test_linear_checkpoint_head_remains_one_by_one_convolution() -> None:
    heads = build_checkpoint_dense_heads(
        {"name": "linear"},
        input_channels=32,
        output_channels={"kp": 14, "seg": 7, "line": 1},
    )

    assert all(isinstance(head, nn.Conv2d) for head in heads.values())
    assert heads["kp"].kernel_size == (1, 1)
    assert heads["kp"](torch.randn(1, 32, 5, 7)).shape == (1, 14, 5, 7)


def test_residual_checkpoint_head_requires_every_target_branch() -> None:
    config = _residual_config()
    config.pop("line")

    with pytest.raises(ValueError, match="missing branch 'line'"):
        build_checkpoint_dense_heads(
            config,
            input_channels=32,
            output_channels={"kp": 14, "seg": 7, "line": 1},
        )


def test_residual_checkpoint_head_validates_group_divisibility() -> None:
    config = _residual_config()
    config["kp"] = {"hidden_channels": 10, "depth": 2}

    with pytest.raises(ValueError, match="divisible"):
        build_checkpoint_dense_heads(
            config,
            input_channels=32,
            output_channels={"kp": 14, "seg": 7, "line": 1},
        )
