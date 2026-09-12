"""Unit tests for task-specific Court dense residual heads."""

from __future__ import annotations

from types import MappingProxyType

import pytest
import torch
from torch import nn

from src.tasks.court_detection.configuration import (
    CourtDenseHeadBranchConfig,
    CourtDenseHeadConfig,
)
from src.tasks.court_detection.models.dense_head import (
    CourtDenseResidualBlock,
    CourtDenseResidualHead,
    build_court_dense_head,
)


def _residual_config() -> CourtDenseHeadConfig:
    return CourtDenseHeadConfig(
        name="residual",
        normalization_groups=4,
        branches=MappingProxyType(
            {
                "kp": CourtDenseHeadBranchConfig(hidden_channels=16, depth=2),
                "seg": CourtDenseHeadBranchConfig(hidden_channels=12, depth=1),
                "line": CourtDenseHeadBranchConfig(hidden_channels=8, depth=3),
            }
        ),
    )


@pytest.mark.parametrize(
    ("kind", "output_channels", "hidden_channels", "depth"),
    [("kp", 14, 16, 2), ("seg", 7, 12, 1), ("line", 1, 8, 3)],
)
def test_residual_heads_honor_per_target_capacity(
    kind: str,
    output_channels: int,
    hidden_channels: int,
    depth: int,
) -> None:
    head = build_court_dense_head(
        kind=kind,  # type: ignore[arg-type]
        input_channels=32,
        output_channels=output_channels,
        config=_residual_config(),
    )

    assert isinstance(head, CourtDenseResidualHead)
    assert isinstance(head.stem[0], nn.Conv2d)
    assert head.stem[0].out_channels == hidden_channels
    assert len(head.blocks) == depth
    assert all(isinstance(block, CourtDenseResidualBlock) for block in head.blocks)
    output = head(torch.randn(2, 32, 9, 11))
    assert output.shape == (2, output_channels, 9, 11)
    output.mean().backward()
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_linear_head_remains_explicit_for_legacy_architectures() -> None:
    head = build_court_dense_head(
        kind="line",
        input_channels=32,
        output_channels=1,
        config=CourtDenseHeadConfig(
            name="linear",
            normalization_groups=None,
            branches=MappingProxyType({}),
        ),
    )

    assert isinstance(head, nn.Conv2d)
    assert head.kernel_size == (1, 1)
