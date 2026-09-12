"""Task-specific residual heads for dense Court outputs."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from src.tasks.court_detection.configuration import (
    CourtDenseHeadBranchConfig,
    CourtDenseHeadConfig,
)
from src.tasks.court_detection.data.contracts import CourtTargetKind


class CourtDenseResidualBlock(nn.Module):
    """Spatial residual adapter with depthwise context and pointwise mixing."""

    def __init__(self, channels: int, *, normalization_groups: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("Dense residual block channels must be positive.")
        if normalization_groups <= 0 or channels % normalization_groups:
            raise ValueError(
                "Dense residual block channels must be divisible by "
                "normalization_groups."
            )
        self.network = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(normalization_groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(normalization_groups, channels),
        )
        self.activation = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return cast("Tensor", self.activation(features + self.network(features)))


class CourtDenseResidualHead(nn.Module):
    """Adapt one shared DPT feature map to one dense target representation."""

    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
        config: CourtDenseHeadBranchConfig,
        normalization_groups: int,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or output_channels <= 0:
            raise ValueError("Dense head input/output channels must be positive.")
        if config.hidden_channels % normalization_groups:
            raise ValueError(
                "Dense head hidden_channels must be divisible by normalization_groups."
            )
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                config.hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(normalization_groups, config.hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *(
                CourtDenseResidualBlock(
                    config.hidden_channels,
                    normalization_groups=normalization_groups,
                )
                for _ in range(config.depth)
            )
        )
        self.output = nn.Conv2d(
            config.hidden_channels,
            output_channels,
            kernel_size=1,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return cast("Tensor", self.output(self.blocks(self.stem(features))))


def build_court_dense_head(
    *,
    kind: CourtTargetKind,
    input_channels: int,
    output_channels: int,
    config: CourtDenseHeadConfig,
) -> nn.Module:
    """Build either the legacy linear seam or a configured residual adapter."""
    if config.name == "linear":
        return nn.Conv2d(input_channels, output_channels, kernel_size=1)
    if config.normalization_groups is None or kind not in config.branches:
        raise ValueError("Residual dense head configuration is incomplete.")
    return CourtDenseResidualHead(
        input_channels=input_channels,
        output_channels=output_channels,
        config=config.branches[kind],
        normalization_groups=config.normalization_groups,
    )


__all__ = [
    "CourtDenseResidualBlock",
    "CourtDenseResidualHead",
    "build_court_dense_head",
]
