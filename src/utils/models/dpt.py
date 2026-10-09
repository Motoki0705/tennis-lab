"""DPT multiscale spatial fusion shared by court and ball heatmap models."""
from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from src.utils.models.blocks import Conv2dWiseWiseBlock


def _apply_tensor_module(module: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, module(tensor))


class ResidualConvUnit(nn.Module):
    """Original DPT-style two-convolution residual unit, without batch statistics."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(F.relu(self.conv1(F.relu(x))))


class DPTFeatureFusionBlock(nn.Module):
    """RefineNet-style residual fusion block used by DPT decoders."""

    def __init__(self, channels: int, *, block_style: str = "wise") -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        if block_style == "wise":
            self.skip_block: nn.Module = Conv2dWiseWiseBlock(channels, channels)
            self.output_block: nn.Module = Conv2dWiseWiseBlock(channels, channels)
        elif block_style == "residual":
            self.skip_block = ResidualConvUnit(channels)
            self.output_block = ResidualConvUnit(channels)
        else:
            raise ValueError("DPT block_style must be wise or residual")

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = x + self.skip_block(skip)
        return cast("torch.Tensor", self.output_block(x))


class DPTDecoder(nn.Module):
    """DPT decoder for ViT features reassembled at multiple image scales."""

    def __init__(
        self,
        *,
        encoder_channels: Sequence[int],
        decoder_channels: int,
        reassemble_factors: Sequence[float],
        block_style: str = "wise",
    ) -> None:
        super().__init__()
        self.encoder_channels = tuple(int(channel) for channel in encoder_channels)
        self.decoder_channels = int(decoder_channels)
        self.reassemble_factors = tuple(float(factor) for factor in reassemble_factors)
        self._validate_init_args(
            encoder_channels=self.encoder_channels,
            decoder_channels=self.decoder_channels,
            reassemble_factors=self.reassemble_factors,
        )

        self.output_channels = self.decoder_channels
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    in_channels, self.decoder_channels, kernel_size=1, bias=False
                ),
                nn.GroupNorm(1, self.decoder_channels),
                nn.GELU(),
            )
            for in_channels in self.encoder_channels
        )
        self.reassembly = nn.ModuleList(
            nn.Identity()
            if factor == 1.0
            else nn.Upsample(
                scale_factor=factor,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )
            for factor in self.reassemble_factors
        )
        self.fusion_blocks = nn.ModuleList(
            DPTFeatureFusionBlock(self.decoder_channels, block_style=block_style)
            for _ in range(len(self.encoder_channels))
        )

    @staticmethod
    def _validate_init_args(
        *,
        encoder_channels: Sequence[int],
        decoder_channels: int,
        reassemble_factors: Sequence[float],
    ) -> None:
        if len(encoder_channels) != 4:
            raise ValueError(
                "DPTDecoder expects four encoder feature levels, "
                f"got {len(encoder_channels)}."
            )
        if decoder_channels <= 0:
            raise ValueError("decoder_channels must be positive.")
        if len(reassemble_factors) != 4:
            raise ValueError("reassemble_factors must contain exactly four values.")
        if any(factor <= 0.0 for factor in reassemble_factors):
            raise ValueError("reassemble_factors must be positive.")

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        projected_feats = [
            _apply_tensor_module(
                reassemble,
                _apply_tensor_module(projection, feat),
            )
            for projection, reassemble, feat in zip(
                self.projections,
                self.reassembly,
                feats,
                strict=True,
            )
        ]

        deepest_fusion = cast("DPTFeatureFusionBlock", self.fusion_blocks[-1])
        x = deepest_fusion.output_block(projected_feats[-1])
        for block, skip in zip(
            reversed(self.fusion_blocks[:-1]),
            reversed(projected_feats[:-1]),
            strict=True,
        ):
            x = cast("DPTFeatureFusionBlock", block)(x, skip)
        return cast("torch.Tensor", x)

