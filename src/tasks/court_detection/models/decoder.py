"""Shared decoders for court detection models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.models.blocks import Conv2dWiseWiseBlock


class CourtFPNDecoder(nn.Module):
    """Top-down decoder that fuses a four-stage feature pyramid."""

    def __init__(
        self,
        *,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
    ) -> None:
        super().__init__()
        self.encoder_channels = tuple(int(channel) for channel in encoder_channels)
        self.decoder_channels = tuple(int(channel) for channel in decoder_channels)
        self._validate_init_args(
            encoder_channels=self.encoder_channels,
            decoder_channels=self.decoder_channels,
        )

        self.output_channels = self.decoder_channels[0]
        self.lateral_blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for in_channels, out_channels in zip(
                self.encoder_channels,
                self.decoder_channels,
                strict=True,
            )
        )
        self.deepest_block = Conv2dWiseWiseBlock(
            self.decoder_channels[-1],
            self.decoder_channels[-1],
        )
        self.fusion_blocks = nn.ModuleList(
            Conv2dWiseWiseBlock(
                self.decoder_channels[level] + self.decoder_channels[level + 1],
                self.decoder_channels[level],
            )
            for level in range(len(self.decoder_channels) - 2, -1, -1)
        )

    @staticmethod
    def _validate_init_args(
        *,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
    ) -> None:
        if len(encoder_channels) != 4:
            raise ValueError(
                "CourtFPNDecoder expects four encoder feature levels, "
                f"got {len(encoder_channels)}."
            )
        if len(decoder_channels) != 4:
            raise ValueError(
                "CourtFPNDecoder expects four decoder channels, "
                f"got {len(decoder_channels)}."
            )
        if any(channel <= 0 for channel in decoder_channels):
            raise ValueError("decoder_channels must contain positive integers.")

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        self._validate_forward_inputs(feats)

        projected_feats = [
            lateral_block(feat)
            for lateral_block, feat in zip(self.lateral_blocks, feats, strict=True)
        ]
        x = self.deepest_block(projected_feats[-1])

        for block_index, level in enumerate(range(len(projected_feats) - 2, -1, -1)):
            lateral_feat = projected_feats[level]
            if x.shape[-2:] != lateral_feat.shape[-2:]:
                x = F.interpolate(
                    x,
                    size=lateral_feat.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            x = torch.cat([x, lateral_feat], dim=1)
            x = self.fusion_blocks[block_index](x)
        return cast("torch.Tensor", x)

    def _validate_forward_inputs(self, feats: Sequence[torch.Tensor]) -> None:
        if len(feats) != len(self.encoder_channels):
            raise ValueError(
                f"CourtFPNDecoder expects four feature levels, got {len(feats)}."
            )


class CourtUNetDecoder(nn.Module):
    """U-Net style decoder that fuses a four-stage feature hierarchy."""

    def __init__(
        self,
        *,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
    ) -> None:
        super().__init__()
        self.encoder_channels = tuple(int(channel) for channel in encoder_channels)
        self.decoder_channels = tuple(int(channel) for channel in decoder_channels)
        self._validate_init_args(
            encoder_channels=self.encoder_channels,
            decoder_channels=self.decoder_channels,
        )

        self.output_channels = self.decoder_channels[0]
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for in_channels, out_channels in zip(
                self.encoder_channels,
                self.decoder_channels,
                strict=True,
            )
        )
        self.deepest_block = Conv2dWiseWiseBlock(
            self.decoder_channels[-1],
            self.decoder_channels[-1],
        )
        self.decode_blocks = nn.ModuleList(
            Conv2dWiseWiseBlock(
                self.decoder_channels[level + 1] + self.decoder_channels[level],
                self.decoder_channels[level],
            )
            for level in range(len(self.decoder_channels) - 2, -1, -1)
        )

    @staticmethod
    def _validate_init_args(
        *,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
    ) -> None:
        if len(encoder_channels) != 4:
            raise ValueError(
                "CourtUNetDecoder expects four encoder feature levels, "
                f"got {len(encoder_channels)}."
            )
        if len(decoder_channels) != 4:
            raise ValueError(
                "CourtUNetDecoder expects four decoder channels, "
                f"got {len(decoder_channels)}."
            )
        if any(channel <= 0 for channel in decoder_channels):
            raise ValueError("decoder_channels must contain positive integers.")

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        self._validate_forward_inputs(feats)

        projected_feats = [
            projection(feat)
            for projection, feat in zip(self.projections, feats, strict=True)
        ]
        x = self.deepest_block(projected_feats[-1])

        for block_index, level in enumerate(range(len(projected_feats) - 2, -1, -1)):
            skip = projected_feats[level]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            x = self.decode_blocks[block_index](torch.cat([x, skip], dim=1))
        return cast("torch.Tensor", x)

    def _validate_forward_inputs(self, feats: Sequence[torch.Tensor]) -> None:
        if len(feats) != len(self.encoder_channels):
            raise ValueError(
                f"CourtUNetDecoder expects four feature levels, got {len(feats)}."
            )


class DPTFeatureFusionBlock(nn.Module):
    """RefineNet-style residual fusion block used by DPT decoders."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        self.skip_block = Conv2dWiseWiseBlock(channels, channels)
        self.output_block = Conv2dWiseWiseBlock(channels, channels)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            x = x + self.skip_block(skip)
        return cast("torch.Tensor", self.output_block(x))


class CourtDPTDecoder(nn.Module):
    """DPT decoder for ViT features reassembled at multiple image scales."""

    def __init__(
        self,
        *,
        encoder_channels: Sequence[int],
        decoder_channels: int,
        reassemble_factors: Sequence[float],
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
        self.fusion_blocks = nn.ModuleList(
            DPTFeatureFusionBlock(self.decoder_channels)
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
                "CourtDPTDecoder expects four encoder feature levels, "
                f"got {len(encoder_channels)}."
            )
        if decoder_channels <= 0:
            raise ValueError("decoder_channels must be positive.")
        if len(reassemble_factors) != 4:
            raise ValueError("reassemble_factors must contain exactly four values.")
        if any(factor <= 0.0 for factor in reassemble_factors):
            raise ValueError("reassemble_factors must be positive.")

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        self._validate_forward_inputs(feats)
        projected_feats = [
            self._resize_by_factor(projection(feat), factor)
            for projection, feat, factor in zip(
                self.projections,
                feats,
                self.reassemble_factors,
                strict=True,
            )
        ]

        x = self.fusion_blocks[-1](projected_feats[-1])
        for block, skip in zip(
            reversed(self.fusion_blocks[:-1]),
            reversed(projected_feats[:-1]),
            strict=True,
        ):
            x = block(x, skip)
        return cast("torch.Tensor", x)

    @staticmethod
    def _resize_by_factor(x: torch.Tensor, factor: float) -> torch.Tensor:
        if factor == 1.0:
            return x
        return F.interpolate(
            x,
            scale_factor=factor,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )

    def _validate_forward_inputs(self, feats: Sequence[torch.Tensor]) -> None:
        if len(feats) != len(self.encoder_channels):
            raise ValueError(
                f"CourtDPTDecoder expects four feature levels, got {len(feats)}."
            )


def build_court_decoder(
    *,
    decoder_name: str,
    encoder_channels: Sequence[int],
    decoder_channels: Sequence[int] | int,
    reassemble_factors: Sequence[float] | None,
) -> CourtDecoder:
    """Build a court decoder from a decoder name."""

    resolved_decoder_name = str(decoder_name).lower()
    if resolved_decoder_name == "fpn":
        return CourtFPNDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=_parse_decoder_channel_sequence(decoder_channels),
        )
    if resolved_decoder_name in {"unet", "u-net"}:
        return CourtUNetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=_parse_decoder_channel_sequence(decoder_channels),
        )
    if resolved_decoder_name == "dpt":
        return CourtDPTDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=_parse_decoder_channel_scalar(decoder_channels),
            reassemble_factors=_require_reassemble_factors(reassemble_factors),
        )
    raise ValueError(f"Unsupported court decoder: {decoder_name}")


CourtDecoder: TypeAlias = CourtFPNDecoder | CourtUNetDecoder | CourtDPTDecoder


def _require_reassemble_factors(value: Sequence[float] | None) -> Sequence[float]:
    if value is None:
        raise ValueError("DPT decoder requires reassemble_factors.")
    return value


def _parse_decoder_channel_sequence(value: Sequence[int] | int) -> tuple[int, ...]:
    if isinstance(value, int):
        raise ValueError("CNN decoders require a sequence of four channel counts.")
    return tuple(int(channel) for channel in value)


def _parse_decoder_channel_scalar(value: Sequence[int] | int) -> int:
    if isinstance(value, int):
        return value
    channels = tuple(int(channel) for channel in value)
    if len(channels) != 1:
        raise ValueError("DPT decoder expects a single channel count.")
    return channels[0]


__all__ = [
    "CourtDPTDecoder",
    "CourtFPNDecoder",
    "CourtUNetDecoder",
    "DPTFeatureFusionBlock",
    "build_court_decoder",
]
