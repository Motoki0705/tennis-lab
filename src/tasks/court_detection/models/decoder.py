"""Shared FPN decoder for court detection models."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tasks.court_detection.models.blocks import Conv2dWiseWiseBlock


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
        if len(self.encoder_channels) != 4:
            raise ValueError(
                "CourtFPNDecoder expects four encoder feature levels, "
                f"got {len(self.encoder_channels)}."
            )
        if len(self.decoder_channels) != 4:
            raise ValueError(
                "CourtFPNDecoder expects four decoder channels, "
                f"got {len(self.decoder_channels)}."
            )
        if any(channel <= 0 for channel in self.decoder_channels):
            raise ValueError("decoder_channels must contain positive integers.")

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

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(feats) != len(self.encoder_channels):
            raise ValueError(
                "CourtFPNDecoder expects four feature levels, "
                f"got {len(feats)}."
            )

        projected_feats = [
            lateral_block(feat)
            for lateral_block, feat in zip(self.lateral_blocks, feats, strict=True)
        ]
        x = self.deepest_block(projected_feats[-1])

        for block_index, level in enumerate(range(len(projected_feats) - 2, -1, -1)):
            lateral_feat = projected_feats[level]
            if x.shape[-2:] != lateral_feat.shape[-2:]:
                x = F.interpolate(x, size=lateral_feat.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, lateral_feat], dim=1)
            x = self.fusion_blocks[block_index](x)
        return x


__all__ = ["CourtFPNDecoder"]