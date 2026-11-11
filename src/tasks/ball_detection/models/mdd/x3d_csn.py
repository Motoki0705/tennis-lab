"""X3D-style expansion and CSN depthwise spatiotemporal residual blocks."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from src.tasks.ball_detection.models.mdd.config import MDDPretrainConfig
from src.tasks.ball_detection.models.mdd.execution import checkpoint_call
from src.tasks.ball_detection.models.mdd.stage_config import MDDStageHeatmapConfig


class FrameGroupNorm(nn.GroupNorm):
    """Normalize C,H,W independently for every B,T entry, including clip edges."""

    def forward(self, x: Tensor) -> Tensor:
        b, c, t, h, w = x.shape
        frames = x.transpose(1, 2).reshape(b * t, c, h, w)
        return super().forward(frames).reshape(b, t, c, h, w).transpose(1, 2)


class X3DCSNBlock(nn.Module):
    """IB, expansion-before/after DW, or pointwise FFN; frame-local norms."""

    def __init__(self, channels: int, *, kind: str = "ib", expansion: float = 2.0) -> None:
        super().__init__()
        if kind not in {"ib", "convnext", "ffn"}:
            raise ValueError("Unknown X3D/CSN block kind")
        if type(expansion) not in (float, int) or not math.isfinite(expansion) or expansion < 1:
            raise ValueError("Expansion must be finite and >= 1")
        self.hidden_channels = math.ceil(channels * expansion)
        expanded = self.hidden_channels
        expand = [
            nn.Conv3d(channels, expanded, 1, bias=False),
            FrameGroupNorm(1, expanded),
            nn.GELU(),
        ]
        dw_channels = expanded if kind == "ib" else channels
        depthwise = [
            nn.Conv3d(dw_channels, dw_channels, 3, padding=1, groups=dw_channels, bias=False),
            FrameGroupNorm(1, dw_channels), nn.GELU(),
        ] if kind != "ffn" else []
        self.layers = nn.Sequential(
            *(expand + depthwise if kind == "ib" else depthwise + expand),
            nn.Conv3d(expanded, channels, 1, bias=False),
            FrameGroupNorm(1, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.layers(x))


class X3DCSNEncoder(nn.Module):
    """Return /8,/16,/32,/64 B,T,C,H,W with no temporal stride.

    Only the depthwise 3x3x3 in each residual block mixes time. Therefore the
    /64 radius is exactly sum(residual_blocks); the standard five stage depths
    1/2/3/3/2 give 11. Normalization and downsampling are frame-local. Halo size
    is derived from the topology and contributes no parameters or buffers.
    """

    def __init__(self, config: MDDPretrainConfig) -> None:
        super().__init__()
        if config.encoder_variant != "x3d_csn" or config.temporal_mixing != "depthwise3d":
            raise ValueError("X3D/CSN requires x3d_csn/depthwise3d configuration")
        self.activation_checkpointing = config.activation_checkpointing
        self.frame_chunk_size = config.frame_chunk_size
        kinds = config.stage_blocks if isinstance(config, MDDStageHeatmapConfig) else ("ib",) * 5
        expansions = config.expansion_ratios if isinstance(config, MDDStageHeatmapConfig) else (2.0,) * 5
        radii, fields, radius, field, jump = [], [], 0, 3, 2
        for stage, (kind, depth) in enumerate(zip(kinds, config.residual_blocks[1:], strict=True), 1):
            field += 2 * jump
            jump *= 2
            if kind != "ffn":
                radius += depth
                field += 2 * jump * depth
            if stage >= 2:
                radii.append(radius)
                fields.append(field)
        self.temporal_radius = radius
        self.feature_temporal_radii = tuple(radii)
        # Convolution-only spatial support; FrameGroupNorm additionally mixes
        # every spatial position within each frame (never across time).
        self.convolutional_spatial_fields = tuple(fields)
        widths = (*config.stem_channels, *config.mixed_channels)
        self.feature_channels = widths[2:]
        stem = nn.Sequential(
            nn.Conv3d(config.input_channels, widths[0], (1, 3, 3),
                      stride=(1, 2, 2), padding=(0, 1, 1), bias=True),
            FrameGroupNorm(1, widths[0]),
            nn.GELU(),
        )
        stages: list[nn.Module] = [stem]
        for previous, width, depth, kind, expansion in zip(widths[:-1], widths[1:], config.residual_blocks[1:], kinds, expansions, strict=True):
            stages.append(nn.Sequential(
                nn.Conv3d(previous, previous, (1, 3, 3), stride=(1, 2, 2),
                          padding=(0, 1, 1), groups=previous, bias=False),
                FrameGroupNorm(1, previous),
                nn.GELU(),
                nn.Conv3d(previous, width, 1, bias=False),
                FrameGroupNorm(1, width),
                nn.GELU(),
                *(X3DCSNBlock(width, kind=kind, expansion=expansion) for _ in range(depth)),
            ))
        self.stages = nn.ModuleList(stages)

    def _encode(self, x: Tensor) -> tuple[Tensor, ...]:
        features = []
        for index, stage in enumerate(self.stages):
            x = checkpoint_call(stage, x, enabled=self.activation_checkpointing and self.training)
            if index >= 2:
                features.append(x.transpose(1, 2))
        return tuple(features)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        if self.frame_chunk_size is None:
            return self._encode(x)
        frames = x.shape[2]
        chunks = []
        for start in range(0, frames, self.frame_chunk_size):
            stop = min(start + self.frame_chunk_size, frames)
            left, right = max(0, start - self.temporal_radius), min(frames, stop + self.temporal_radius)
            levels = self._encode(x[:, :, left:right])
            chunks.append(tuple(level[:, start - left:stop - left] for level in levels))
        return tuple(torch.cat(levels, dim=1) for levels in zip(*chunks, strict=True))
