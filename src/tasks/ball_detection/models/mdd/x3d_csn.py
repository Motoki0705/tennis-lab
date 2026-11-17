"""X3D-style expansion and CSN depthwise spatiotemporal residual blocks."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.tasks.ball_detection.models.mdd.config import MDDPretrainConfig
from src.tasks.ball_detection.models.mdd.execution import checkpoint_call


class FrameGroupNorm(nn.GroupNorm):
    """Normalize C,H,W independently for every B,T entry, including clip edges."""

    def forward(self, x: Tensor) -> Tensor:
        b, c, t, h, w = x.shape
        frames = x.transpose(1, 2).reshape(b * t, c, h, w)
        return super().forward(frames).reshape(b, t, c, h, w).transpose(1, 2)


class X3DCSNBlock(nn.Module):
    """Expand by two, depthwise 3x3x3, project, then add the residual."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        expanded = 2 * channels
        self.layers = nn.Sequential(
            nn.Conv3d(channels, expanded, 1, bias=False),
            FrameGroupNorm(1, expanded),
            nn.GELU(),
            nn.Conv3d(expanded, expanded, 3, padding=1, groups=expanded, bias=False),
            FrameGroupNorm(1, expanded),
            nn.GELU(),
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
        self.temporal_radius = sum(config.residual_blocks)
        widths = (*config.stem_channels, *config.mixed_channels)
        self.feature_channels = widths[2:]
        stem = nn.Sequential(
            nn.Conv3d(config.input_channels, widths[0], (1, 3, 3),
                      stride=(1, 2, 2), padding=(0, 1, 1), bias=True),
            FrameGroupNorm(1, widths[0]),
            nn.GELU(),
        )
        stages: list[nn.Module] = [stem]
        for previous, width, depth in zip(widths[:-1], widths[1:], config.residual_blocks[1:], strict=True):
            stages.append(nn.Sequential(
                nn.Conv3d(previous, previous, (1, 3, 3), stride=(1, 2, 2),
                          padding=(0, 1, 1), groups=previous, bias=False),
                FrameGroupNorm(1, previous),
                nn.GELU(),
                nn.Conv3d(previous, width, 1, bias=False),
                FrameGroupNorm(1, width),
                nn.GELU(),
                *(X3DCSNBlock(width) for _ in range(depth)),
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
