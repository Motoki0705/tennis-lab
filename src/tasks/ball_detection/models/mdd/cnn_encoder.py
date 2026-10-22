"""Spatial residual CNN; only two Conv3d layers mix sampled timesteps."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.tasks.ball_detection.models.mdd.blocks import (
    ConvNeXtV2Residual,
    FactorizedTemporalMix,
    PartialSpatialResidual,
)
from src.tasks.ball_detection.models.mdd.config import MDDPretrainConfig
from src.tasks.ball_detection.models.mdd.execution import checkpoint_call


class SpatialConv(nn.Sequential):
    def __init__(self, inputs: int, outputs: int, stride: int = 1) -> None:
        # Learned, randomly initialized channel offsets avoid an all-zero chain
        # through per-frame normalization on the mandated zero first MDD frame.
        super().__init__(nn.Conv2d(inputs, outputs, 3, stride=stride, padding=1, bias=True),
                         nn.GroupNorm(1, outputs), nn.GELU())


class SpatialResidual(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(SpatialConv(channels, channels),
                                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                    nn.GroupNorm(1, channels))

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x + self.layers(x))


class TemporalMix(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: Tensor, batch: int, frames: int) -> Tensor:
        _, c, h, w = x.shape
        x = self.conv(x.reshape(batch, frames, c, h, w).transpose(1, 2))
        x = x.transpose(1, 2).reshape(batch * frames, c, h, w)
        # Normalization is per frame, never across the time dimension.
        return F.gelu(self.norm(x))


class DeepMDDEncoder(nn.Module):
    """Return /8, /16, /32, /64 features as B,T,C,H,W, including both 3D layers."""
    def __init__(self, config: MDDPretrainConfig) -> None:
        super().__init__()
        self.activation_checkpointing = config.activation_checkpointing
        self.frame_chunk_size = config.frame_chunk_size
        widths = (*config.stem_channels, *config.mixed_channels)
        stages: list[nn.Module] = []
        previous = config.input_channels
        residual = {"residual": SpatialResidual, "convnext_v2": ConvNeXtV2Residual,
                    "fasternet": PartialSpatialResidual}[config.encoder_variant]
        for i, (width, depth) in enumerate(zip(widths, config.residual_blocks, strict=True)):
            spatial: list[nn.Module] = [SpatialConv(previous, width, stride=2)]
            if i >= 4:
                spatial.append(SpatialConv(width, width))
            spatial.extend(residual(width) for _ in range(depth))
            stages.append(nn.Sequential(*spatial))
            previous = width
        self.stages = nn.ModuleList(stages)
        temporal = TemporalMix if config.temporal_mixing == "dense3d" else FactorizedTemporalMix
        self.temporal = nn.ModuleList(temporal(c) for c in config.mixed_channels)
        self.feature_channels = widths[2:]

    def forward(self, mdd: Tensor) -> tuple[Tensor, ...]:
        if self.frame_chunk_size is None:
            return checkpoint_call(self._encode, mdd, enabled=self.activation_checkpointing and self.training)
        frames = mdd.shape[2]
        chunks: list[tuple[Tensor, ...]] = []
        for start in range(0, frames, self.frame_chunk_size):
            stop = min(start + self.frame_chunk_size, frames)
            # Two stride-one temporal kernels of size three require two real
            # neighbours on each side. Only clip edges receive zero padding.
            left, right = max(0, start - 2), min(frames, stop + 2)
            levels = checkpoint_call(self._encode, mdd[:, :, left:right],
                                     enabled=self.activation_checkpointing and self.training)
            chunks.append(tuple(level[:, start - left:stop - left] for level in levels))
        return tuple(torch.cat(levels, dim=1) for levels in zip(*chunks, strict=True))

    def _encode(self, mdd: Tensor) -> tuple[Tensor, ...]:
        b, c, t, height, width = mdd.shape
        x = mdd.transpose(1, 2).reshape(b * t, c, height, width)
        features: list[Tensor] = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i >= 4:
                x = self.temporal[i - 4](x, b, t)
            if i >= 2:
                features.append(x.reshape(b, t, *x.shape[1:]))
        return tuple(features)
