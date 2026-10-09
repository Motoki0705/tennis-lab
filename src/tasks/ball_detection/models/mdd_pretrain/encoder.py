"""Spatial residual CNN; only two Conv3d layers mix sampled timesteps."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import MDDPretrainConfig


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
        widths = (*config.stem_channels, *config.mixed_channels)
        stages: list[nn.Module] = []
        previous = 2
        for i, (width, depth) in enumerate(zip(widths, config.residual_blocks, strict=True)):
            spatial: list[nn.Module] = [SpatialConv(previous, width, stride=2)]
            if i >= 4:
                spatial.append(SpatialConv(width, width))
            spatial.extend(SpatialResidual(width) for _ in range(depth))
            stages.append(nn.Sequential(*spatial))
            previous = width
        self.stages = nn.ModuleList(stages)
        self.temporal = nn.ModuleList(TemporalMix(c) for c in config.mixed_channels)
        self.feature_channels = widths[2:]

    def forward(self, mdd: Tensor) -> tuple[Tensor, ...]:
        b, c, t, height, width = mdd.shape
        x = mdd.transpose(1, 2).reshape(b * t, c, height, width)
        features: list[Tensor] = []
        for i, stage in enumerate(self.stages):
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
            if i >= 4:
                x = self.temporal[i - 4](x, b, t)
            if i >= 2:
                features.append(x.reshape(b, t, *x.shape[1:]))
        return tuple(features)
