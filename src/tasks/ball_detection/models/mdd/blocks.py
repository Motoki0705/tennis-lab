"""Paper-inspired spatial blocks; frame-local and initialized from scratch.

ConvNeXt V2: arxiv.org/abs/2301.00808; FasterNet: arxiv.org/abs/2303.03667.
These use the MDD encoder's existing widths/strides, not published model presets.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GlobalResponseNorm(nn.Module):
    """Per-frame NHWC response competition, accumulating energy in FP32."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: Tensor) -> Tensor:
        # A reduction avoids materializing a full-resolution FP32 squared feature map.
        # vector_norm has a defined zero-gradient at zero response (blank MDD).
        energy = torch.linalg.vector_norm(x, ord=2, dim=(1, 2), keepdim=True, dtype=torch.float32)
        response = energy / (energy.mean(-1, keepdim=True) + 1.e-6)
        return x + self.gamma.to(x.dtype) * (x * response.to(x.dtype)) + self.beta.to(x.dtype)


class ConvNeXtV2Residual(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.spatial = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1.e-6)
        self.expand = nn.Linear(channels, 4 * channels)
        self.response = GlobalResponseNorm(4 * channels)
        self.project = nn.Linear(4 * channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        y = self.spatial(x).permute(0, 2, 3, 1)
        y = self.project(self.response(F.gelu(self.expand(self.norm(y)))))
        return x + y.permute(0, 3, 1, 2)


class PartialSpatialResidual(nn.Module):
    """PConv on one quarter of channels and ratio-2 pointwise mixing.

Frame-local GroupNorm replaces the reference model's BatchNorm, preventing
temporal leakage when B and T are flattened together.
"""
    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels < 4:
            raise ValueError("PConv requires at least four channels")
        self.partial_channels = channels // 4
        self.spatial = nn.Conv2d(self.partial_channels, self.partial_channels, 3, padding=1, bias=False)
        self.mix = nn.Sequential(nn.Conv2d(channels, 2 * channels, 1), nn.GroupNorm(1, 2 * channels),
                                 nn.GELU(), nn.Conv2d(2 * channels, channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        selected, retained = x.split((self.partial_channels, x.shape[1] - self.partial_channels), dim=1)
        return x + self.mix(torch.cat((self.spatial(selected), retained), dim=1))


class FactorizedTemporalMix(nn.Module):
    """Spatial 3x3 then temporal 3x1x1; C-wide R(2+1)D-inspired factorization."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.spatial = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                                     nn.GroupNorm(1, channels), nn.GELU())
        self.conv = nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0), bias=False)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: Tensor, batch: int, frames: int) -> Tensor:
        x = self.spatial(x)
        _, c, h, w = x.shape
        x = self.conv(x.reshape(batch, frames, c, h, w).transpose(1, 2))
        return F.gelu(self.norm(x.transpose(1, 2).reshape(batch * frames, c, h, w)))
