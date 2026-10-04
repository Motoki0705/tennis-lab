"""Four spatial /8 encoders, all with local temporal receptive field 7.

The feature normalization is channel-only at each voxel. Group/BatchNorm over
time would make the claimed local temporal dependency false during training.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class VoxelNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)


class SpatialReduction(nn.Module):
    def __init__(self, method: str) -> None:
        super().__init__()
        self.method = method

    def forward(self, x: Tensor) -> Tensor:
        if self.method == "average":
            return F.avg_pool3d(x, (1, 2, 2))
        b, c, t, h, w = x.shape
        phases = F.pixel_unshuffle(x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w), 2)
        if self.method == "haar":
            phases = phases.reshape(b * t, c, 4, h // 2, w // 2)
            a, b0, c0, d = phases.unbind(2)
            # Orthonormal 2D Haar, all subbands; divisible input requires no extension.
            phases = torch.stack((a + b0 + c0 + d, a - b0 + c0 - d,
                                  a + b0 - c0 - d, a - b0 - c0 + d), 2).flatten(1, 2) / 2
        return phases.reshape(b, t, 4 * c, h // 2, w // 2).permute(0, 2, 1, 3, 4)


class MDDTokenEncoder(nn.Module):
    """B,2,T,H,W -> B,T,(H/8*W/8),D; time stride is always one."""

    def __init__(self, method: str, channels: tuple[int, int, int], dim: int) -> None:
        super().__init__()
        if method not in {"conv3d", "average", "unshuffle", "haar"}:
            raise ValueError("Unsupported spatial reduction")
        stages: list[nn.Module] = []
        previous = 2
        for width in channels:
            if method == "conv3d":
                conv = nn.Conv3d(previous, width, 3, stride=(1, 2, 2), padding=1)
                stages.append(nn.Sequential(conv, VoxelNorm(width), nn.GELU()))
            else:
                multiplier = 1 if method == "average" else 4
                stages.append(nn.Sequential(
                    SpatialReduction(method),
                    nn.Conv3d(previous * multiplier, width, (3, 1, 1), padding=(1, 0, 0)),
                    VoxelNorm(width), nn.GELU(),
                ))
            previous = width
        self.stages = nn.Sequential(*stages)
        self.project = nn.Conv3d(previous, dim, 1)
        self.spatial_position = nn.Linear(2, dim)

    def forward(self, mdd: Tensor) -> Tensor:
        x = self.project(self.stages(mdd))
        h, w = x.shape[-2:]
        yy, xx = torch.meshgrid(torch.linspace(0, 1, h, device=x.device, dtype=x.dtype),
                                torch.linspace(0, 1, w, device=x.device, dtype=x.dtype), indexing="ij")
        position = self.spatial_position(torch.stack((xx, yy), -1).reshape(h * w, 2))
        return x.flatten(-2).permute(0, 2, 3, 1) + position[None, None]
