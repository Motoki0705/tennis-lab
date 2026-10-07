"""Frame-local /16 compression, then two shared 2D/2D/3D blocks to /64.

Only the two final Conv3d layers mix MDD timesteps. Each reads t-1,t,t+1;
composition gives encoder receptive field five. Normalization is voxel-local.
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


class FrameConv(nn.Module):
    """Literal Conv2d, sharing weights across frames without mixing them."""

    def __init__(self, inputs: int, outputs: int, *, kernel: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(inputs, outputs, kernel, stride=stride, padding=kernel // 2)
        self.norm = VoxelNorm(outputs)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        b, c, t, h, w = x.shape
        frames = self.conv(x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w))
        x = frames.reshape(b, t, *frames.shape[1:]).permute(0, 2, 1, 3, 4)
        return self.activation(self.norm(x))


class SpatialReduction(nn.Module):
    def __init__(self, method: str, factor: int = 2) -> None:
        super().__init__()
        if method not in {"average", "unshuffle", "haar"} or factor < 1:
            raise ValueError("Invalid spatial reduction")
        if method == "haar" and factor != 2:
            raise ValueError("Each Haar packet level has factor two")
        self.method, self.factor = method, factor

    def forward(self, x: Tensor) -> Tensor:
        s = self.factor
        if self.method == "average":
            return F.avg_pool3d(x, (1, s, s))
        b, c, t, h, w = x.shape
        phases = F.pixel_unshuffle(x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w), s)
        if self.method == "haar":
            phases = phases.reshape(b * t, c, 4, h // 2, w // 2)
            a, b0, c0, d = phases.unbind(2)
            phases = torch.stack((a + b0 + c0 + d, a - b0 + c0 - d,
                                  a + b0 - c0 - d, a - b0 - c0 + d), 2).flatten(1, 2) / 2
        return phases.reshape(b, t, s * s * c, h // s, w // s).permute(0, 2, 1, 3, 4)


class MixedBlock(nn.Sequential):
    """One spatial /2 reduction; temporal mixing occurs only in the last layer."""

    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__(
            FrameConv(inputs, outputs, kernel=3, stride=2),
            FrameConv(outputs, outputs, kernel=3),
            nn.Conv3d(outputs, outputs, 3, stride=1, padding=1),
            VoxelNorm(outputs), nn.GELU(),
        )


class MDDTokenEncoder(nn.Module):
    """B,2,T,H,W -> B,T,ceil(H/64)*ceil(W/64),D; no time subsampling."""

    def __init__(
        self, method: str, stem_channels: tuple[int, int, int, int],
        mixed_channels: tuple[int, int], dim: int,
    ) -> None:
        super().__init__()
        if method not in {"conv2d", "average", "unshuffle", "haar"}:
            raise ValueError("Unsupported spatial reduction")
        if method == "conv2d":
            stages: list[nn.Module] = []
            previous = 2
            for width in stem_channels:
                stages.append(FrameConv(previous, width, kernel=3, stride=2))
                previous = width
            self.stem = nn.Sequential(*stages)
        else:
            # /16 methods operate on original MDD before any learned projection.
            reduction = (nn.Sequential(*(SpatialReduction("haar") for _ in range(4)))
                         if method == "haar" else SpatialReduction(method, 16))
            inputs = 2 if method == "average" else 2 * 16**2
            self.stem = nn.Sequential(reduction, FrameConv(inputs, stem_channels[-1], kernel=1))
        self.mixed = nn.Sequential(
            MixedBlock(stem_channels[-1], mixed_channels[0]),
            MixedBlock(mixed_channels[0], mixed_channels[1]),
        )
        self.project = nn.Conv3d(mixed_channels[-1], dim, 1)
        self.spatial_position = nn.Linear(2, dim)

    def forward(self, mdd: Tensor) -> Tensor:
        height, width = mdd.shape[-2:]
        # Only the /16 reductions require divisible dimensions. Later Conv2d
        # strides use their ordinary padding, so 720p needs no input resize/pad.
        padded = F.pad(mdd, (0, (-width) % 16, 0, (-height) % 16))
        x = self.project(self.mixed(self.stem(padded)))
        h, w = x.shape[-2:]
        # Every output cell overlaps the real image. The final partial cell uses
        # the centre of its real support, not coordinates of padded pixels.
        y0 = torch.arange(h, device=x.device, dtype=x.dtype) * 64
        x0 = torch.arange(w, device=x.device, dtype=x.dtype) * 64
        yc = (y0 + (y0 + 63).clamp_max(height - 1)) / (2 * (height - 1))
        xc = (x0 + (x0 + 63).clamp_max(width - 1)) / (2 * (width - 1))
        yy, xx = torch.meshgrid(yc, xc, indexing="ij")
        position = self.spatial_position(torch.stack((xx, yy), -1).reshape(h * w, 2))
        return x.flatten(-2).permute(0, 2, 3, 1) + position[None, None]
