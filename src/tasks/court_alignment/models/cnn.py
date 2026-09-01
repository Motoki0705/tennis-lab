"""Small full-resolution CNN for multi-court keypoint alignment.

The model deliberately has a very small public contract: a one-channel
ground-plane evidence image is converted to fourteen keypoint logits and one
shared two-channel center-vote field.  The coordinate grid is concatenated
inside the model so callers cannot accidentally omit the UV coordinates.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

NUM_KEYPOINTS = 14
NUM_CENTER_VOTE_CHANNELS = 2


def _group_count(channels: int, requested: int) -> int:
    """Choose a GroupNorm group count that divides *channels*."""
    return max((groups for groups in range(min(channels, requested), 0, -1) if channels % groups == 0), default=1)


def _validate_channels(value: int, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


class _ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, groups: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(out_channels, groups), out_channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.activation(self.norm(self.conv(x))))


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, groups: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            _ConvNormAct(in_channels, out_channels, groups=groups),
            _ConvNormAct(out_channels, out_channels, groups=groups),
        )

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.layers(x))


@dataclass(frozen=True, slots=True)
class CourtAlignmentModelOutput(Mapping[str, Tensor]):
    """Typed model output, also usable as the historical mapping contract."""

    heatmap_logits: Tensor
    center_votes: Tensor

    def __post_init__(self) -> None:
        if self.heatmap_logits.ndim != 4 or any(size <= 0 for size in self.heatmap_logits.shape):
            raise ValueError("heatmap_logits must have shape (B,14,H,W).")
        if self.heatmap_logits.shape[1] != NUM_KEYPOINTS:
            raise ValueError("heatmap_logits must have fourteen channels.")
        if self.center_votes.ndim != 4:
            raise ValueError("center_votes must have shape (B,2,H,W).")
        if self.center_votes.shape[1] != NUM_CENTER_VOTE_CHANNELS:
            raise ValueError("center_votes must have two channels.")
        if self.center_votes.shape[0] != self.heatmap_logits.shape[0] or self.center_votes.shape[2:] != self.heatmap_logits.shape[2:]:
            raise ValueError("heatmap_logits and center_votes must share batch and spatial shape.")
        if self.center_votes.device != self.heatmap_logits.device:
            raise ValueError("heatmap_logits and center_votes must share a device.")
        for name, value in (("heatmap_logits", self.heatmap_logits), ("center_votes", self.center_votes)):
            if not value.is_floating_point():
                raise TypeError(f"{name} must be floating point.")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain only finite values.")

    def __getitem__(self, key: str) -> Tensor:
        if key == "heatmap_logits":
            return self.heatmap_logits
        if key == "center_votes":
            return self.center_votes
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(("heatmap_logits", "center_votes"))

    def __len__(self) -> int:
        return 2

    @property
    def heatmaps(self) -> Tensor:
        """Short alias used by lightweight inference callers."""
        return self.heatmap_logits


class CourtAlignmentCNN(nn.Module):
    """Lightweight U-Net producing full-resolution KP and vote maps.

    ``center_votes`` are vectors in output-image pixels, ordered ``(dx, dy)``
    from a keypoint pixel to its court center.  Keeping this convention in
    pixels makes the association decoder independent of a particular UV
    normalisation range.
    """

    def __init__(
        self,
        *,
        base_channels: int = 24,
        group_norm_groups: int = 8,
        num_keypoints: int = NUM_KEYPOINTS,
    ) -> None:
        super().__init__()
        if num_keypoints != NUM_KEYPOINTS:
            raise ValueError("Court alignment currently requires exactly fourteen keypoints.")
        _validate_channels(base_channels, name="base_channels")
        if type(group_norm_groups) is not int or group_norm_groups <= 0:
            raise ValueError("group_norm_groups must be a positive integer.")
        self.in_channels = 1
        self.num_keypoints = NUM_KEYPOINTS
        self.base_channels = base_channels

        # The first layer sees evidence, U, and V.  All subsequent layers are
        # ordinary 2-D convolutions, retaining the strong spatial inductive bias.
        self.stem = _ConvBlock(3, base_channels, groups=group_norm_groups)
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(base_channels * 2, group_norm_groups), base_channels * 2),
            nn.SiLU(inplace=True),
            _ConvNormAct(base_channels * 2, base_channels * 2, groups=group_norm_groups),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(base_channels * 4, group_norm_groups), base_channels * 4),
            nn.SiLU(inplace=True),
            _ConvNormAct(base_channels * 4, base_channels * 4, groups=group_norm_groups),
        )
        self.bridge = _ConvBlock(base_channels * 4, base_channels * 8, groups=group_norm_groups)
        self.decode2 = _ConvBlock(base_channels * 8 + base_channels * 2, base_channels * 2, groups=group_norm_groups)
        self.decode1 = _ConvBlock(base_channels * 2 + base_channels, base_channels, groups=group_norm_groups)
        self.head = nn.Sequential(
            _ConvNormAct(base_channels, base_channels, groups=group_norm_groups),
            nn.Conv2d(base_channels, NUM_KEYPOINTS + NUM_CENTER_VOTE_CHANNELS, 1),
        )

    @staticmethod
    def _validate_input(x: Tensor) -> None:
        if not isinstance(x, Tensor):
            raise TypeError("Court alignment input must be a torch.Tensor.")
        if x.ndim != 4:
            raise ValueError(f"Court alignment input must have shape (B,1,H,W), got {tuple(x.shape)}.")
        if x.shape[0] <= 0 or x.shape[1] != 1 or any(size <= 0 for size in x.shape[2:]):
            raise ValueError(f"Court alignment input must have shape (B,1,H,W), got {tuple(x.shape)}.")
        if not x.is_floating_point():
            raise TypeError("Court alignment input must be floating point.")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("Court alignment input must contain only finite values.")

    @staticmethod
    def _coordinates(x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        # Normalised coordinates are centred at zero; this is a stable input
        # for both square and rectangular UV canvases.
        ys = torch.linspace(-1.0, 1.0, height, dtype=x.dtype, device=x.device)
        xs = torch.linspace(-1.0, 1.0, width, dtype=x.dtype, device=x.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack((xx, yy), dim=0).unsqueeze(0).expand(x.shape[0], -1, -1, -1)

    def forward(self, x: Tensor) -> CourtAlignmentModelOutput:
        self._validate_input(x)
        parameter = next(self.parameters())
        if x.dtype != parameter.dtype:
            raise TypeError(f"Court alignment input dtype must match model dtype {parameter.dtype}, got {x.dtype}.")
        grid = self._coordinates(x)
        stem = self.stem(torch.cat((x, grid), dim=1))
        skip1 = self.down1(stem)
        skip2 = self.down2(skip1)
        bridge = self.bridge(skip2)
        decoded2 = F.interpolate(bridge, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        decoded2 = self.decode2(torch.cat((decoded2, skip1), dim=1))
        decoded1 = F.interpolate(decoded2, size=stem.shape[-2:], mode="bilinear", align_corners=False)
        decoded1 = self.decode1(torch.cat((decoded1, stem), dim=1))
        output = self.head(decoded1)
        return CourtAlignmentModelOutput(
            heatmap_logits=output[:, :NUM_KEYPOINTS],
            center_votes=output[:, NUM_KEYPOINTS:],
        )


# Stable aliases for configuration factories and downstream code.
CourtAlignmentModel = CourtAlignmentCNN
CourtAlignmentOutput = CourtAlignmentModelOutput
CourtAlignmentKP14CNN = CourtAlignmentCNN


__all__ = [
    "CourtAlignmentCNN",
    "CourtAlignmentKP14CNN",
    "CourtAlignmentModel",
    "CourtAlignmentModelOutput",
    "CourtAlignmentOutput",
    "NUM_CENTER_VOTE_CHANNELS",
    "NUM_KEYPOINTS",
]
