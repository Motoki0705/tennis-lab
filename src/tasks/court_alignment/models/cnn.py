"""Small full-resolution CNN for multi-court keypoint alignment.

The model deliberately has a very small public contract: a one-channel
ground-plane evidence image is converted to fourteen keypoint logits and one
shared two-channel center-vote field.  The coordinate grid is concatenated
inside the model so callers cannot accidentally omit the UV coordinates.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

NUM_KEYPOINTS = 14
NUM_CENTER_VOTE_CHANNELS = 2
NUM_ENCODER_DOWNSAMPLES = 4
RECEPTIVE_FIELD_PX = 221


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

    The encoder has four stride-two stages (full, half, quarter, eighth,
    sixteenth resolution).  Two bridge blocks at the 1/16 map provide a
    receptive field of 221 input pixels with the default 3x3 convolutions,
    covering the corner-to-centre distance of the 256-pixel training canvas.
    """

    def __init__(
        self,
        *,
        base_channels: int = 24,
        group_norm_groups: int = 8,
        num_keypoints: int = NUM_KEYPOINTS,
        heatmap_prior_probability: float = 0.1,
    ) -> None:
        super().__init__()
        if num_keypoints != NUM_KEYPOINTS:
            raise ValueError("Court alignment currently requires exactly fourteen keypoints.")
        _validate_channels(base_channels, name="base_channels")
        if type(group_norm_groups) is not int or group_norm_groups <= 0:
            raise ValueError("group_norm_groups must be a positive integer.")
        if not math.isfinite(float(heatmap_prior_probability)) or not 0.0 < heatmap_prior_probability < 1.0:
            raise ValueError("heatmap_prior_probability must be finite and in (0,1).")
        self.in_channels = 1
        self.num_keypoints = NUM_KEYPOINTS
        self.num_downsamples = NUM_ENCODER_DOWNSAMPLES
        self.receptive_field_px = RECEPTIVE_FIELD_PX
        self.base_channels = base_channels
        self.heatmap_prior_probability = float(heatmap_prior_probability)

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
        self.down3 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(base_channels * 8, group_norm_groups), base_channels * 8),
            nn.SiLU(inplace=True),
            _ConvNormAct(base_channels * 8, base_channels * 8, groups=group_norm_groups),
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 16, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(base_channels * 16, group_norm_groups), base_channels * 16),
            nn.SiLU(inplace=True),
            _ConvNormAct(base_channels * 16, base_channels * 16, groups=group_norm_groups),
        )
        self.bridge = nn.Sequential(
            _ConvBlock(base_channels * 16, base_channels * 16, groups=group_norm_groups),
            _ConvBlock(base_channels * 16, base_channels * 16, groups=group_norm_groups),
        )
        self.decode3 = _ConvBlock(base_channels * 16 + base_channels * 8, base_channels * 8, groups=group_norm_groups)
        self.decode2 = _ConvBlock(base_channels * 8 + base_channels * 4, base_channels * 4, groups=group_norm_groups)
        self.decode1 = _ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2, groups=group_norm_groups)
        self.decode0 = _ConvBlock(base_channels * 2 + base_channels, base_channels, groups=group_norm_groups)
        self.head = nn.Sequential(
            _ConvNormAct(base_channels, base_channels, groups=group_norm_groups),
            nn.Conv2d(base_channels, NUM_KEYPOINTS + NUM_CENTER_VOTE_CHANNELS, 1),
        )
        output_projection = self.head[-1]
        if not isinstance(output_projection, nn.Conv2d):  # pragma: no cover - construction invariant
            raise RuntimeError("Court alignment output projection must be a Conv2d.")
        if output_projection.bias is None:  # pragma: no cover - construction invariant
            raise RuntimeError("Court alignment output projection requires a bias.")
        prior_logit = math.log(self.heatmap_prior_probability / (1.0 - self.heatmap_prior_probability))
        with torch.no_grad():
            output_projection.bias[:NUM_KEYPOINTS].fill_(prior_logit)
            output_projection.bias[NUM_KEYPOINTS:].zero_()

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
        grid = self._coordinates(x)
        stem = self.stem(torch.cat((x, grid), dim=1))
        skip1 = self.down1(stem)
        skip2 = self.down2(skip1)
        skip3 = self.down3(skip2)
        skip4 = self.down4(skip3)
        bridge = self.bridge(skip4)
        decoded3 = F.interpolate(bridge, size=skip3.shape[-2:], mode="bilinear", align_corners=False)
        decoded3 = self.decode3(torch.cat((decoded3, skip3), dim=1))
        decoded2 = F.interpolate(decoded3, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        decoded2 = self.decode2(torch.cat((decoded2, skip2), dim=1))
        decoded1 = F.interpolate(decoded2, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        decoded1 = self.decode1(torch.cat((decoded1, skip1), dim=1))
        decoded0 = F.interpolate(decoded1, size=stem.shape[-2:], mode="bilinear", align_corners=False)
        decoded0 = self.decode0(torch.cat((decoded0, stem), dim=1))
        output = self.head(decoded0)
        return CourtAlignmentModelOutput(
            heatmap_logits=output[:, :NUM_KEYPOINTS],
            center_votes=output[:, NUM_KEYPOINTS:],
        )


def validate_court_alignment_input(
    value: object,
    *,
    expected_dtype: torch.dtype | None = None,
) -> Tensor:
    """Validate an input tensor at an inference or training boundary."""
    if not isinstance(value, Tensor):
        raise TypeError("Court alignment input must be a torch.Tensor.")
    if value.ndim != 4:
        raise ValueError(
            "Court alignment input must have shape (B,1,H,W), "
            f"got {tuple(value.shape)}."
        )
    if value.shape[0] <= 0 or value.shape[1] != 1 or any(
        size <= 0 for size in value.shape[2:]
    ):
        raise ValueError(
            "Court alignment input must have shape (B,1,H,W), "
            f"got {tuple(value.shape)}."
        )
    if not value.is_floating_point():
        raise TypeError("Court alignment input must be floating point.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Court alignment input must contain only finite values.")
    if bool(torch.any((value < 0.0) | (value > 1.0))):
        raise ValueError("Court alignment input values must lie in [0,1].")
    if expected_dtype is not None and value.dtype != expected_dtype:
        raise TypeError(
            "Court alignment input dtype must match model dtype "
            f"{expected_dtype}, got {value.dtype}."
        )
    return value


def validate_court_alignment_output(value: object) -> CourtAlignmentModelOutput:
    """Validate model outputs after a forward call at a runtime boundary."""
    if isinstance(value, CourtAlignmentModelOutput):
        result = value
    elif isinstance(value, Mapping):
        try:
            heatmaps = value["heatmap_logits"]
            votes = value["center_votes"]
        except KeyError as error:
            raise ValueError(
                "Model mapping must contain heatmap_logits and center_votes."
            ) from error
        if not isinstance(heatmaps, Tensor) or not isinstance(votes, Tensor):
            raise TypeError("Court-alignment model mapping values must be tensors.")
        result = CourtAlignmentModelOutput(heatmap_logits=heatmaps, center_votes=votes)
    else:
        raise TypeError("Court-alignment model must return a tensor mapping.")
    heatmaps = result.heatmap_logits
    votes = result.center_votes
    if heatmaps.ndim != 4 or any(size <= 0 for size in heatmaps.shape):
        raise ValueError("heatmap_logits must have shape (B,14,H,W).")
    if heatmaps.shape[1] != NUM_KEYPOINTS:
        raise ValueError("heatmap_logits must have fourteen channels.")
    if votes.ndim != 4 or votes.shape[1] != NUM_CENTER_VOTE_CHANNELS:
        raise ValueError("center_votes must have shape (B,2,H,W).")
    if votes.shape[0] != heatmaps.shape[0] or votes.shape[2:] != heatmaps.shape[2:]:
        raise ValueError(
            "heatmap_logits and center_votes must share batch and spatial shape."
        )
    if votes.device != heatmaps.device:
        raise ValueError("heatmap_logits and center_votes must share a device.")
    for name, output in (("heatmap_logits", heatmaps), ("center_votes", votes)):
        if not output.is_floating_point():
            raise TypeError(f"{name} must be floating point.")
    return result


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
    "NUM_ENCODER_DOWNSAMPLES",
    "NUM_KEYPOINTS",
    "RECEPTIVE_FIELD_PX",
    "validate_court_alignment_input",
    "validate_court_alignment_output",
]
