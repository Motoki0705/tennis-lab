"""One-channel court evidence adapters for the official DINO detector.

All modes operate on an unnormalised heatmap in ``[0, 1]``.  The heatmap is
first resized with the aspect-ratio-preserving transform used by the released
DINO COCO detector, then mapped to three channels, and only then normalised by
the ImageNet RGB mean and standard deviation.  Keeping this boundary here
prevents data pipelines from accidentally normalising the learnable adapter's
input twice.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

DINO_DEFAULT_SHORT_SIDE = 800
DINO_DEFAULT_MAX_LONG_SIDE = 1333
IMAGENET_RGB_MEAN = (0.485, 0.456, 0.406)
IMAGENET_RGB_STD = (0.229, 0.224, 0.225)

DinoInputMode = Literal["repeat_rgb", "learnable_1x1", "red_only"]
_INPUT_MODES = frozenset(("repeat_rgb", "learnable_1x1", "red_only"))


def _validate_resize_limits(short_side: int, max_long_side: int) -> None:
    if type(short_side) is not int or type(max_long_side) is not int:
        raise TypeError("short_side and max_long_side must be integers.")
    if short_side <= 0 or max_long_side < short_side:
        raise ValueError(
            "Expected 0 < short_side <= max_long_side, got "
            f"{short_side} and {max_long_side}."
        )


def dino_resize_shape(
    height: int,
    width: int,
    *,
    short_side: int,
    max_long_side: int,
) -> tuple[int, int]:
    """Return DINO's aspect-ratio-preserving detector input shape."""
    _validate_resize_limits(short_side, max_long_side)
    if type(height) is not int or type(width) is not int:
        raise TypeError("height and width must be integers.")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive.")
    return _compute_dino_resize_shape(
        height,
        width,
        short_side=short_side,
        max_long_side=max_long_side,
    )


def _compute_dino_resize_shape(
    height: int,
    width: int,
    *,
    short_side: int,
    max_long_side: int,
) -> tuple[int, int]:
    """Compute the resize shape after the public boundary has validated it."""

    scale = short_side / min(height, width)
    if max(height, width) * scale > max_long_side:
        scale = max_long_side / max(height, width)
    return int(round(height * scale)), int(round(width * scale))


def validate_dino_heatmaps(heatmaps: Tensor) -> None:
    """Validate raw one-channel DINO evidence outside model ``forward``."""

    if not isinstance(heatmaps, Tensor):
        raise TypeError("DINO heatmap input must be a torch.Tensor.")
    if heatmaps.ndim != 4 or heatmaps.shape[1] != 1 or heatmaps.shape[0] <= 0:
        raise ValueError(
            "DINO heatmap input must have shape (B,1,H,W), got "
            f"{tuple(heatmaps.shape)}."
        )
    if not heatmaps.is_floating_point():
        raise TypeError("DINO heatmap input must be floating point.")
    if any(size <= 0 for size in heatmaps.shape[2:]):
        raise ValueError("DINO heatmap spatial dimensions must be positive.")
    if not bool(torch.isfinite(heatmaps).all()):
        raise ValueError("DINO heatmap input must contain only finite values.")
    if bool(torch.any((heatmaps < 0.0) | (heatmaps > 1.0))):
        raise ValueError("DINO heatmap input values must lie in [0, 1].")


class DinoHeatmapInputAdapter(nn.Module):
    """Convert ``(B,1,H,W)`` court heatmaps into normalised DINO RGB input.

    ``repeat_rgb`` duplicates the heatmap in R/G/B. ``learnable_1x1`` uses a
    trainable ``Conv2d(1, 3, 1)`` initialised to exactly that duplication.
    ``red_only`` places evidence in R while setting G/B to zero.  ImageNet
    normalisation follows channel construction, so a zero G/B channel becomes
    ``-mean / std`` rather than zero at the detector boundary.
    """

    def __init__(
        self,
        *,
        mode: DinoInputMode,
        short_side: int = DINO_DEFAULT_SHORT_SIDE,
        max_long_side: int = DINO_DEFAULT_MAX_LONG_SIDE,
    ) -> None:
        super().__init__()
        if mode not in _INPUT_MODES:
            raise ValueError(
                f"Unsupported DINO input mode {mode!r}; expected one of "
                f"{sorted(_INPUT_MODES)}."
            )
        _validate_resize_limits(short_side, max_long_side)
        self.mode = mode
        self.short_side = short_side
        self.max_long_side = max_long_side
        if mode == "learnable_1x1":
            projection = nn.Conv2d(1, 3, kernel_size=1, bias=True)
            with torch.no_grad():
                projection.weight.fill_(1.0)
                if projection.bias is None:  # pragma: no cover - construction invariant
                    raise RuntimeError("learnable_1x1 projection requires a bias.")
                projection.bias.zero_()
            self.projection: nn.Conv2d | None = projection
        else:
            self.projection = None
        self.imagenet_mean: Tensor
        self.register_buffer(
            "imagenet_mean",
            torch.tensor(IMAGENET_RGB_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.imagenet_std: Tensor
        self.register_buffer(
            "imagenet_std",
            torch.tensor(IMAGENET_RGB_STD).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, heatmaps: Tensor) -> Tensor:
        target_shape = _compute_dino_resize_shape(
            heatmaps.shape[-2],
            heatmaps.shape[-1],
            short_side=self.short_side,
            max_long_side=self.max_long_side,
        )
        resized = F.interpolate(
            heatmaps,
            size=target_shape,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        if self.mode == "repeat_rgb":
            rgb = resized.repeat(1, 3, 1, 1)
        elif self.mode == "red_only":
            rgb = torch.cat((resized, torch.zeros_like(resized).repeat(1, 2, 1, 1)), dim=1)
        else:
            rgb = cast(nn.Conv2d, self.projection)(resized)
        result = (rgb - self.imagenet_mean) / self.imagenet_std
        return result

    def validate_input(self, heatmaps: Tensor) -> None:
        """Validate heatmaps before invoking the computation-only ``forward``."""

        validate_dino_heatmaps(heatmaps)


__all__: Sequence[str] = (
    "DINO_DEFAULT_MAX_LONG_SIDE",
    "DINO_DEFAULT_SHORT_SIDE",
    "DinoHeatmapInputAdapter",
    "DinoInputMode",
    "IMAGENET_RGB_MEAN",
    "IMAGENET_RGB_STD",
    "dino_resize_shape",
    "validate_dino_heatmaps",
)
