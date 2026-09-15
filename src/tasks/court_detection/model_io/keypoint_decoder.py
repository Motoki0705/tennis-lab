"""Explicit Court keypoint heatmap decoding contracts.

Turning Court KP logits into typed sparse candidates is a model-I/O concern,
so this module lives beside the adapter that produces and validates those
logits rather than inside the predictor package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import torch
from torch import Tensor

from src.tasks.court_detection.model_io.contracts import (
    CourtKeypointPrediction,
    CourtModelIOError,
)
from src.utils.data.heatmaps import heatmaps_to_peaks, refine_peaks_log_parabolic


@dataclass(frozen=True, slots=True)
class CourtKeypointDecoderConfig:
    """Configuration for converting Court KP logits into sparse candidates.

    The default contract is the ordered single-court KP14 task: each semantic
    channel emits at most one local maximum. The low extraction threshold keeps
    score calibration separate from downstream geometric filtering; callers
    can raise it explicitly. Multi-court callers must opt in to additional
    candidates by setting ``max_peaks`` explicitly.

    Fields are validated as their declared types, so a config never silently
    accepts a coerced value such as ``"0.05"`` or ``True``.
    """

    threshold: float = 0.05
    nms_kernel: int = 7
    max_peaks: int = 1

    def __post_init__(self) -> None:
        _require_unit_interval_threshold(self.threshold)
        _require_positive_int(self.nms_kernel, name="nms_kernel")
        if self.nms_kernel % 2 == 0:
            raise ValueError("Court keypoint nms_kernel must be a positive odd integer.")
        _require_positive_int(self.max_peaks, name="max_peaks")


def decode_court_keypoint_logits(
    logits: Tensor,
    *,
    original_size_hw: tuple[int, int],
    subpixel_refine: bool,
    config: CourtKeypointDecoderConfig,
) -> CourtKeypointPrediction:
    """Decode one image of Court KP logits under an explicit peak contract.

    Args:
        logits: Floating logits with shape ``(1, C, H, W)`` and finite values.
        original_size_hw: ``(height, width)`` of the image the logits describe,
            as two positive ints.
        subpixel_refine: Whether to refine lattice peaks to sub-cell precision.
        config: The peak threshold, NMS kernel, and candidate budget.

    Returns:
        Typed KP prediction with ``[C, P, *]`` peak-axis tensors on the CPU.

    Raises:
        CourtModelIOError: The logits or original size violate the contract.
    """
    _validate_logits(logits)
    original_height, original_width = _validate_original_size_hw(original_size_hw)

    probability = torch.sigmoid(logits)
    coords, scores, valid = heatmaps_to_peaks(
        probability,
        threshold=config.threshold,
        nms_kernel=config.nms_kernel,
        max_peaks=config.max_peaks,
    )
    if subpixel_refine:
        coords = refine_peaks_log_parabolic(probability, coords)

    scale = coords.new_tensor(
        [float(max(original_width - 1, 0)), float(max(original_height - 1, 0))]
    )
    return CourtKeypointPrediction(
        keypoints=(coords[0] * scale).cpu(),
        scores=scores[0].cpu(),
        valid=valid[0].cpu(),
        heatmaps=logits[0].cpu(),
    )


def _require_unit_interval_threshold(value: object) -> None:
    """Require a real, finite threshold inside ``[0, 1]``.

    ``value`` is typed as ``object`` so the runtime check also covers callers
    that bypass the annotation, e.g. by coercing a config from raw YAML.
    """

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            f"Court keypoint threshold must be a real number, got {type(value).__name__}."
        )
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError("Court keypoint threshold must be finite and in [0, 1].")


def _require_positive_int(value: object, *, name: str) -> None:
    if type(value) is not int:
        raise TypeError(
            f"Court keypoint {name} must be an int, got {type(value).__name__}."
        )
    if value <= 0:
        raise ValueError(f"Court keypoint {name} must be positive.")


def _validate_logits(logits: object) -> None:
    if not isinstance(logits, Tensor):
        raise CourtModelIOError(
            "Court keypoint logits must be a Tensor, got "
            f"{type(logits).__name__}."
        )
    if logits.ndim != 4 or logits.shape[0] != 1:
        raise CourtModelIOError(
            "Court keypoint decoding requires logits with shape (1,C,H,W), got "
            f"{tuple(logits.shape)}."
        )
    _, channels, height, width = logits.shape
    if channels <= 0 or height <= 0 or width <= 0:
        raise CourtModelIOError(
            "Court keypoint logits must have positive channel and spatial sizes, "
            f"got C={channels}, H={height}, W={width}."
        )
    if not logits.is_floating_point():
        raise CourtModelIOError(
            f"Court keypoint logits must be floating point, got {logits.dtype}."
        )
    if not bool(torch.isfinite(logits).all()):
        raise CourtModelIOError("Court keypoint logits must contain only finite values.")


def _validate_original_size_hw(original_size_hw: object) -> tuple[int, int]:
    if not isinstance(original_size_hw, tuple) or len(original_size_hw) != 2:
        raise CourtModelIOError(
            "Court keypoint original_size_hw must be a (height, width) tuple with "
            f"exactly two ints, got {original_size_hw!r}."
        )
    height, width = original_size_hw
    if type(height) is not int or type(width) is not int:
        raise CourtModelIOError(
            "Court keypoint original_size_hw values must be ints, got "
            f"{(type(height).__name__, type(width).__name__)}."
        )
    if height <= 0 or width <= 0:
        raise CourtModelIOError("Court keypoint original image size must be positive.")
    return height, width


__all__ = ["CourtKeypointDecoderConfig", "decode_court_keypoint_logits"]
