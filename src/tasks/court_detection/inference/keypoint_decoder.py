"""Explicit Court keypoint heatmap decoding contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass

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
    """

    threshold: float = 0.05
    nms_kernel: int = 7
    max_peaks: int = 1

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Court keypoint threshold must be finite and in [0, 1].")
        if self.nms_kernel <= 0 or self.nms_kernel % 2 == 0:
            raise ValueError(
                "Court keypoint nms_kernel must be a positive odd integer."
            )
        if self.max_peaks <= 0:
            raise ValueError("Court keypoint max_peaks must be positive.")


def decode_court_keypoint_logits(
    logits: Tensor,
    *,
    original_size_hw: tuple[int, int],
    subpixel_refine: bool,
    config: CourtKeypointDecoderConfig,
) -> CourtKeypointPrediction:
    """Decode one image of Court KP logits under an explicit peak contract."""
    if logits.ndim != 4 or logits.shape[0] != 1:
        raise CourtModelIOError(
            "Court keypoint decoding requires logits with shape (1,C,H,W)."
        )
    original_height, original_width = original_size_hw
    if original_height <= 0 or original_width <= 0:
        raise CourtModelIOError("Court keypoint original image size must be positive.")

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


__all__ = ["CourtKeypointDecoderConfig", "decode_court_keypoint_logits"]
