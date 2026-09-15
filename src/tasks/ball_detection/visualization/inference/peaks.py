"""Peak decoding for ball-detection inference in original image pixels.

Heatmap decoding goes through the repository helpers rather than a UI-local
argmax: ``heatmaps_to_peaks`` applies the canonical threshold + NMS + top-k
selection and ``refine_peaks_log_parabolic`` removes lattice quantisation when
the checkpoint asks for sub-cell peaks.  Multiple peaks are kept, so a frame
with several labelled balls can show several predictions.

The model sees frames resized to the checkpoint's ``data.image_size``, so
normalized peak coordinates are scaled by the *original* frame size minus one;
that is the same conversion ``BallDetectionMetrics`` performs when it scores
predictions in original-image pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from src.utils.data.heatmaps import heatmaps_to_peaks, refine_peaks_log_parabolic


@dataclass(frozen=True, slots=True)
class FramePeaks:
    """Decoded peaks for one frame in original-image pixel coordinates."""

    points: tuple[tuple[float, float], ...]
    scores: tuple[float, ...]


def decode_frame_peaks(
    heatmaps: Tensor,
    *,
    original_size: tuple[int, int],
    threshold: float,
    nms_kernel: int,
    max_peaks: int,
    subpixel_refine: bool,
) -> tuple[FramePeaks, ...]:
    """Decode ``(T, H, W)`` probability heatmaps into per-frame pixel peaks."""
    if heatmaps.ndim != 3:
        raise ValueError(
            f"heatmaps must have shape (T, H, W), got {tuple(heatmaps.shape)}."
        )
    width, height = original_size
    if width <= 0 or height <= 0:
        raise ValueError("original_size must be positive.")

    coords, values, valid = heatmaps_to_peaks(
        heatmaps,
        threshold=threshold,
        nms_kernel=nms_kernel,
        max_peaks=max_peaks,
    )
    if subpixel_refine:
        coords = refine_peaks_log_parabolic(heatmaps, coords)
    coordinates: NDArray[np.float32] = coords.detach().cpu().numpy()
    scores: NDArray[np.float32] = values.detach().cpu().numpy()
    mask: NDArray[np.bool_] = valid.detach().cpu().numpy()

    scale_x = float(max(width - 1, 0))
    scale_y = float(max(height - 1, 0))
    decoded: list[FramePeaks] = []
    for frame_index in range(coordinates.shape[0]):
        selected = mask[frame_index]
        points = tuple(
            (
                float(coordinates[frame_index, peak, 0]) * scale_x,
                float(coordinates[frame_index, peak, 1]) * scale_y,
            )
            for peak in range(coordinates.shape[1])
            if bool(selected[peak])
        )
        frame_scores = tuple(
            float(scores[frame_index, peak])
            for peak in range(scores.shape[1])
            if bool(selected[peak])
        )
        decoded.append(FramePeaks(points=points, scores=frame_scores))
    return tuple(decoded)


def peaks_to_points(peaks: FramePeaks, *, prefix: str = "p") -> list[dict[str, object]]:
    """Shape decoded peaks for the shared frame-layer JSON contract."""
    return [
        {
            "x": x,
            "y": y,
            "label": f"{prefix}{index + 1}",
            "score": score,
            "visible": True,
        }
        for index, ((x, y), score) in enumerate(zip(peaks.points, peaks.scores, strict=True))
    ]


__all__ = ["FramePeaks", "decode_frame_peaks", "peaks_to_points"]
