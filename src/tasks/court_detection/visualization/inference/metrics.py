"""Task-specific, schema-checked metrics for Court review inference.

Only layers whose supervision schema matches between the checkpoint bundle and
the reviewed dataset are scored; the caller guarantees that match before calling
here, and every comparison resizes the prediction to the ground-truth grid with
an explicit nearest-neighbour resize recorded in the result.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from src.tasks.court_detection.model_io.contracts import (
    CourtDecodedPrediction,
    CourtKeypointPrediction,
    CourtLinePrediction,
    CourtSegmentationPrediction,
)


def resize_labels_nearest(
    labels: NDArray[np.integer[object]], size_hw: tuple[int, int]
) -> NDArray[np.int64]:
    """Resize a categorical map to ``size_hw`` with nearest-neighbour sampling."""
    array = np.asarray(labels)
    if array.shape == size_hw:
        return array.astype(np.int64, copy=False)
    image = Image.fromarray(array.astype(np.int32), mode="I")
    resized = image.resize((size_hw[1], size_hw[0]), Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.int64)


def resize_probability_bilinear(
    probability: NDArray[np.floating[object]], size_hw: tuple[int, int]
) -> NDArray[np.float64]:
    """Resize a ``[0, 1]`` probability map to ``size_hw`` by bilinear sampling."""
    array = np.asarray(probability, dtype=np.float32)
    if array.shape == size_hw:
        return array.astype(np.float64)
    image = Image.fromarray(array, mode="F")
    resized = image.resize((size_hw[1], size_hw[0]), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float64)


def keypoint_metrics(
    prediction: CourtKeypointPrediction,
    *,
    ground_truth_xy: NDArray[np.float64],
    ground_truth_visible: NDArray[np.bool_],
    physical_indices: NDArray[np.int64],
) -> dict[str, object]:
    """Score single-peak channels against visible ground-truth points in pixels.

    ``ground_truth_xy`` has shape ``(C, 2)`` and ``physical_indices`` ``(C,)``;
    each channel must correspond to exactly one physical Ground Court point.
    """
    keypoints = prediction.keypoints.detach().cpu().numpy()
    valid = prediction.valid.detach().cpu().numpy()
    scores = prediction.scores.detach().cpu().numpy()
    channels = ground_truth_xy.shape[0]
    if keypoints.shape[0] != channels or physical_indices.shape != (channels,):
        raise ValueError(
            "Court keypoint metrics require one ground-truth point per channel."
        )
    distances: list[float] = []
    predicted = 0
    for channel in range(channels):
        if valid.shape[1] == 0:
            continue
        usable = np.flatnonzero(valid[channel])
        if usable.size == 0:
            continue
        peak = int(usable[np.argmax(scores[channel][usable])])
        predicted += 1
        if not bool(ground_truth_visible[channel]):
            continue
        delta = keypoints[channel, peak] - ground_truth_xy[channel]
        distances.append(float(np.hypot(float(delta[0]), float(delta[1]))))
    if distances:
        values = np.asarray(distances, dtype=np.float64)
        mean_error: float | None = float(values.mean())
        median_error: float | None = float(np.median(values))
        p90_error: float | None = float(np.percentile(values, 90.0))
    else:
        mean_error = None
        median_error = None
        p90_error = None
    return {
        "kp_predicted_channels": predicted,
        "kp_scored_points": len(distances),
        "kp_mean_error_px": mean_error,
        "kp_median_error_px": median_error,
        "kp_p90_error_px": p90_error,
    }


def categorical_metrics(
    prediction: CourtSegmentationPrediction,
    *,
    ground_truth: NDArray[np.integer[object]],
    prefix: str,
) -> dict[str, object]:
    """Score a categorical head by pixel accuracy and mean IoU over classes."""
    size_hw = (int(ground_truth.shape[0]), int(ground_truth.shape[1]))
    predicted = resize_labels_nearest(prediction.mask.detach().cpu().numpy(), size_hw)
    truth = np.asarray(ground_truth, dtype=np.int64)
    if predicted.shape != truth.shape:
        raise ValueError("Court categorical metrics require matching mask shapes.")
    accuracy = float((predicted == truth).mean()) if truth.size else None
    ious: list[float] = []
    for label in sorted(
        set(np.unique(truth).tolist()) | set(np.unique(predicted).tolist())
    ):
        if label == 0:
            continue
        truth_mask = truth == label
        predicted_mask = predicted == label
        union = int(np.count_nonzero(truth_mask | predicted_mask))
        if union == 0:
            continue
        ious.append(float(np.count_nonzero(truth_mask & predicted_mask)) / union)
    return {
        f"{prefix}_pixel_accuracy": accuracy,
        f"{prefix}_mean_iou": (float(np.mean(ious)) if ious else None),
        f"{prefix}_foreground_iou_classes": len(ious),
    }


def line_metrics(
    prediction: CourtLinePrediction,
    *,
    ground_truth: NDArray[np.bool_],
    threshold: float,
) -> dict[str, object]:
    """Score the binary line head by intersection-over-union and Dice."""
    size_hw = (int(ground_truth.shape[0]), int(ground_truth.shape[1]))
    probability = resize_probability_bilinear(
        prediction.probability.detach().cpu().numpy(), size_hw
    )
    predicted = probability >= float(threshold)
    truth = np.asarray(ground_truth, dtype=bool)
    if predicted.shape != truth.shape:
        raise ValueError("Court line metrics require matching mask shapes.")
    intersection = int(np.count_nonzero(predicted & truth))
    union = int(np.count_nonzero(predicted | truth))
    predicted_total = int(np.count_nonzero(predicted))
    truth_total = int(np.count_nonzero(truth))
    iou = float(intersection) / union if union else None
    dice = (
        float(2 * intersection) / (predicted_total + truth_total)
        if (predicted_total + truth_total)
        else None
    )
    return {
        "line_iou": iou,
        "line_dice": dice,
        "line_predicted_pixels": predicted_total,
        "line_ground_truth_pixels": truth_total,
    }


def resize_note(
    prediction: NDArray[np.integer[object]], ground_truth: NDArray[np.integer[object]]
) -> str | None:
    """Describe a prediction/ground-truth grid mismatch for the warnings list."""
    predicted_hw = (int(prediction.shape[0]), int(prediction.shape[1]))
    truth_hw = (int(ground_truth.shape[0]), int(ground_truth.shape[1]))
    if predicted_hw == truth_hw:
        return None
    return (
        f"予測 grid {predicted_hw[1]}x{predicted_hw[0]} を採点のため GT grid "
        f"{truth_hw[1]}x{truth_hw[0]} へ再標本化しました。"
    )


def decoded_mask_shape(prediction: CourtDecodedPrediction) -> tuple[int, int]:
    """Return the prediction grid of one decoded head."""
    if isinstance(prediction, CourtSegmentationPrediction):
        return (int(prediction.mask.shape[0]), int(prediction.mask.shape[1]))
    if isinstance(prediction, CourtLinePrediction):
        return (
            int(prediction.probability.shape[0]),
            int(prediction.probability.shape[1]),
        )
    return (0, 0)


__all__ = [
    "categorical_metrics",
    "decoded_mask_shape",
    "keypoint_metrics",
    "line_metrics",
    "resize_labels_nearest",
    "resize_note",
    "resize_probability_bilinear",
]
