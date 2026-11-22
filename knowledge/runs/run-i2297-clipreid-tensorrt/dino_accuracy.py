"""Compare decoded DINO person boxes for one fixed input image."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.utils.geometry.bbox import pairwise_iou

MIN_IOU = 0.5


def compare_dino_frame(
    reference_boxes, reference_scores, candidate_boxes, candidate_scores
):
    """Match as many eligible boxes as possible, breaking ties by total IoU."""
    reference_boxes = np.asarray(reference_boxes, dtype=np.float64)
    candidate_boxes = np.asarray(candidate_boxes, dtype=np.float64)
    reference_scores = np.asarray(reference_scores, dtype=np.float64)
    candidate_scores = np.asarray(candidate_scores, dtype=np.float64)
    if not all(
        np.isfinite(array).all()
        for array in (
            reference_boxes,
            reference_scores,
            candidate_boxes,
            candidate_scores,
        )
    ):
        raise ValueError("Nonfinite DINO box coordinates or scores")
    iou = pairwise_iou(reference_boxes, candidate_boxes)
    if reference_scores.shape != (len(reference_boxes),) or candidate_scores.shape != (
        len(candidate_boxes),
    ):
        raise ValueError("Expected one score for each DINO box")
    eligible = iou >= MIN_IOU
    # One additional match is worth more than the total IoU of any assignment.
    weights = np.where(eligible, min(iou.shape) + 1 + iou, 0.0)
    rows, cols = linear_sum_assignment(weights, maximize=True)
    keep = eligible[rows, cols]
    rows, cols = rows[keep], cols[keep]
    return {
        "iou": iou[rows, cols],
        "xyxy_abs_error_px": np.abs(reference_boxes[rows] - candidate_boxes[cols]),
        "score_abs_error": np.abs(reference_scores[rows] - candidate_scores[cols]),
        "missing": len(reference_boxes) - len(rows),
        "extra": len(candidate_boxes) - len(cols),
    }
