"""Confidence-ranked PROSAC followed by an explicit inlier-only court fit."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

PROSAC_OPTIONS: dict[str, int | float | bool] = {
    "sampler": cv2.SAMPLING_PROSAC,
    "score": cv2.SCORE_METHOD_MSAC,
    "confidence": 0.999,
    "maxIterations": 10000,
    "randomGeneratorState": 42,
    "isParallel": False,
    "loMethod": cv2.LOCAL_OPTIM_INNER_LO,
    "loIterations": 5,
    "loSampleSize": 14,
    "final_polisher": cv2.NONE_POLISHER,
}


@dataclass(frozen=True)
class ConfidenceHomographyResult:
    """Masks/indices always refer to the original semantic KP order.

    Scores guide sampling, not least-squares weights or calibrated probabilities.
    ``fit_inliers`` records the points used for the final fit; ``inliers`` is
    recomputed against that final matrix. Failure never returns a substitute H.
    """

    matrix: NDArray[np.float64] | None
    projected: NDArray[np.float64]
    ranked_indices: NDArray[np.int64]
    fit_inliers: NDArray[np.bool_]
    inliers: NDArray[np.bool_]
    residuals_px: NDArray[np.float64]
    status: str


def estimate_confidence_homography(
    template_xy: NDArray[np.floating],
    image_xy: NDArray[np.floating],
    scores: NDArray[np.floating],
    *,
    reprojection_threshold_px: float,
    min_score: float = 0.05,
) -> ConfidenceHomographyResult:
    """Estimate template-to-image H using ranked minimal four-point samples.

    Invalid image coordinates are omitted; template and scores must be finite.
    Stable descending score order breaks ties by input index. USAC rejects
    degenerate samples and expands the ranked sampling pool. MSAC evaluates
    geometric consensus, followed by least-squares/LM on its inliers only.
    The seed, iteration budget and single-thread setting are fixed for repeatable
    inference. Threshold units are original image pixels (caller-owned scale).
    """
    src = np.asarray(template_xy, dtype=np.float64)
    dst = np.asarray(image_xy, dtype=np.float64)
    quality = np.asarray(scores, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 2 or dst.shape != src.shape:
        raise ValueError("template_xy and image_xy must have matching (N,2) shapes")
    if not np.isfinite(src).all():
        raise ValueError("template_xy must be finite")
    if quality.shape != (len(src),) or not np.isfinite(quality).all():
        raise ValueError("scores must be finite with shape (N,)")
    if np.any((quality < 0) | (quality > 1)):
        raise ValueError("scores must lie in [0,1]")
    if not np.isfinite(min_score) or not 0 <= min_score <= 1:
        raise ValueError("min_score must lie in [0,1]")
    if not np.isfinite(reprojection_threshold_px) or reprojection_threshold_px <= 0:
        raise ValueError("reprojection_threshold_px must be finite and positive")
    eligible = np.isfinite(dst).all(axis=1) & (quality >= min_score)
    indices = np.flatnonzero(eligible).astype(np.int64)
    order = indices[np.argsort(-quality[indices], kind="stable")]
    fit_mask: NDArray[np.bool_] = np.zeros(len(src), dtype=bool)

    def failure(reason: str) -> ConfidenceHomographyResult:
        return ConfidenceHomographyResult(
            None,
            np.full_like(src, np.nan),
            order,
            fit_mask.copy(),
            np.zeros(len(src), dtype=bool),
            np.full(len(src), np.nan),
            reason,
        )

    if len(order) < 4:
        return failure("insufficient_points")
    if not _has_support(src[order], dst[order]):
        return failure("degenerate_correspondences")

    params = cv2.UsacParams()
    for name, value in PROSAC_OPTIONS.items():
        setattr(params, name, value)
    params.threshold = float(reprojection_threshold_px)
    # Final least-squares/LM is explicit below; do not polish twice.
    initial, mask = cv2.findHomography(src[order], dst[order], params)
    if initial is None or mask is None:
        return failure("prosac_failed")
    fit_mask[order] = np.asarray(mask).reshape(-1).astype(bool)
    if fit_mask.sum() < 4 or not _has_support(src[fit_mask], dst[fit_mask]):
        return failure("degenerate_consensus")
    final, _ = cv2.findHomography(src[fit_mask], dst[fit_mask], method=0)
    if final is None:
        return failure("refit_failed")
    matrix = np.asarray(final, dtype=np.float64)
    if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
        return failure("singular_refit")
    homogeneous = np.c_[src, np.ones(len(src))] @ matrix.T
    depth = homogeneous[:, 2]
    # A pole crossing the court cannot be rendered as connected finite lines.
    epsilon = 1e-10 * max(float(np.max(np.abs(depth))), 1e-300)
    if not (np.all(depth > epsilon) or np.all(depth < -epsilon)):
        return failure("court_crosses_projection_pole")
    projected = homogeneous[:, :2] / depth[:, None]
    if not np.isfinite(projected).all():
        return failure("nonfinite_projection")
    residuals = np.full(len(src), np.nan)
    residuals[eligible] = np.linalg.norm(projected[eligible] - dst[eligible], axis=1)
    inliers = eligible & (residuals <= reprojection_threshold_px)
    if inliers.sum() < 4 or not _has_support(src[inliers], dst[inliers]):
        return failure("insufficient_final_consensus")
    return ConfidenceHomographyResult(
        matrix, projected, order, fit_mask, inliers, residuals, "ok"
    )


def _has_support(src: NDArray[np.float64], dst: NDArray[np.float64]) -> bool:
    """Require eight independent homography constraints in normalized units."""
    normalized = []
    for points in (src, dst):
        centered = points - points.mean(axis=0)
        scale = float(np.sqrt(np.mean(centered**2)))
        if scale <= np.finfo(float).tiny:
            return False
        normalized.append(centered / scale)
    x, y = normalized[0].T
    u, v = normalized[1].T
    z, o = np.zeros(len(src)), np.ones(len(src))
    design = np.vstack(
        (
            np.stack((-x, -y, -o, z, z, z, u * x, u * y, u), axis=1),
            np.stack((z, z, z, -x, -y, -o, v * x, v * y, v), axis=1),
        )
    )
    return bool(np.linalg.matrix_rank(design) >= 8)
