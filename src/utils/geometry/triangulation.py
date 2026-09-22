"""CPU multi-view triangulation in the coordinate system of supplied cameras.

Cameras are fixed OpenCV pinhole matrices ``P = K [R | t]`` with positive
camera-Z in front. Observations must be undistorted pixel coordinates. This
module does not estimate calibration, associate people, or smooth time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class TriangulationResult:
    """Invalid 3D points/reprojections are NaN; ``used_views`` records input use."""

    points: FloatArray
    valid: NDArray[np.bool_]
    used_views: NDArray[np.bool_]
    reprojected: FloatArray
    depths: FloatArray


def project_multiview(
    points: FloatArray, cameras: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Project ``(..., 3)`` points through ``(V, 3, 4)`` matrices, in pixels."""
    homogeneous = np.concatenate((points, np.ones((*points.shape[:-1], 1))), axis=-1)
    image = np.einsum("vij,...j->...vi", cameras, homogeneous)
    depth = image[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = image[..., :2] / depth[..., None]
    uv = np.where((np.abs(depth) > 1e-12)[..., None], uv, np.nan)
    return uv, depth


def triangulate_multiview(
    observations: FloatArray,
    scores: FloatArray,
    cameras: FloatArray,
    *,
    min_score: float = 0.3,
    refinement_steps: int = 12,
) -> TriangulationResult:
    """Weighted DLT followed by weighted pixel reprojection minimization.

    ``observations``: (..., V, 2); ``scores``: (..., V). At least two finite
    observations with score >= min_score and score > 0 are required. Weights
    are clipped detector scores, not calibrated uncertainty probabilities.
    Missing views, degenerate DLT and negative depth in any used view produce
    explicit invalid points. No outlier rejection or anatomical prior is used.
    """
    uv = np.asarray(observations, dtype=np.float64)
    score = np.asarray(scores, dtype=np.float64)
    p = np.asarray(cameras, dtype=np.float64)
    if p.ndim != 3 or p.shape[1:] != (3, 4) or len(p) < 2:
        raise ValueError("cameras must have shape (V>=2, 3, 4)")
    if uv.shape[-2:] != (len(p), 2) or score.shape != uv.shape[:-1]:
        raise ValueError(
            "observations/scores must have matching (..., V, 2)/(..., V) shapes"
        )
    if (
        not np.isfinite(p).all()
        or not np.isfinite(min_score)
        or min_score < 0
        or refinement_steps < 0
    ):
        raise ValueError("invalid camera, threshold, or refinement_steps")
    leading = uv.shape[:-2]
    uv = uv.reshape(-1, len(p), 2)
    score = score.reshape(-1, len(p))
    used = (
        np.isfinite(uv).all(-1)
        & np.isfinite(score)
        & (score >= min_score)
        & (score > 0)
    )
    weight = np.where(used, np.clip(score, 0, 1), 0)
    uv = np.where(used[..., None], uv, 0)
    a = uv[..., None] * p[None, :, 2:3, :] - p[None, :, :2, :]
    a = (a * np.sqrt(weight)[..., None, None]).reshape(-1, 2 * len(p), 4)
    _, singular, vh = np.linalg.svd(a, full_matrices=False)
    h = vh[:, -1]
    valid = (used.sum(-1) >= 2) & (np.abs(h[:, 3]) > 1e-10)
    valid &= singular[:, -2] > np.maximum(singular[:, 0] * 1e-10, 1e-12)
    x = np.full((len(uv), 3), np.nan)
    x[valid] = h[valid, :3] / h[valid, 3:4]

    indices = np.flatnonzero(valid)
    if len(indices) and refinement_steps:
        y = x[indices].copy()
        target = uv[indices]
        w = weight[indices]
        for _ in range(refinement_steps):
            pred, depth = project_multiview(y, p)
            # A missing camera may have zero depth: 0 * NaN is still NaN.
            pred = np.where(w[..., None] > 0, pred, 0)
            safe_depth = np.where(np.abs(depth) > 1e-8, depth, 1e-8)
            jac = p[None, :, :2, :3] - pred[..., None] * p[None, :, 2:3, :3]
            jac /= safe_depth[..., None, None]
            residual = pred - target
            normal = np.einsum("nvai,nvaj,nv->nij", jac, jac, w)
            gradient = np.einsum("nvai,nva,nv->ni", jac, residual, w)
            damping = np.maximum(np.trace(normal, axis1=-2, axis2=-1), 1) * 1e-8
            normal += damping[:, None, None] * np.eye(3)
            step = np.linalg.solve(normal, gradient[..., None])[..., 0]
            cost = np.sum(residual**2 * w[..., None], axis=(1, 2))
            accepted: NDArray[np.bool_] = np.zeros(len(y), dtype=bool)
            for scale in (1.0, 0.5, 0.25, 0.125):
                candidate = y - scale * step
                trial, _ = project_multiview(candidate, p)
                trial = np.where(w[..., None] > 0, trial, 0)
                trial_cost = np.sum((trial - target) ** 2 * w[..., None], axis=(1, 2))
                take = (~accepted) & np.isfinite(trial_cost) & (trial_cost <= cost)
                y[take] = candidate[take]
                accepted |= take
        x[indices] = y
    projected, depths = project_multiview(x, p)
    valid &= np.isfinite(x).all(-1) & ((depths > 1e-6) | ~used).all(-1)
    x[~valid] = np.nan
    projected[~valid] = np.nan
    depths[~valid] = np.nan
    return TriangulationResult(
        points=x.reshape(*leading, 3),
        valid=valid.reshape(leading),
        used_views=used.reshape(*leading, len(p)),
        reprojected=projected.reshape(*leading, len(p), 2),
        depths=depths.reshape(*leading, len(p)),
    )
