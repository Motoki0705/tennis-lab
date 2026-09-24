"""Auditable multi-view point triangulation, without learned priors or gap filling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares


class PointRejection(IntEnum):
    VALID = 0
    INSUFFICIENT_VIEWS = 1
    DEGENERATE_RAYS = 2
    BEHIND_CAMERA = 3
    REPROJECTION = 4
    OUTSIDE_BOUNDS = 5
    EXCESSIVE_SPEED = 6
    OPTIMIZATION_FAILED = 7


@dataclass(frozen=True)
class PinholeCamera:
    camera_id: str
    intrinsic: NDArray[np.float64]
    rotation: NDArray[np.float64]  # world -> camera
    translation: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not self.camera_id or self.intrinsic.shape != (3, 3) or self.rotation.shape != (3, 3) or self.translation.shape != (3,):
            raise ValueError("Invalid pinhole camera dimensions")
        if not all(np.isfinite(x).all() for x in (self.intrinsic, self.rotation, self.translation)):
            raise ValueError("Camera parameters must be finite")
        if self.intrinsic[0, 0] <= 0 or self.intrinsic[1, 1] <= 0 or abs(np.linalg.det(self.intrinsic)) < 1e-12:
            raise ValueError("Camera intrinsic matrix must be invertible with positive focal lengths")
        if not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(self.rotation), 1, atol=1e-5):
            raise ValueError("Camera rotation must be proper")

    @property
    def matrix(self) -> NDArray[np.float64]:
        return self.intrinsic @ np.column_stack((self.rotation, self.translation))

    @property
    def center(self) -> NDArray[np.float64]:
        return -self.rotation.T @ self.translation

    def project(self, points: NDArray[np.floating]) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        camera = np.asarray(points, np.float64) @ self.rotation.T + self.translation
        projected = camera @ self.intrinsic.T
        front = np.isfinite(camera).all(-1) & (camera[..., 2] > 1e-6)
        uv = np.zeros((*points.shape[:-1], 2), np.float64)
        np.divide(projected[..., :2], projected[..., 2, None], out=uv, where=front[..., None])
        return uv, front

    def half_turned(self, turn: bool) -> PinholeCamera:
        gauge = np.diag([-1.0, -1.0, 1.0]) if turn else np.eye(3)
        return PinholeCamera(self.camera_id, self.intrinsic.copy(), self.rotation @ gauge, self.translation.copy())


@dataclass(frozen=True)
class PointTriangulationConfig:
    max_reprojection_px: float
    height_range_m: tuple[float, float]
    max_abs_xy_m: float = 40.0
    min_ray_angle_deg: float = 1.0
    max_nfev: int = 30

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_reprojection_px) or self.max_reprojection_px <= 0:
            raise ValueError("Reprojection threshold must be finite and positive")
        if not 0 < self.min_ray_angle_deg < 90 or self.max_nfev < 1:
            raise ValueError("Invalid triangulation solver configuration")
        if not all(math.isfinite(x) for x in (*self.height_range_m, self.max_abs_xy_m)) or self.height_range_m[0] >= self.height_range_m[1] or self.max_abs_xy_m <= 0:
            raise ValueError("Invalid geometric bounds")


@dataclass(frozen=True)
class TriangulatedPoints:
    positions: NDArray[np.float32]  # (...,3), zero when invalid
    valid: NDArray[np.bool_]
    reasons: NDArray[np.uint8]
    inliers: NDArray[np.bool_]  # (V,...)
    reprojection_px: NDArray[np.float32]  # (V,...), zero without a valid projection


def solve_homogeneous_dlt(design: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Batched SVD shared with the legacy scene-teacher triangulator."""
    _, singular, vectors = np.linalg.svd(design)
    homogeneous = vectors[..., -1, :]
    valid = (singular[..., -2] > 1e-7) & (np.abs(homogeneous[..., 3]) > 1e-9)
    points = np.zeros((*design.shape[:-2], 3), np.float64)
    np.divide(homogeneous[..., :3], homogeneous[..., 3, None], out=points, where=valid[..., None])
    return points, valid


def _dlt(uv: NDArray[np.float64], matrices: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64] | None:
    design = np.stack((uv[:, 0, None] * matrices[:, 2] - matrices[:, 0], uv[:, 1, None] * matrices[:, 2] - matrices[:, 1]), 1)
    point, valid = solve_homogeneous_dlt((design * np.sqrt(weights)[:, None, None]).reshape(-1, 4))
    return point if bool(valid) and np.isfinite(point).all() else None


def _reprojection_residual(
    point: NDArray[np.float64], projections: NDArray[np.float64],
    target: NDArray[np.float64], scale: NDArray[np.float64],
) -> NDArray[np.float64]:
    projected = projections[:, :, :3] @ point + projections[:, :, 3]
    denominator = np.where(np.abs(projected[:, 2]) > 1e-9, projected[:, 2], 1e-9)
    return np.asarray(((projected[:, :2] / denominator[:, None] - target) * scale).ravel(), np.float64)


def triangulate_points(
    uv_px: NDArray[np.floating],
    visibility: NDArray[np.bool_],
    cameras: tuple[PinholeCamera, ...],
    *,
    config: PointTriangulationConfig,
    confidence: NDArray[np.floating] | None = None,
) -> TriangulatedPoints:
    """Pair hypotheses, inlier consensus, weighted DLT and bounded robust refinement."""
    if uv_px.ndim < 3 or uv_px.shape[-1] != 2 or visibility.shape != uv_px.shape[:-1] or visibility.dtype != np.bool_:
        raise ValueError("Expected UV (V,...,2) and boolean visibility (V,...)")
    views = len(cameras)
    if views != len(uv_px) or views < 2 or len({c.camera_id for c in cameras}) != views:
        raise ValueError("Distinct cameras must match observations")
    if not np.isfinite(uv_px[visibility]).all():
        raise ValueError("Visible UV must be finite")
    conf = np.ones(visibility.shape, np.float64) if confidence is None else np.asarray(confidence, np.float64)
    if conf.shape != visibility.shape or not np.isfinite(conf).all() or ((conf < 0) | (conf > 1)).any():
        raise ValueError("Point confidence must be finite in [0,1]")
    shape = visibility.shape[1:]
    uv = np.asarray(uv_px, np.float64).reshape(views, -1, 2)
    observed = visibility.reshape(views, -1) & (conf.reshape(views, -1) > 0)
    weights = conf.reshape(views, -1)
    count = uv.shape[1]
    points = np.zeros((count, 3), np.float32)
    reasons = np.full(count, int(PointRejection.INSUFFICIENT_VIEWS), np.uint8)
    inliers = np.zeros((views, count), bool)
    errors = np.zeros((views, count), np.float32)
    matrices = np.stack([c.matrix for c in cameras])
    rotations = np.stack([c.rotation for c in cameras])
    translations = np.stack([c.translation for c in cameras])
    inv_k = np.stack([np.linalg.inv(c.intrinsic) for c in cameras])
    intrinsics = np.stack([c.intrinsic for c in cameras])

    def project(point: NDArray[np.float64], observation: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        xyz = np.einsum("vij,j->vi", rotations, point) + translations
        q = np.einsum("vij,vj->vi", intrinsics, xyz)
        front = xyz[:, 2] > 1e-6
        predicted = q[:, :2] / np.where(front, q[:, 2], 1)[:, None]
        return np.where(front, np.linalg.norm(predicted - observation, axis=-1), np.inf), front

    for index in np.flatnonzero(observed.sum(0) >= 2):
        active = np.flatnonzero(observed[:, index])
        rays = np.einsum("vji,vj->vi", rotations, np.einsum("vij,vj->vi", inv_k, np.c_[uv[:, index], np.ones(views)]))
        rays /= np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-12)
        best: tuple[int, float, tuple[str, str], NDArray[np.bool_]] | None = None
        reasons[index] = int(PointRejection.DEGENERATE_RAYS)
        for first, second in combinations(active, 2):
            angle = math.degrees(math.acos(float(np.clip(abs(rays[first] @ rays[second]), 0, 1))))
            if angle < config.min_ray_angle_deg:
                continue
            pair = [first, second]
            candidate = _dlt(uv[pair, index], matrices[pair], weights[pair, index])
            if candidate is None:
                continue
            residual, front = project(candidate, uv[:, index])
            if not front[pair].all():
                reasons[index] = int(PointRejection.BEHIND_CAMERA)
                continue
            selected = observed[:, index] & front & (residual <= config.max_reprojection_px)
            number = int(selected.sum())
            if number < 2:
                reasons[index] = int(PointRejection.REPROJECTION)
                continue
            key = (-number, float(residual[selected].mean()), tuple(sorted((cameras[first].camera_id, cameras[second].camera_id))))
            if best is None or key < best[:3]:
                best = (*key, selected)
        if best is None:
            continue
        selected = best[3]
        candidate = _dlt(uv[selected, index], matrices[selected], weights[selected, index])
        if candidate is None:
            continue
        projections = matrices[selected]
        target = uv[selected, index]
        scale = np.sqrt(weights[selected, index])[:, None]

        fit = least_squares(_reprojection_residual, candidate, args=(projections, target, scale), loss="huber", f_scale=config.max_reprojection_px, max_nfev=config.max_nfev)
        if not fit.success or not np.isfinite(fit.x).all():
            reasons[index] = int(PointRejection.OPTIMIZATION_FAILED)
            continue
        candidate = fit.x
        residual, front = project(candidate, uv[:, index])
        if not front[selected].all():
            reasons[index] = int(PointRejection.BEHIND_CAMERA)
            continue
        if (residual[selected] > config.max_reprojection_px).any():
            reasons[index] = int(PointRejection.REPROJECTION)
            continue
        if (np.abs(candidate[:2]) > config.max_abs_xy_m).any() or not config.height_range_m[0] <= candidate[2] <= config.height_range_m[1]:
            reasons[index] = int(PointRejection.OUTSIDE_BOUNDS)
            continue
        points[index] = candidate
        reasons[index] = 0
        inliers[:, index] = selected
        errors[:, index] = np.where(observed[:, index] & front, residual, 0)
    return TriangulatedPoints(points.reshape(*shape, 3), (reasons == 0).reshape(shape), reasons.reshape(shape), inliers.reshape(visibility.shape), errors.reshape(visibility.shape))


def reject_excessive_speed(result: TriangulatedPoints, *, fps: float, max_speed_mps: float) -> TriangulatedPoints:
    """One-pass rejection of both endpoints; never bridge a missing frame."""
    if result.positions.ndim != 2 or not math.isfinite(fps) or fps <= 0 or max_speed_mps <= 0:
        raise ValueError("Speed gating expects one trajectory and positive fps/speed")
    bad_edges = result.valid[:-1] & result.valid[1:] & (np.linalg.norm(np.diff(result.positions, axis=0), axis=-1) * fps > max_speed_mps)
    bad = np.r_[bad_edges, False] | np.r_[False, bad_edges]
    points, valid, reasons = result.positions.copy(), result.valid.copy(), result.reasons.copy()
    points[bad] = 0
    valid[bad] = False
    reasons[bad] = int(PointRejection.EXCESSIVE_SPEED)
    inliers = result.inliers & valid
    return TriangulatedPoints(points, valid, reasons, inliers, result.reprojection_px.copy())


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
