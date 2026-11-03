"""Sparse, court-anchored camera/clock/ball bundle adjustment on CPU.

Principal point, aspect ratio and zero distortion stay fixed. Metric KP14
fixes the spatial gauge; the lexically first selected camera fixes the clock.
Camera frame t observes world time t/fps + delta_v.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from src.tennis_scene.ball_calibration.court import CourtViewObservations
from src.tennis_scene.pipeline.components.ball_reconstruction import (
    BallReconstructionResult,
    reconstruct_ball,
)
from src.tennis_scene.pipeline.observation_types import GroupedObservations
from src.utils.geometry.triangulation import PinholeCamera
from src.utils.schema.court import STANDARD_COURT_CONFIG, court_keypoints_3d

# Parameter units, prior standard deviations, and one quarter of camera
# hard bounds: log focal, rotation radians, camera centre metres.
CAMERA_SCALES = np.array([0.05, 0.02, 0.02, 0.02, 0.5, 0.5, 0.5])


@dataclass(frozen=True)
class BundleAdjustmentConfig:
    court_sigma_px: float = 3.0
    ball_sigma_px: float = 6.0
    prior_weight: float = 1.0
    max_offset_frames: float = 0.75
    max_ball_frames: int = 240
    max_nfev: int = 200

    def __post_init__(self) -> None:
        for name in ("court_sigma_px", "ball_sigma_px", "max_offset_frames"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_offset_frames >= 1:
            raise ValueError("max_offset_frames must be less than one source frame")
        if (
            isinstance(self.prior_weight, bool)
            or not math.isfinite(self.prior_weight)
            or self.prior_weight < 0
        ):
            raise ValueError("prior_weight must be finite and nonnegative")
        if (
            type(self.max_ball_frames) is not int
            or self.max_ball_frames < 8
            or type(self.max_nfev) is not int
            or self.max_nfev < 1
        ):
            raise ValueError("Invalid bundle adjustment solver budget")


@dataclass(frozen=True)
class CalibrationRefinement:
    cameras: tuple[PinholeCamera, ...]
    time_offsets_seconds: NDArray[np.float64]
    fit_frame_indices: NDArray[np.int64]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class CalibratedBall:
    ball: BallReconstructionResult
    calibration: CalibrationRefinement


def court_world_points(half_turn: bool) -> NDArray[np.float64]:
    """Apply the decided half-turn to the original source homography template.

    PROSAC x_reflected is search provenance: the final H maps the original
    template. Applying that reflection again would reverse semantic labels.
    """
    xyz = court_keypoints_3d(STANDARD_COURT_CONFIG).numpy()[:14].astype(np.float64)
    if half_turn:
        xyz[:, :2] *= -1
    return xyz


def resample_observations(
    observations: GroupedObservations, offsets_frames: NDArray[np.float64]
) -> GroupedObservations:
    """Sample camera source time t-delta without extrapolation or gap bridging."""
    count, views, frames, joints, _ = observations.uv_px.shape
    if count != 1 or joints != 1 or offsets_frames.shape != (views,):
        raise ValueError("Clock correction requires one ball and one offset per view")
    if not np.isfinite(offsets_frames).all() or np.any(np.abs(offsets_frames) >= 1):
        raise ValueError("Clock correction requires finite subframe offsets")
    uv = np.zeros_like(observations.uv_px)
    confidence = np.zeros_like(observations.confidence)
    visible = np.zeros_like(observations.visibility)
    raw = np.full_like(observations.raw_indices, -1)
    for v, delta in enumerate(offsets_frames):
        query = np.arange(frames, dtype=np.float64) - delta
        lower = np.floor(query).astype(np.int64)
        fraction = query - lower
        upper = lower + (fraction != 0)
        inside = (lower >= 0) & (upper < frames)
        lo, hi = np.clip(lower, 0, frames - 1), np.clip(upper, 0, frames - 1)
        valid = inside & observations.visibility[0, v, lo, 0] & observations.visibility[0, v, hi, 0]
        values = (
            observations.uv_px[0, v, lo, 0].astype(np.float64) * (1 - fraction[:, None])
            + observations.uv_px[0, v, hi, 0] * fraction[:, None]
        )
        uv[0, v, valid, 0] = values[valid]
        confidence[0, v, valid, 0] = np.minimum(
            observations.confidence[0, v, lo[valid], 0],
            observations.confidence[0, v, hi[valid], 0],
        )
        visible[0, v, :, 0] = valid
        # Resampled correspondences have no single raw row.
        if delta == 0:
            raw[0, v] = observations.raw_indices[0, v]
    return GroupedObservations(observations.identities, uv, confidence, visible, raw)


def _cameras(
    original: tuple[PinholeCamera, ...], params: NDArray[np.float64]
) -> tuple[PinholeCamera, ...]:
    result = []
    for camera, delta in zip(original, params * CAMERA_SCALES, strict=True):
        intrinsic = camera.intrinsic.copy()
        intrinsic[0, 0] *= np.exp(delta[0])
        intrinsic[1, 1] *= np.exp(delta[0])
        rotation = Rotation.from_rotvec(delta[1:4]).as_matrix() @ camera.rotation
        centre = camera.center + delta[4:7]
        result.append(PinholeCamera(camera.camera_id, intrinsic, rotation, -rotation @ centre))
    return tuple(result)


def refine_court_ball(
    observations: GroupedObservations,
    cameras: tuple[PinholeCamera, ...],
    courts: tuple[CourtViewObservations, ...],
    half_turns: tuple[bool, ...],
    *,
    fps: float,
    reprojection_px: float,
    min_frames: int,
    config: BundleAdjustmentConfig,
) -> CalibratedBall:
    """Jointly fit cameras, clocks and 3D points, then replay triangulation.

    Only the source homography's selected KP enter fitting; other valid raw KP
    are reserved for reporting. The interface has no labels or held-out views.
    Failure raises, with no substitute calibration.
    """
    ids = tuple(c.camera_id for c in cameras)
    views = len(cameras)
    if (
        views < 2 or len(set(ids)) != views
        or tuple(c.camera_id for c in courts) != ids
        or len(half_turns) != views or any(type(x) is not bool for x in half_turns)
        or not math.isfinite(fps) or fps <= 0
    ):
        raise ValueError("Bundle adjustment camera/court/clock axes disagree")
    if observations.uv_px.shape[1] != views or observations.uv_px.shape[0] != 1:
        raise ValueError("Bundle adjustment requires a single ball on the camera axis")
    world = tuple(court_world_points(turn) for turn in half_turns)
    fit_masks = tuple(c.valid & c.selected_for_homography for c in courts)
    court_ranks = []
    for camera, xyz, mask in zip(cameras, world, fit_masks, strict=True):
        chosen = np.flatnonzero(mask.any(0))
        if len(chosen) < 4 or np.linalg.matrix_rank(xyz[chosen, :2] - xyz[chosen[0], :2]) < 2:
            raise ValueError("At least four noncollinear selected court keypoints are required")
        origin, front = camera.project(xyz[chosen])
        jacobian = []
        for axis in range(7):
            step: NDArray[np.float64] = np.zeros((1, 7))
            step[0, axis] = 1e-4
            moved = _cameras((camera,), step)[0].project(xyz[chosen])[0]
            jacobian.append((moved - origin).ravel() / 1e-4)
        rank = int(np.linalg.matrix_rank(np.stack(jacobian, 1), tol=1e-5))
        if not front.all() or rank != 7:
            raise ValueError("Selected court observations do not identify seven camera parameters")
        court_ranks.append(rank)
    baseline = reconstruct_ball(
        observations, cameras, fps=fps, reprojection_px=reprojection_px, min_frames=min_frames
    )
    uv = baseline.uv_px.astype(np.float64)
    inliers = baseline.trajectory.inliers
    stable = inliers[:, :-2] & inliers[:, 1:-1] & inliers[:, 2:]
    frames: NDArray[np.int64] = np.flatnonzero((stable.sum(0) >= 2) & baseline.trajectory.valid[1:-1]) + 1
    if len(frames) < 8:
        raise ValueError("Insufficient continuous baseline ball correspondences for calibration")
    if len(frames) > config.max_ball_frames:
        frames = frames[np.linspace(0, len(frames) - 1, config.max_ball_frames).astype(int)]
    frames = frames.astype(np.int64)
    supports = stable[:, frames - 1]
    # An unconnected or static stream cannot identify a relative clock. Priors
    # are regularization, not a reason to claim a measured zero offset.
    connected = {ids.index(min(ids))}
    while True:
        expanded = connected | {v for v in range(views) if any(
            np.count_nonzero(supports[v] & supports[u]) >= 8 for u in connected)}
        if expanded == connected:
            break
        connected = expanded
    if len(connected) != views or any(
        np.max(np.linalg.norm(uv[v, frames[supports[v]] + 1] - uv[v, frames[supports[v]] - 1], axis=-1), initial=0) < 1e-5
        for v in range(views)
    ):
        raise ValueError("Ball correspondences do not identify connected moving camera clocks")
    npoints = len(frames)
    reference = ids.index(min(ids))
    clock_views = tuple(v for v in range(views) if v != reference)
    clock_columns = {v: 7 * views + i for i, v in enumerate(clock_views)}
    global_count = 8 * views - 1
    initial = np.concatenate((np.zeros(global_count), baseline.trajectory.positions[frames].ravel())).astype(np.float64)

    def decode(p: NDArray[np.float64]) -> tuple[tuple[PinholeCamera, ...], NDArray[np.float64], NDArray[np.float64]]:
        fitted = _cameras(cameras, p[:7 * views].reshape(views, 7))
        delta: NDArray[np.float64] = np.zeros(views, np.float64)
        delta[list(clock_views)] = p[7 * views:global_count]
        return fitted, delta, p[global_count:].reshape(npoints, 3)

    court_indices = tuple(np.nonzero(mask) for mask in fit_masks)
    ball_indices = tuple(np.flatnonzero(mask) for mask in supports)

    def residual(p: NDArray[np.float64]) -> NDArray[np.float64]:
        fitted, delta, points = decode(p)
        rows = []
        for v, camera in enumerate(fitted):
            f, kp = court_indices[v]
            projected, front = camera.project(world[v][kp])
            court_residual = ((projected - courts[v].keypoints_px[f, kp]) / config.court_sigma_px).ravel()
            court_residual += np.repeat(~front, 2) * 1e5
            rows.append(court_residual)
            index = ball_indices[v]
            source_frames = frames[index]
            d = delta[v]
            neighbour = source_frames - 1 if d >= 0 else source_frames + 1
            target = uv[v, source_frames] + abs(d) * (uv[v, neighbour] - uv[v, source_frames])
            projected, _ = camera.project(points[index])
            rows.append(((projected - target) / config.ball_sigma_px).ravel())
            depth = points[index] @ camera.rotation[2] + camera.translation[2]
            rows.append(np.minimum(depth - 0.01, 0) * 100)
        rows.append(config.prior_weight * p[:7 * views])
        rows.append(config.prior_weight * p[7 * views:global_count] / 0.5)
        return np.concatenate(rows)

    # A ball residual depends on one camera, its clock and one latent point.
    sparsity = lil_matrix((len(residual(initial)), len(initial)), dtype=np.int8)
    cursor = 0
    for v in range(views):
        count = 2 * len(court_indices[v][0])
        sparsity[cursor:cursor + count, 7 * v:7 * (v + 1)] = 1
        cursor += count
        indices = ball_indices[v]
        for row, point in enumerate(indices):
            for rr in (cursor + 2 * row, cursor + 2 * row + 1, cursor + 2 * len(indices) + row):
                sparsity[rr, 7 * v:7 * (v + 1)] = 1
                sparsity[rr, global_count + 3 * point:global_count + 3 * (point + 1)] = 1
                if v in clock_columns:
                    sparsity[rr, clock_columns[v]] = 1
        cursor += 3 * len(indices)
    for col in range(global_count):
        sparsity[cursor + col, col] = 1
    lower = np.full(len(initial), -np.inf)
    upper = -lower
    lower[:7 * views], upper[:7 * views] = -4, 4
    lower[7 * views:global_count], upper[7 * views:global_count] = -config.max_offset_frames, config.max_offset_frames
    result = least_squares(
        residual, initial, jac_sparsity=sparsity.tocsr(), bounds=(lower, upper),
        loss="soft_l1", f_scale=1, x_scale="jac", max_nfev=config.max_nfev,
        ftol=1e-7, xtol=1e-7, gtol=1e-7,
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Court/ball bundle adjustment failed: {result.message}")
    refined, offsets, points = decode(result.x)
    if any(not camera.project(points)[1].all() for camera in refined):
        raise RuntimeError("Bundle adjustment returned points behind a camera")
    corrected = resample_observations(observations, offsets)
    ball = reconstruct_ball(
        corrected, refined, fps=fps, reprojection_px=reprojection_px, min_frames=min_frames
    )
    calibration = CalibrationRefinement(
        refined, offsets / fps, frames,
        {
            "status": "converged", "solver_message": result.message,
            "nfev": result.nfev, "cost": float(result.cost),
            "optimality": float(result.optimality),
            "reference_camera": ids[reference], "offset_frames": offsets,
            "camera_parameter_units": CAMERA_SCALES,
            "camera_parameter_deltas": result.x[:7 * views].reshape(views, 7) * CAMERA_SCALES,
            "court_fit_count": [int(m.sum()) for m in fit_masks],
            "court_camera_jacobian_rank": court_ranks,
            "court_holdout_count": [int((c.valid & ~m).sum()) for c, m in zip(courts, fit_masks, strict=True)],
            "ball_fit_count": supports.sum(1).tolist(),
            "fit_3d_points": points, "fit_support": supports,
            "resampled_uv_px": ball.uv_px,
            "resampled_visibility": ball.visibility,
            "parameter_count": len(initial), "residual_count": len(result.fun),
            "camera_bounds_hit": bool(np.any(np.abs(result.x[:7 * views]) > 3.99)),
            "clock_bounds_hit": bool(np.any(np.abs(offsets) > config.max_offset_frames * .999)),
        },
    )
    # SceneResult's 2D fields remain measurements on each original source
    # frame. The resampled coordinates are used only by the 3D solve and
    # retained explicitly in diagnostics, never relabelled as source pixels.
    source_clock_ball = BallReconstructionResult(
        baseline.uv_px, baseline.visibility, ball.trajectory, ball.status
    )
    return CalibratedBall(source_clock_ball, calibration)
