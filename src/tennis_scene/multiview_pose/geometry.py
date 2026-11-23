"""Geometric screening and robust triangulation of frame-local COCO17 groups."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.multiview_pose.schema import PoseConfig, RowRejection
from src.utils.geometry.triangulation import PinholeCamera, PointRejection
from src.utils.schema.player import COCO17_BONE_LENGTH_EDGES, COCO17_TORSO_TWIST_JOINTS

_TORSO = list(COCO17_TORSO_TWIST_JOINTS)
_BONE_START, _BONE_END = np.asarray(COCO17_BONE_LENGTH_EDGES).T
# Broad upper bounds (metres), in the order of COCO17_BONE_LENGTH_EDGES.
_BONE_MAX_M = np.array(
    [0.85, 0.75, 1.15, 1.15, 0.80, 0.75, 0.80, 0.75, 1.0, 1.0, 1.0, 1.0]
)
_XYZ_IDENTITY = np.eye(3)


@dataclass(frozen=True)
class ViewRays:
    uv: NDArray[np.float64]
    directions: NDArray[np.float64]
    observed: NDArray[np.bool_]
    strong: NDArray[np.bool_]

    def slice(self, start: int, end: int) -> ViewRays:
        return ViewRays(
            self.uv[start:end],
            self.directions[start:end],
            self.observed[start:end],
            self.strong[start:end],
        )


def view_rays(
    pose: NDArray[np.float32],
    camera: PinholeCamera,
    size: tuple[int, int],
    config: PoseConfig,
) -> ViewRays:
    uv = np.asarray(pose[..., :2], np.float64)
    observed = (pose[..., 2] >= config.min_joint_score) & ((uv >= 0) & (uv < size)).all(
        axis=-1
    )
    homogeneous = np.concatenate((uv, np.ones_like(uv[..., :1])), axis=-1)
    directions = homogeneous @ (np.linalg.inv(camera.intrinsic).T @ camera.rotation)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    return ViewRays(
        uv, directions, observed, observed & (pose[..., 2] >= config.strong_joint_score)
    )


@dataclass(frozen=True)
class PairFits:
    points: NDArray[np.float64]  # (rows_a,rows_b,17,3)
    observed: NDArray[np.bool_]
    angular: NDArray[np.bool_]
    forward: NDArray[np.bool_]
    physical: NDArray[np.bool_]
    errors_px: NDArray[np.float64]
    compatible_rows: NDArray[np.bool_]


def _within_world(points: NDArray[np.float64], config: PoseConfig) -> NDArray[np.bool_]:
    low, high = config.height_range_m
    return (
        (np.abs(points[..., :2]) <= config.max_abs_xy_m).all(axis=-1)
        & (points[..., 2] >= low)
        & (points[..., 2] <= high)
    )


def compare_rays(
    a: ViewRays,
    b: ViewRays,
    camera_a: PinholeCamera,
    camera_b: PinholeCamera,
    config: PoseConfig,
    threshold_px: float,
) -> PairFits:
    left, right = a.directions[:, None], b.directions[None, :]
    center_a, center_b = camera_a.center, camera_b.center
    separation = center_a - center_b
    cosine = (left * right).sum(axis=-1)
    sine_squared = 1 - cosine**2
    observed = a.observed[:, None] & b.observed[None, :]
    angular = observed & (
        sine_squared >= np.sin(np.deg2rad(config.min_ray_angle_deg)) ** 2
    )
    denominator = np.where(angular, sine_squared, 1.0)
    offset_a, offset_b = left @ separation, right @ separation
    depth_a = (cosine * offset_b - offset_a) / denominator
    depth_b = (offset_b - cosine * offset_a) / denominator
    # Midpoint of the closest points on the two viewing rays.
    midpoint = (
        center_a + depth_a[..., None] * left + center_b + depth_b[..., None] * right
    ) * 0.5
    projected_a, front_a = camera_a.project(midpoint)
    projected_b, front_b = camera_b.project(midpoint)
    errors = np.maximum(
        np.linalg.norm(projected_a - a.uv[:, None], axis=-1),
        np.linalg.norm(projected_b - b.uv[None, :], axis=-1),
    )
    forward = angular & (depth_a > 0) & (depth_b > 0) & front_a & front_b
    physical = forward & _within_world(midpoint, config)
    compatible = (physical[..., _TORSO] & (errors[..., _TORSO] <= threshold_px)).sum(
        axis=-1
    ) >= 3
    return PairFits(midpoint, observed, angular, forward, physical, errors, compatible)


@dataclass(frozen=True)
class GroupFits:
    reasons: NDArray[np.uint8]
    positions_m: NDArray[np.float64]
    joint_valid: NDArray[np.bool_]
    joint_reasons: NDArray[np.uint8]
    inliers: NDArray[np.bool_]
    errors: NDArray[np.float64]
    error_valid: NDArray[np.bool_]
    scores: NDArray[np.float64]


def _project(
    points: NDArray[np.float64],
    cameras: tuple[PinholeCamera, ...],
    uv: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    projections = [camera.project(points) for camera in cameras]
    image = np.stack([px for px, _ in projections], axis=1)
    front = np.stack([valid for _, valid in projections], axis=1)
    return np.linalg.norm(image - uv, axis=-1), front


def fit_groups(
    groups: NDArray[np.int64],
    views: tuple[ViewRays, ...],
    cameras: tuple[PinholeCamera, ...],
    pairs: dict[tuple[int, int], PairFits],
    config: PoseConfig,
    pixel_scale: float,
) -> GroupFits:
    count, n_views = groups.shape
    present = groups >= 0
    uv: NDArray[np.float64] = np.zeros((count, n_views, 17, 2))
    rays: NDArray[np.float64] = np.zeros((count, n_views, 17, 3))
    observed: NDArray[np.bool_] = np.zeros((count, n_views, 17), bool)
    strong = np.zeros_like(observed)
    for camera, view in enumerate(views):
        active = present[:, camera]
        rows = groups[active, camera]
        uv[active, camera] = view.uv[rows]
        rays[active, camera] = view.directions[rows]
        observed[active, camera] = view.observed[rows]
        strong[active, camera] = view.strong[rows]

    limit = config.reprojection_px * pixel_scale
    mixed_limit = config.mixed_reprojection_px * pixel_scale
    best_number: NDArray[np.int64] = np.zeros((count, 17), np.int64)
    best_error: NDArray[np.float64] = np.full((count, 17), np.inf)
    best_views = np.zeros_like(observed)
    has_angle = np.zeros_like(best_number, bool)
    has_forward = np.zeros_like(has_angle)
    has_physical = np.zeros_like(has_angle)
    has_strong_pair = np.zeros_like(has_angle)
    has_consistent_strong_pair = np.zeros_like(has_angle)
    for (left, right), pair in pairs.items():
        active = present[:, left] & present[:, right]
        if not active.any():
            continue
        rows = groups[active, left], groups[active, right]
        physical = pair.physical[rows]
        has_angle[active] |= pair.angular[rows]
        has_forward[active] |= pair.forward[rows]
        has_physical[active] |= physical
        strong_pair = physical & strong[active, left] & strong[active, right]
        has_strong_pair[active] |= strong_pair
        has_consistent_strong_pair[active] |= strong_pair & (
            pair.errors_px[rows] <= mixed_limit
        )
        residual, front = _project(pair.points[rows], cameras, uv[active])
        supported = observed[active] & front & (residual <= limit)
        # Seed support requires the original two rays to agree with the proposal.
        seed_valid = physical & supported[:, left] & supported[:, right]
        number = np.where(seed_valid, supported.sum(axis=1), 0)
        cost = np.where(supported, residual**2, 0).sum(axis=1)
        replace = seed_valid & (
            (number > best_number[active])
            | ((number == best_number[active]) & (cost < best_error[active]))
        )
        best_views[active] = np.where(replace[:, None], supported, best_views[active])
        best_number[active] = np.where(replace, number, best_number[active])
        best_error[active] = np.where(replace, cost, best_error[active])

    # Refit on the winning support via the linear least-squares ray distance.
    # Each winning set includes a checked non-parallel seed pair.
    normals = _XYZ_IDENTITY - rays[..., :, None] * rays[..., None, :]
    normals *= best_views[..., None, None]
    centres = np.stack([camera.center for camera in cameras])
    matrices = np.where(
        (best_number >= 2)[..., None, None], normals.sum(axis=1), _XYZ_IDENTITY
    )
    target = np.einsum("gvjkl,vl->gjk", normals, centres)
    positions = np.linalg.solve(matrices, target[..., None])[..., 0]
    errors, front = _project(positions, cameras, uv)
    inliers = best_views & front & (errors <= limit)
    separated = np.zeros_like(has_angle)
    for left, right in pairs:
        cosine = (rays[:, left] * rays[:, right]).sum(axis=-1)
        separated |= (
            inliers[:, left]
            & inliers[:, right]
            & (1 - cosine**2 >= np.sin(np.deg2rad(config.min_ray_angle_deg)) ** 2)
        )
    within = _within_world(positions, config)
    valid = separated & within
    inliers &= valid[:, None]
    error_valid = valid[:, None] & observed & front

    mixed_reprojection = (
        strong & valid[:, None] & (~front | (errors > mixed_limit))
    ).any(axis=(1, 2))
    mixed_reprojection |= (~valid & has_strong_pair & ~has_consistent_strong_pair).any(
        axis=1
    )
    bone_visible = valid[:, _BONE_START] & valid[:, _BONE_END]
    bone_lengths = np.linalg.norm(
        positions[:, _BONE_END] - positions[:, _BONE_START], axis=-1
    )
    mixed_bones = (bone_visible & (bone_lengths > _BONE_MAX_M)).any(axis=1)
    supported_views = inliers[..., _TORSO].sum(axis=-1) >= 3
    supported_group = (supported_views | ~present).all(axis=1)

    reasons = np.full(count, RowRejection.INCONSISTENT_GEOMETRY, np.uint8)
    reasons[~(has_angle[:, _TORSO].sum(axis=-1) >= 3)] = RowRejection.DEGENERATE_RAYS
    reasons[supported_group] = RowRejection.ASSIGNED
    reasons[mixed_reprojection | mixed_bones] = RowRejection.MIXED_POSE
    joint_reasons = np.full((count, 17), PointRejection.INSUFFICIENT_VIEWS, np.uint8)
    joint_reasons[observed.sum(axis=1) >= 2] = PointRejection.DEGENERATE_RAYS
    joint_reasons[has_angle] = PointRejection.BEHIND_CAMERA
    joint_reasons[has_forward] = PointRejection.OUTSIDE_BOUNDS
    joint_reasons[has_physical] = PointRejection.REPROJECTION
    joint_reasons[(best_number >= 2) & ~within] = PointRejection.OUTSIDE_BOUNDS
    joint_reasons[valid] = PointRejection.VALID

    scores: NDArray[np.float64] = np.zeros(count)
    for left, right in pairs:
        residual = np.maximum(errors[:, left], errors[:, right]) / limit
        scores += np.where(
            inliers[:, left] & inliers[:, right], 1 - 0.5 * residual**2, 0
        ).mean(axis=-1)
    return GroupFits(
        reasons,
        np.where(valid[..., None], positions, 0),
        valid,
        joint_reasons,
        inliers,
        np.where(error_valid, errors, 0),
        error_valid,
        scores,
    )
