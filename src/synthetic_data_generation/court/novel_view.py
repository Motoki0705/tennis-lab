"""Deterministic, support-bounded novel-view sampling for court detection.

The sampler operates in accepted metric court coordinates.  It perturbs only
captured poses whose fourteen line keypoints are already fully framed, rejects
poses outside a coupled SE(3) support ball, and applies explicit near-plane,
camera-height, sparse-scene collision, and court-framing gates.  A farthest-view
selection pass then chooses a diverse subset without relaxing any gate.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from src.synthetic_data_generation.scene_contract import (
    SceneCamera,
    SimilarityTransform,
)

_LINE_KEYPOINT_COUNT = 14
_EPSILON = 1.0e-12


@dataclass(frozen=True)
class NovelViewThresholds:
    """Pre-registered safety limits for one accepted scene."""

    translation_limit_m: float = 0.25
    rotation_limit_deg: float = 1.5
    support_score_limit: float = 1.0
    near_plane_m: float = 0.10
    min_camera_height_m: float = 1.20
    min_image_margin_px: float = 0.0
    min_line_keypoints_visible: int = _LINE_KEYPOINT_COUNT
    collision_neighbor_rank: int = 8
    min_collision_clearance_m: float = 0.25

    def __post_init__(self) -> None:
        positive = {
            "translation_limit_m": self.translation_limit_m,
            "rotation_limit_deg": self.rotation_limit_deg,
            "support_score_limit": self.support_score_limit,
            "near_plane_m": self.near_plane_m,
            "min_camera_height_m": self.min_camera_height_m,
            "min_collision_clearance_m": self.min_collision_clearance_m,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not np.isfinite(self.min_image_margin_px):
            raise ValueError("min_image_margin_px must be finite.")
        if not 1 <= self.min_line_keypoints_visible <= _LINE_KEYPOINT_COUNT:
            raise ValueError(
                "min_line_keypoints_visible must be between 1 and 14."
            )
        if (
            isinstance(self.collision_neighbor_rank, bool)
            or self.collision_neighbor_rank < 1
        ):
            raise ValueError("collision_neighbor_rank must be a positive integer.")


@dataclass(frozen=True)
class NovelViewCamera:
    """One accepted novel camera with complete geometry evidence."""

    camera_id: str
    anchor_camera_id: str
    width: int
    height: int
    intrinsics: tuple[float, ...]
    camera_to_court: tuple[float, ...]
    camera_to_scene: tuple[float, ...]
    court_keypoints_uv: tuple[float, ...]
    court_keypoints_depth_m: tuple[float, ...]
    court_keypoints_visible: tuple[bool, ...]
    translation_from_anchor_m: float
    rotation_from_anchor_deg: float
    nearest_captured_translation_m: float
    nearest_captured_rotation_deg: float
    extrapolation_score: float
    collision_clearance_m: float
    min_court_depth_m: float
    min_line_margin_px: float


@dataclass(frozen=True)
class NovelViewSamplingResult:
    """Selected cameras plus auditable proposal and rejection counts."""

    seed: int
    safe_anchor_count: int
    proposal_count: int
    accepted_candidate_count: int
    rejection_counts: tuple[tuple[str, int], ...]
    selected: tuple[NovelViewCamera, ...]


@dataclass(frozen=True)
class _Pose:
    center: NDArray[np.float64]
    rotation: NDArray[np.float64]


@dataclass(frozen=True)
class _PoseCloud:
    centers: NDArray[np.float64]
    rotations: NDArray[np.float64]


@dataclass(frozen=True)
class _ProjectionEvidence:
    uv: NDArray[np.float64]
    depth: NDArray[np.float64]
    visible: NDArray[np.bool_]
    collision_clearance_m: float
    min_line_margin_px: float


@dataclass(frozen=True)
class _Candidate:
    proposal_id: str
    anchor: SceneCamera
    pose: _Pose
    projection: _ProjectionEvidence
    translation_from_anchor_m: float
    rotation_from_anchor_deg: float
    nearest_captured_translation_m: float
    nearest_captured_rotation_deg: float
    extrapolation_score: float


def sample_safe_novel_views(
    cameras: Sequence[SceneCamera],
    court_from_scene: SimilarityTransform,
    court_keypoints_court: NDArray[np.floating],
    support_points_scene: NDArray[np.floating],
    *,
    seed: int,
    proposals_per_anchor: int,
    max_views: int,
    thresholds: NovelViewThresholds | None = None,
) -> NovelViewSamplingResult:
    """Sample and select deterministic novel cameras without a fallback.

    Args:
        cameras: Accepted captured OpenCV cameras.
        court_from_scene: Accepted metric transform.
        court_keypoints_court: CourtKP20 coordinates with shape ``[20, 3]``.
        support_points_scene: Finite sparse scene support with shape ``[N, 3]``.
        seed: Non-negative deterministic seed.
        proposals_per_anchor: Number of six-dimensional ball samples per safe
            captured anchor.
        max_views: Required number of accepted cameras after farthest selection.
        thresholds: Explicit safety thresholds.

    Raises:
        ValueError: If inputs are invalid, no safe anchor exists, or fewer than
            ``max_views`` candidates pass every gate.
    """
    limits = thresholds or NovelViewThresholds()
    camera_tuple = tuple(cameras)
    if not camera_tuple:
        raise ValueError("cameras must not be empty.")
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if isinstance(proposals_per_anchor, bool) or proposals_per_anchor < 1:
        raise ValueError("proposals_per_anchor must be a positive integer.")
    if isinstance(max_views, bool) or max_views < 1:
        raise ValueError("max_views must be a positive integer.")

    court_keypoints = _finite_points(
        court_keypoints_court,
        name="court_keypoints_court",
        minimum_count=20,
    )
    if court_keypoints.shape != (20, 3):
        raise ValueError(
            f"court_keypoints_court must have shape (20, 3), "
            f"got {court_keypoints.shape}."
        )
    support_scene = _finite_points(
        support_points_scene,
        name="support_points_scene",
        minimum_count=limits.collision_neighbor_rank,
    )
    support_court = court_from_scene.apply(support_scene)
    collision_tree = cKDTree(support_court)

    captured_poses = tuple(
        _camera_pose_in_court(camera, court_from_scene) for camera in camera_tuple
    )
    captured_cloud = _pose_cloud(captured_poses)
    safe_anchors: list[tuple[SceneCamera, _Pose]] = []
    for camera, pose in zip(camera_tuple, captured_poses, strict=True):
        _, reason = _evaluate_pose(
            camera,
            pose,
            court_keypoints,
            collision_tree,
            limits,
        )
        if reason is None:
            safe_anchors.append((camera, pose))
    if not safe_anchors:
        raise ValueError("No captured camera passes every pre-registered safety gate.")

    rng = np.random.default_rng(seed)
    candidates: list[_Candidate] = []
    rejections: Counter[str] = Counter()
    rotation_limit_rad = np.deg2rad(limits.rotation_limit_deg)
    for anchor, anchor_pose in safe_anchors:
        for proposal_index in range(proposals_per_anchor):
            unit_delta = _sample_unit_ball(rng, dimension=6)
            local_translation = (
                unit_delta[:3] * limits.translation_limit_m
            )
            local_rotation_vector = unit_delta[3:] * rotation_limit_rad
            center = (
                anchor_pose.center
                + anchor_pose.rotation @ local_translation
            )
            rotation = (
                anchor_pose.rotation
                @ Rotation.from_rotvec(local_rotation_vector).as_matrix()
            )
            pose = _Pose(center=center, rotation=rotation)

            nearest_translation, nearest_rotation, support_score = (
                _nearest_pose_support(
                    pose,
                    captured_cloud,
                    limits,
                )
            )
            if support_score > limits.support_score_limit + 1.0e-10:
                rejections["extrapolation"] += 1
                continue
            projection, reason = _evaluate_pose(
                anchor,
                pose,
                court_keypoints,
                collision_tree,
                limits,
            )
            if reason is not None or projection is None:
                rejections[reason or "unknown"] += 1
                continue

            candidates.append(
                _Candidate(
                    proposal_id=(
                        f"{anchor.camera_id}-proposal-{proposal_index:04d}"
                    ),
                    anchor=anchor,
                    pose=pose,
                    projection=projection,
                    translation_from_anchor_m=float(
                        np.linalg.norm(center - anchor_pose.center)
                    ),
                    rotation_from_anchor_deg=_rotation_distance_deg(
                        anchor_pose.rotation,
                        rotation,
                    ),
                    nearest_captured_translation_m=nearest_translation,
                    nearest_captured_rotation_deg=nearest_rotation,
                    extrapolation_score=support_score,
                )
            )

    proposal_count = len(safe_anchors) * proposals_per_anchor
    if len(candidates) < max_views:
        raise ValueError(
            f"Only {len(candidates)} of {proposal_count} proposals passed all "
            f"gates; max_views={max_views}. Rejections: "
            f"{dict(sorted(rejections.items()))}."
        )

    selected_indices = _farthest_view_selection(
        candidates,
        max_views=max_views,
        thresholds=limits,
    )
    selected = tuple(
        _publish_candidate(candidate, index, court_from_scene)
        for index, candidate in enumerate(
            candidates[value] for value in selected_indices
        )
    )
    return NovelViewSamplingResult(
        seed=seed,
        safe_anchor_count=len(safe_anchors),
        proposal_count=proposal_count,
        accepted_candidate_count=len(candidates),
        rejection_counts=tuple(sorted(rejections.items())),
        selected=selected,
    )


def pose_distance_score(
    first_camera_to_court: NDArray[np.floating],
    second_camera_to_court: NDArray[np.floating],
    thresholds: NovelViewThresholds,
) -> float:
    """Return the normalized coupled translation/rotation pose distance."""
    first = _pose_from_matrix(first_camera_to_court, name="first_camera_to_court")
    second = _pose_from_matrix(
        second_camera_to_court,
        name="second_camera_to_court",
    )
    translation = float(np.linalg.norm(first.center - second.center))
    rotation = _rotation_distance_deg(first.rotation, second.rotation)
    return float(
        np.hypot(
            translation / thresholds.translation_limit_m,
            rotation / thresholds.rotation_limit_deg,
        )
    )


def _camera_pose_in_court(
    camera: SceneCamera,
    court_from_scene: SimilarityTransform,
) -> _Pose:
    camera_to_scene = np.asarray(
        camera.camera_to_scene,
        dtype=np.float64,
    ).reshape(4, 4)
    scene_rotation_to_court = np.asarray(
        court_from_scene.rotation,
        dtype=np.float64,
    ).reshape(3, 3)
    center = court_from_scene.apply(camera_to_scene[None, :3, 3])[0]
    rotation = scene_rotation_to_court @ camera_to_scene[:3, :3]
    return _Pose(center=center, rotation=rotation)


def _pose_from_matrix(
    matrix: NDArray[np.floating],
    *,
    name: str,
) -> _Pose:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix.")
    if not np.allclose(value[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-9, rtol=0.0):
        raise ValueError(f"{name} must have homogeneous bottom row.")
    _validate_rotation(value[:3, :3], name=f"{name} rotation")
    return _Pose(center=value[:3, 3].copy(), rotation=value[:3, :3].copy())


def _evaluate_pose(
    camera: SceneCamera,
    pose: _Pose,
    court_keypoints: NDArray[np.float64],
    collision_tree: cKDTree,
    thresholds: NovelViewThresholds,
) -> tuple[_ProjectionEvidence | None, str | None]:
    if pose.center[2] < thresholds.min_camera_height_m:
        return None, "camera_height"

    collision_distances, _ = collision_tree.query(
        pose.center,
        k=thresholds.collision_neighbor_rank,
    )
    collision_clearance = float(np.atleast_1d(collision_distances)[-1])
    if collision_clearance < thresholds.min_collision_clearance_m:
        return None, "collision"

    points_camera = (court_keypoints - pose.center) @ pose.rotation
    depth = points_camera[:, 2]
    if float(np.min(depth)) <= thresholds.near_plane_m:
        return None, "near_plane"

    intrinsics = np.asarray(camera.intrinsics, dtype=np.float64).reshape(3, 3)
    homogeneous = points_camera @ intrinsics.T
    uv = homogeneous[:, :2] / depth[:, None]
    margins = np.minimum.reduce(
        (
            uv[:, 0],
            float(camera.width - 1) - uv[:, 0],
            uv[:, 1],
            float(camera.height - 1) - uv[:, 1],
        )
    )
    visible = (depth > thresholds.near_plane_m) & (
        margins >= thresholds.min_image_margin_px
    )
    if int(np.count_nonzero(visible[:_LINE_KEYPOINT_COUNT])) < (
        thresholds.min_line_keypoints_visible
    ):
        return None, "court_framing"

    return (
        _ProjectionEvidence(
            uv=uv,
            depth=depth,
            visible=visible,
            collision_clearance_m=collision_clearance,
            min_line_margin_px=float(np.min(margins[:_LINE_KEYPOINT_COUNT])),
        ),
        None,
    )


def _nearest_pose_support(
    pose: _Pose,
    captured: _PoseCloud,
    thresholds: NovelViewThresholds,
) -> tuple[float, float, float]:
    translations = np.linalg.norm(captured.centers - pose.center, axis=1)
    traces = np.einsum(
        "ij,kij->k",
        pose.rotation,
        captured.rotations,
    )
    cosines = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
    rotations = np.degrees(np.arccos(cosines))
    scores = np.hypot(
        translations / thresholds.translation_limit_m,
        rotations / thresholds.rotation_limit_deg,
    )
    index = int(np.argmin(scores))
    return (
        float(translations[index]),
        float(rotations[index]),
        float(scores[index]),
    )


def _farthest_view_selection(
    candidates: Sequence[_Candidate],
    *,
    max_views: int,
    thresholds: NovelViewThresholds,
) -> list[int]:
    minimum_distances = np.asarray(
        [candidate.extrapolation_score for candidate in candidates],
        dtype=np.float64,
    )
    candidate_cloud = _pose_cloud(
        tuple(candidate.pose for candidate in candidates)
    )
    available: NDArray[np.bool_] = np.ones(len(candidates), dtype=np.bool_)
    selected: list[int] = []
    for _ in range(max_views):
        scores = np.where(available, minimum_distances, -np.inf)
        index = int(np.argmax(scores))
        if not np.isfinite(scores[index]):
            raise RuntimeError("Farthest-view selection exhausted candidates.")
        selected.append(index)
        available[index] = False
        chosen = candidates[index].pose
        translations = np.linalg.norm(
            candidate_cloud.centers - chosen.center,
            axis=1,
        )
        traces = np.einsum(
            "ij,kij->k",
            chosen.rotation,
            candidate_cloud.rotations,
        )
        rotations = np.degrees(
            np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0))
        )
        distances = np.hypot(
            translations / thresholds.translation_limit_m,
            rotations / thresholds.rotation_limit_deg,
        )
        minimum_distances = np.minimum(minimum_distances, distances)
    return selected


def _pose_cloud(poses: Sequence[_Pose]) -> _PoseCloud:
    return _PoseCloud(
        centers=np.stack([pose.center for pose in poses]),
        rotations=np.stack([pose.rotation for pose in poses]),
    )


def _publish_candidate(
    candidate: _Candidate,
    selected_index: int,
    court_from_scene: SimilarityTransform,
) -> NovelViewCamera:
    camera_to_court = np.eye(4, dtype=np.float64)
    camera_to_court[:3, :3] = candidate.pose.rotation
    camera_to_court[:3, 3] = candidate.pose.center

    scene_from_court = court_from_scene.inverse()
    scene_rotation_from_court = np.asarray(
        scene_from_court.rotation,
        dtype=np.float64,
    ).reshape(3, 3)
    camera_to_scene = np.eye(4, dtype=np.float64)
    camera_to_scene[:3, :3] = (
        scene_rotation_from_court @ candidate.pose.rotation
    )
    camera_to_scene[:3, 3] = scene_from_court.apply(
        candidate.pose.center[None]
    )[0]
    _validate_rotation(
        camera_to_scene[:3, :3],
        name="published camera_to_scene rotation",
    )

    projection = candidate.projection
    return NovelViewCamera(
        camera_id=f"novel_{selected_index:06d}",
        anchor_camera_id=candidate.anchor.camera_id,
        width=candidate.anchor.width,
        height=candidate.anchor.height,
        intrinsics=candidate.anchor.intrinsics,
        camera_to_court=tuple(float(value) for value in camera_to_court.ravel()),
        camera_to_scene=tuple(float(value) for value in camera_to_scene.ravel()),
        court_keypoints_uv=tuple(float(value) for value in projection.uv.ravel()),
        court_keypoints_depth_m=tuple(float(value) for value in projection.depth),
        court_keypoints_visible=tuple(bool(value) for value in projection.visible),
        translation_from_anchor_m=candidate.translation_from_anchor_m,
        rotation_from_anchor_deg=candidate.rotation_from_anchor_deg,
        nearest_captured_translation_m=(
            candidate.nearest_captured_translation_m
        ),
        nearest_captured_rotation_deg=candidate.nearest_captured_rotation_deg,
        extrapolation_score=candidate.extrapolation_score,
        collision_clearance_m=projection.collision_clearance_m,
        min_court_depth_m=float(np.min(projection.depth)),
        min_line_margin_px=projection.min_line_margin_px,
    )


def _sample_unit_ball(
    rng: np.random.Generator,
    *,
    dimension: int,
) -> NDArray[np.float64]:
    direction = rng.normal(size=dimension)
    norm = float(np.linalg.norm(direction))
    while norm <= _EPSILON:
        direction = rng.normal(size=dimension)
        norm = float(np.linalg.norm(direction))
    radius = float(rng.random()) ** (1.0 / dimension)
    return np.asarray(radius * direction / norm, dtype=np.float64)


def _rotation_distance_deg(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _finite_points(
    value: NDArray[np.floating],
    *,
    name: str,
    minimum_count: int,
) -> NDArray[np.float64]:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3], got {points.shape}.")
    if points.shape[0] < minimum_count:
        raise ValueError(f"{name} must contain at least {minimum_count} points.")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain only finite values.")
    return points


def _validate_rotation(rotation: NDArray[np.float64], *, name: str) -> None:
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=1.0e-7,
        rtol=0.0,
    ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-7, rtol=0.0):
        raise ValueError(f"{name} must be a proper rotation.")
