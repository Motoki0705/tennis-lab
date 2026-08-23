"""Versioned Court labels with renderer-derived point visibility."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TypeAlias, overload

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.court.components.camera_view import (
    AMBIGUOUS_CAMERA_RELATIVE_NEAR_FAR_REASON,
    CAMERA_VIEW_MID_PLANE_TOLERANCE_M,
    AmbiguousCameraRelativeNearFarError,
    camera_view_canonicalization,
    validate_finite_camera_view_projection,
)
from src.synthetic_data_generation.dataset.court.schema import (
    COURT_PHYSICAL_INDICES_BY_CLASS_V1,
    COURT_SEMANTIC_CLASS_NAMES_V1,
    COURT_SEMANTIC_CLASS_NAMES_V2,
    CourtDatasetSchemaVersion,
)
from src.synthetic_data_generation.scene_contract import (
    CourtInstance,
    MultiCourtLayout,
    RigidTransform,
    SceneCamera,
)
from src.utils.schema.court import (
    CAMERA_VIEW_HALF_TURN_INDEX,
    NUM_GROUND_COURT_KP,
    OPPOSITE_COURT_END_INDEX,
    STANDARD_COURT_CONFIG,
    court_keypoints_3d,
)

SEMANTIC_CLASS_NAMES = COURT_SEMANTIC_CLASS_NAMES_V1
PHYSICAL_INDICES_BY_CLASS: tuple[tuple[int, int], ...] = tuple(
    (indices[0], indices[1]) for indices in COURT_PHYSICAL_INDICES_BY_CLASS_V1
)
SEMANTIC_CLASS_NAMES_V2 = COURT_SEMANTIC_CLASS_NAMES_V2
CAMERA_RELATIVE_MID_PLANE_TOLERANCE_M = CAMERA_VIEW_MID_PLANE_TOLERANCE_M
PUBLISHED_COURT_GEOMETRY_ATOL_M = 1.0e-6


def coverage_mode_from_in_frame_point_count(in_frame_point_count: int) -> str:
    """Return the canonical coverage mode for the 14 physical court points."""
    if (
        isinstance(in_frame_point_count, bool)
        or not isinstance(in_frame_point_count, int)
        or not 0 <= in_frame_point_count <= NUM_GROUND_COURT_KP
    ):
        raise ValueError("Court in-frame point count must be an integer in [0, 14].")
    if in_frame_point_count == NUM_GROUND_COURT_KP:
        return "full"
    if in_frame_point_count >= 10:
        return "near_full"
    if in_frame_point_count >= 4:
        return "partial"
    if in_frame_point_count >= 1:
        return "sparse"
    return "none"


@dataclass(frozen=True, slots=True)
class SemanticPoint:
    """One physical line keypoint belonging to a symmetric semantic class."""

    physical_index: int
    uv: tuple[float, float]
    camera_depth_m: float
    scene_xyz_m: tuple[float, float, float]
    in_front: bool
    in_frame: bool
    renderer_visible: bool | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe label point."""
        return {
            "physical_index": self.physical_index,
            "uv": list(self.uv),
            "camera_depth_m": self.camera_depth_m,
            "scene_xyz_m": list(self.scene_xyz_m),
            "in_front": self.in_front,
            "in_frame": self.in_frame,
            "renderer_visible": self.renderer_visible,
        }


@dataclass(frozen=True, slots=True)
class SemanticClass:
    """Exactly two unordered physical points in one of seven target classes."""

    class_id: int
    class_name: str
    points: tuple[SemanticPoint, SemanticPoint]

    def __post_init__(self) -> None:
        if not 0 <= self.class_id < len(SEMANTIC_CLASS_NAMES):
            raise ValueError("class_id is outside the seven-class contract.")
        if self.class_name != SEMANTIC_CLASS_NAMES[self.class_id]:
            raise ValueError("class_name disagrees with class_id.")

    @property
    def renderer_visible(self) -> bool:
        """Return whether this class has visible supervision in the render."""
        return any(point.renderer_visible is True for point in self.points)

    def to_dict(self) -> dict[str, object]:
        """Return the complete two-point semantic class."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "renderer_visible": self.renderer_visible,
            "points": [point.to_dict() for point in self.points],
        }


@dataclass(frozen=True, slots=True)
class CourtProjection:
    """Seven semantic classes projected for one accepted court."""

    court_instance_id: str
    classes: tuple[SemanticClass, ...]

    def __post_init__(self) -> None:
        if not self.court_instance_id.strip():
            raise ValueError("court_instance_id must be non-empty.")
        if tuple(value.class_id for value in self.classes) != tuple(range(7)):
            raise ValueError("Court classes must be ordered exactly 0..6.")

    @property
    def in_frame_point_count(self) -> int:
        """Return the number of projected physical points inside the image."""
        return sum(point.in_frame for value in self.classes for point in value.points)

    @property
    def coverage_mode(self) -> str:
        """Classify geometric coverage without inventing renderer visibility."""
        return coverage_mode_from_in_frame_point_count(self.in_frame_point_count)

    def to_dict(self) -> dict[str, object]:
        """Return complete instance-aware supervision."""
        return {
            "court_instance_id": self.court_instance_id,
            "coverage_mode": self.coverage_mode,
            "classes": [value.to_dict() for value in self.classes],
        }


@dataclass(frozen=True, slots=True)
class MultiCourtProjection:
    """Renderer-visible supervision for all accepted courts and seven classes."""

    camera_id: str
    width: int
    height: int
    courts: tuple[CourtProjection, ...]

    def __post_init__(self) -> None:
        if not self.camera_id.strip() or self.width <= 1 or self.height <= 1:
            raise ValueError("Projection requires a camera ID and valid resolution.")
        court_ids = [court.court_instance_id for court in self.courts]
        if not court_ids or len(court_ids) != len(set(court_ids)):
            raise ValueError("Projection court IDs must be non-empty and unique.")

    @property
    def visible_class_names(self) -> tuple[str, ...]:
        """Return stable unique class names with renderer-visible supervision."""
        visible = {
            value.class_name
            for court in self.courts
            for value in court.classes
            if value.renderer_visible
        }
        return tuple(name for name in SEMANTIC_CLASS_NAMES if name in visible)

    @property
    def visible_point_count(self) -> int:
        """Return total renderer-visible physical points."""
        return sum(
            point.renderer_visible is True
            for court in self.courts
            for value in court.classes
            for point in value.points
        )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical seven-class label payload."""
        return {
            "camera_id": self.camera_id,
            "resolution": [self.width, self.height],
            "coverage_modes": [court.coverage_mode for court in self.courts],
            "visible_class_names": list(self.visible_class_names),
            "visible_point_count": self.visible_point_count,
            "courts": [court.to_dict() for court in self.courts],
        }


@dataclass(frozen=True, slots=True)
class SemanticClassV2:
    """Exactly one physical point in one camera-relative v2 channel."""

    class_id: int
    class_name: str
    points: tuple[SemanticPoint]

    def __post_init__(self) -> None:
        if not 0 <= self.class_id < len(SEMANTIC_CLASS_NAMES_V2):
            raise ValueError("class_id is outside the fourteen-class contract.")
        if self.class_name != SEMANTIC_CLASS_NAMES_V2[self.class_id]:
            raise ValueError("class_name disagrees with class_id.")
        if len(self.points) != 1:
            raise ValueError("Every v2 semantic class must contain exactly one point.")

    @property
    def renderer_visible(self) -> bool:
        """Return whether the singleton has visible renderer supervision."""
        return self.points[0].renderer_visible is True

    def to_dict(self) -> dict[str, object]:
        """Return the complete singleton semantic class."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "renderer_visible": self.renderer_visible,
            "points": [self.points[0].to_dict()],
        }


@dataclass(frozen=True, slots=True)
class CourtProjectionV2:
    """Fourteen camera-relative singleton classes for one accepted court."""

    court_instance_id: str
    classes: tuple[SemanticClassV2, ...]

    def __post_init__(self) -> None:
        if not self.court_instance_id.strip():
            raise ValueError("court_instance_id must be non-empty.")
        if tuple(value.class_id for value in self.classes) != tuple(
            range(NUM_GROUND_COURT_KP)
        ):
            raise ValueError("V2 Court classes must be ordered exactly 0..13.")
        physical_indices = tuple(
            semantic_class.points[0].physical_index for semantic_class in self.classes
        )
        if set(physical_indices) != set(range(NUM_GROUND_COURT_KP)):
            raise ValueError(
                "V2 Court classes must preserve each physical index 0..13 once."
            )
        if physical_indices not in (
            tuple(range(NUM_GROUND_COURT_KP)),
            OPPOSITE_COURT_END_INDEX,
        ):
            raise ValueError(
                "V2 physical indices are not a camera-relative permutation."
            )

    @property
    def in_frame_point_count(self) -> int:
        """Return the number of projected physical points inside the image."""
        return sum(value.points[0].in_frame for value in self.classes)

    @property
    def coverage_mode(self) -> str:
        """Classify geometric coverage using the unchanged 14-point inventory."""
        return coverage_mode_from_in_frame_point_count(self.in_frame_point_count)

    def to_dict(self) -> dict[str, object]:
        """Return complete camera-relative supervision."""
        return {
            "court_instance_id": self.court_instance_id,
            "coverage_mode": self.coverage_mode,
            "classes": [value.to_dict() for value in self.classes],
        }


@dataclass(frozen=True, slots=True)
class MultiCourtProjectionV2:
    """V2 singleton supervision for every accepted court in one camera."""

    camera_id: str
    width: int
    height: int
    courts: tuple[CourtProjectionV2, ...]

    def __post_init__(self) -> None:
        if not self.camera_id.strip() or self.width <= 1 or self.height <= 1:
            raise ValueError("Projection requires a camera ID and valid resolution.")
        court_ids = [court.court_instance_id for court in self.courts]
        if not court_ids or len(court_ids) != len(set(court_ids)):
            raise ValueError("Projection court IDs must be non-empty and unique.")

    @property
    def visible_class_names(self) -> tuple[str, ...]:
        """Return ordered v2 channels with renderer-visible supervision."""
        visible = {
            value.class_name
            for court in self.courts
            for value in court.classes
            if value.renderer_visible
        }
        return tuple(name for name in SEMANTIC_CLASS_NAMES_V2 if name in visible)

    @property
    def visible_point_count(self) -> int:
        """Return total renderer-visible physical points."""
        return sum(
            value.points[0].renderer_visible is True
            for court in self.courts
            for value in court.classes
        )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical fourteen-class label payload."""
        return {
            "camera_id": self.camera_id,
            "resolution": [self.width, self.height],
            "coverage_modes": [court.coverage_mode for court in self.courts],
            "visible_class_names": list(self.visible_class_names),
            "visible_point_count": self.visible_point_count,
            "courts": [court.to_dict() for court in self.courts],
        }


@dataclass(frozen=True, slots=True)
class CourtProjectionV3:
    """Corrected camera-view singleton classes for one accepted court."""

    court_instance_id: str
    classes: tuple[SemanticClassV2, ...]

    def __post_init__(self) -> None:
        if not self.court_instance_id.strip():
            raise ValueError("court_instance_id must be non-empty.")
        if tuple(value.class_id for value in self.classes) != tuple(
            range(NUM_GROUND_COURT_KP)
        ):
            raise ValueError("V3 Court classes must be ordered exactly 0..13.")
        physical_indices = tuple(
            semantic_class.points[0].physical_index for semantic_class in self.classes
        )
        if set(physical_indices) != set(range(NUM_GROUND_COURT_KP)):
            raise ValueError(
                "V3 Court classes must preserve each physical index 0..13 once."
            )
        if physical_indices not in (
            tuple(range(NUM_GROUND_COURT_KP)),
            CAMERA_VIEW_HALF_TURN_INDEX,
        ):
            raise ValueError(
                "V3 physical indices are not a camera-view full-half-turn permutation."
            )
        validate_finite_camera_view_projection(
            np.asarray(
                [semantic_class.points[0].uv for semantic_class in self.classes],
                dtype=np.float64,
            )
        )

    @property
    def in_frame_point_count(self) -> int:
        """Return the number of projected physical points inside the image."""
        return sum(value.points[0].in_frame for value in self.classes)

    @property
    def coverage_mode(self) -> str:
        """Classify geometric coverage using the unchanged 14-point inventory."""
        return coverage_mode_from_in_frame_point_count(self.in_frame_point_count)

    def to_dict(self) -> dict[str, object]:
        """Return complete corrected camera-view supervision."""
        return {
            "court_instance_id": self.court_instance_id,
            "coverage_mode": self.coverage_mode,
            "classes": [value.to_dict() for value in self.classes],
        }


@dataclass(frozen=True, slots=True)
class MultiCourtProjectionV3:
    """V3 camera-view supervision for every accepted court in one camera."""

    camera_id: str
    width: int
    height: int
    courts: tuple[CourtProjectionV3, ...]

    def __post_init__(self) -> None:
        if not self.camera_id.strip() or self.width <= 1 or self.height <= 1:
            raise ValueError("Projection requires a camera ID and valid resolution.")
        court_ids = [court.court_instance_id for court in self.courts]
        if not court_ids or len(court_ids) != len(set(court_ids)):
            raise ValueError("Projection court IDs must be non-empty and unique.")

    @property
    def visible_class_names(self) -> tuple[str, ...]:
        """Return ordered V3 channels with renderer-visible supervision."""
        visible = {
            value.class_name
            for court in self.courts
            for value in court.classes
            if value.renderer_visible
        }
        return tuple(name for name in SEMANTIC_CLASS_NAMES_V2 if name in visible)

    @property
    def visible_point_count(self) -> int:
        """Return total renderer-visible physical points."""
        return sum(
            value.points[0].renderer_visible is True
            for court in self.courts
            for value in court.classes
        )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical corrected fourteen-class label payload."""
        return {
            "camera_id": self.camera_id,
            "resolution": [self.width, self.height],
            "coverage_modes": [court.coverage_mode for court in self.courts],
            "visible_class_names": list(self.visible_class_names),
            "visible_point_count": self.visible_point_count,
            "courts": [court.to_dict() for court in self.courts],
        }


CourtProjectionAny: TypeAlias = CourtProjection | CourtProjectionV2 | CourtProjectionV3
MultiCourtProjectionAny: TypeAlias = (
    MultiCourtProjection | MultiCourtProjectionV2 | MultiCourtProjectionV3
)


def scene_from_court_from_published_points(
    points_by_physical_index: Mapping[int, Sequence[float]],
) -> RigidTransform:
    """Recover one court transform from the named published ground geometry."""
    expected_indices = set(range(NUM_GROUND_COURT_KP))
    if set(points_by_physical_index) != expected_indices:
        raise ValueError(
            "Published Court v2 geometry must contain physical indices 0..13 once."
        )
    points_court = np.asarray(
        court_keypoints_3d(STANDARD_COURT_CONFIG)[:NUM_GROUND_COURT_KP].numpy(),
        dtype=np.float64,
    )
    points_scene = np.asarray(
        [points_by_physical_index[index] for index in range(NUM_GROUND_COURT_KP)],
        dtype=np.float64,
    )
    if points_scene.shape != points_court.shape or not np.isfinite(points_scene).all():
        raise ValueError("Published Court v2 scene points must be finite three-vectors.")

    court_center = np.mean(points_court, axis=0)
    scene_center = np.mean(points_scene, axis=0)
    court_centered = points_court - court_center
    scene_centered = points_scene - scene_center
    left, singular_values, right_transpose = np.linalg.svd(
        court_centered.T @ scene_centered
    )
    if singular_values[1] <= 1.0e-12:
        raise ValueError("Published Court v2 geometry is degenerate.")
    rotation = right_transpose.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_transpose[-1, :] *= -1.0
        rotation = right_transpose.T @ left.T
    translation = scene_center - rotation @ court_center
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    transform = RigidTransform.from_matrix(matrix)
    reconstructed = transform.apply(points_court)
    if not np.allclose(
        reconstructed,
        points_scene,
        atol=PUBLISHED_COURT_GEOMETRY_ATOL_M,
        rtol=0.0,
    ):
        raise ValueError(
            "Published Court v2 scene points do not define one rigid court geometry."
        )
    return transform


def camera_center_court_y(
    camera: SceneCamera,
    *,
    scene_from_court: RigidTransform,
) -> float:
    """Return the persisted camera centre's local court Y coordinate."""
    if not isinstance(camera, SceneCamera):
        raise TypeError("camera must be a SceneCamera.")
    if not isinstance(scene_from_court, RigidTransform):
        raise TypeError("scene_from_court must be a RigidTransform.")
    camera_center_scene = camera.camera_to_scene.matrix()[:3, 3]
    camera_center_court = scene_from_court.inverse().apply(
        camera_center_scene.reshape(1, 3)
    )[0]
    return float(camera_center_court[1])


def camera_relative_physical_indices(
    camera: SceneCamera,
    court: CourtInstance,
) -> tuple[int, ...]:
    """Resolve v2 near/far from camera position in one court coordinate frame."""
    if not isinstance(camera, SceneCamera):
        raise TypeError("camera must be a SceneCamera.")
    if not isinstance(court, CourtInstance):
        raise TypeError("court must be a CourtInstance.")
    camera_center_scene = camera.camera_to_scene.matrix()[:3, 3]
    camera_center_court = court.court_from_scene.apply(
        camera_center_scene.reshape(1, 3)
    )[0]
    local_y = float(camera_center_court[1])
    if abs(local_y) <= CAMERA_RELATIVE_MID_PLANE_TOLERANCE_M:
        raise AmbiguousCameraRelativeNearFarError(court.court_instance_id)
    if local_y < 0.0:
        return tuple(range(NUM_GROUND_COURT_KP))
    return OPPOSITE_COURT_END_INDEX


def project_court_semantics(
    camera: SceneCamera,
    layout: MultiCourtLayout,
    *,
    near_plane_m: float = 0.01,
) -> MultiCourtProjection:
    """Project all accepted courts using the canonical camera/transform contract."""
    if not math.isfinite(near_plane_m) or near_plane_m <= 0.0:
        raise ValueError("near_plane_m must be positive and finite.")
    points_court = np.asarray(
        court_keypoints_3d(STANDARD_COURT_CONFIG)[:14].numpy(),
        dtype=np.float64,
    )
    scene_to_camera = camera.camera_to_scene.inverse()
    intrinsics = np.asarray(camera.intrinsics, dtype=np.float64).reshape(3, 3)
    court_records: list[CourtProjection] = []
    for court in layout.courts:
        points_scene = court.scene_from_court.apply(points_court)
        points_camera = scene_to_camera.apply(points_scene)
        depth = points_camera[:, 2]
        homogeneous = points_camera @ intrinsics.T
        safe_depth = np.where(np.abs(depth) > 1.0e-12, depth, np.nan)
        uv = homogeneous[:, :2] / safe_depth[:, None]
        in_front = depth > near_plane_m
        in_frame = (
            in_front
            & np.isfinite(uv).all(axis=1)
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] < camera.width)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] < camera.height)
        )
        classes: list[SemanticClass] = []
        for class_id, physical_indices in enumerate(PHYSICAL_INDICES_BY_CLASS):
            points_list = [
                SemanticPoint(
                    physical_index=physical_index,
                    uv=(float(uv[physical_index, 0]), float(uv[physical_index, 1])),
                    camera_depth_m=float(depth[physical_index]),
                    scene_xyz_m=(
                        float(points_scene[physical_index, 0]),
                        float(points_scene[physical_index, 1]),
                        float(points_scene[physical_index, 2]),
                    ),
                    in_front=bool(in_front[physical_index]),
                    in_frame=bool(in_frame[physical_index]),
                    renderer_visible=None,
                )
                for physical_index in physical_indices
            ]
            points = (points_list[0], points_list[1])
            classes.append(
                SemanticClass(
                    class_id=class_id,
                    class_name=SEMANTIC_CLASS_NAMES[class_id],
                    points=points,
                )
            )
        court_records.append(
            CourtProjection(
                court_instance_id=court.court_instance_id,
                classes=tuple(classes),
            )
        )
    return MultiCourtProjection(
        camera_id=camera.camera_id,
        width=camera.width,
        height=camera.height,
        courts=tuple(court_records),
    )


def project_court_semantics_v2(
    camera: SceneCamera,
    layout: MultiCourtLayout,
    *,
    near_plane_m: float = 0.01,
) -> MultiCourtProjectionV2:
    """Project all courts into 14 camera-relative singleton classes."""
    if not math.isfinite(near_plane_m) or near_plane_m <= 0.0:
        raise ValueError("near_plane_m must be positive and finite.")
    points_court = np.asarray(
        court_keypoints_3d(STANDARD_COURT_CONFIG)[:NUM_GROUND_COURT_KP].numpy(),
        dtype=np.float64,
    )
    scene_to_camera = camera.camera_to_scene.inverse()
    intrinsics = np.asarray(camera.intrinsics, dtype=np.float64).reshape(3, 3)
    court_records: list[CourtProjectionV2] = []
    for court in layout.courts:
        physical_indices = camera_relative_physical_indices(camera, court)
        points_scene = court.scene_from_court.apply(points_court)
        points_camera = scene_to_camera.apply(points_scene)
        depth = points_camera[:, 2]
        homogeneous = points_camera @ intrinsics.T
        safe_depth = np.where(np.abs(depth) > 1.0e-12, depth, np.nan)
        uv = homogeneous[:, :2] / safe_depth[:, None]
        in_front = depth > near_plane_m
        in_frame = (
            in_front
            & np.isfinite(uv).all(axis=1)
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] < camera.width)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] < camera.height)
        )
        classes = tuple(
            SemanticClassV2(
                class_id=class_id,
                class_name=SEMANTIC_CLASS_NAMES_V2[class_id],
                points=(
                    SemanticPoint(
                        physical_index=physical_index,
                        uv=(
                            float(uv[physical_index, 0]),
                            float(uv[physical_index, 1]),
                        ),
                        camera_depth_m=float(depth[physical_index]),
                        scene_xyz_m=(
                            float(points_scene[physical_index, 0]),
                            float(points_scene[physical_index, 1]),
                            float(points_scene[physical_index, 2]),
                        ),
                        in_front=bool(in_front[physical_index]),
                        in_frame=bool(in_frame[physical_index]),
                        renderer_visible=None,
                    ),
                ),
            )
            for class_id, physical_index in enumerate(physical_indices)
        )
        court_records.append(
            CourtProjectionV2(
                court_instance_id=court.court_instance_id,
                classes=classes,
            )
        )
    return MultiCourtProjectionV2(
        camera_id=camera.camera_id,
        width=camera.width,
        height=camera.height,
        courts=tuple(court_records),
    )


def project_court_semantics_v3(
    camera: SceneCamera,
    layout: MultiCourtLayout,
    *,
    near_plane_m: float = 0.01,
) -> MultiCourtProjectionV3:
    """Project all courts with one shared side decision for V3 pose and KP identity."""
    if not isinstance(camera, SceneCamera):
        raise TypeError("camera must be a SceneCamera.")
    if not isinstance(layout, MultiCourtLayout):
        raise TypeError("layout must be a MultiCourtLayout.")
    if not math.isfinite(near_plane_m) or near_plane_m <= 0.0:
        raise ValueError("near_plane_m must be positive and finite.")
    points_court = np.asarray(
        court_keypoints_3d(STANDARD_COURT_CONFIG)[:NUM_GROUND_COURT_KP].numpy(),
        dtype=np.float64,
    )
    scene_to_camera = camera.camera_to_scene.inverse()
    intrinsics = np.asarray(camera.intrinsics, dtype=np.float64).reshape(3, 3)
    court_records: list[CourtProjectionV3] = []
    for court in layout.courts:
        canonicalization = camera_view_canonicalization(camera, court)
        physical_indices = canonicalization.semantic_to_physical
        points_scene = court.scene_from_court.apply(points_court)
        points_camera = scene_to_camera.apply(points_scene)
        depth = points_camera[:, 2]
        homogeneous = points_camera @ intrinsics.T
        safe_depth = np.where(np.abs(depth) > 1.0e-12, depth, np.nan)
        uv = homogeneous[:, :2] / safe_depth[:, None]
        semantic_uv = uv[np.asarray(physical_indices, dtype=np.int64)]
        validate_finite_camera_view_projection(semantic_uv)
        in_front = depth > near_plane_m
        in_frame = (
            in_front
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] < camera.width)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] < camera.height)
        )
        classes = tuple(
            SemanticClassV2(
                class_id=class_id,
                class_name=SEMANTIC_CLASS_NAMES_V2[class_id],
                points=(
                    SemanticPoint(
                        physical_index=physical_index,
                        uv=(
                            float(uv[physical_index, 0]),
                            float(uv[physical_index, 1]),
                        ),
                        camera_depth_m=float(depth[physical_index]),
                        scene_xyz_m=(
                            float(points_scene[physical_index, 0]),
                            float(points_scene[physical_index, 1]),
                            float(points_scene[physical_index, 2]),
                        ),
                        in_front=bool(in_front[physical_index]),
                        in_frame=bool(in_frame[physical_index]),
                        renderer_visible=None,
                    ),
                ),
            )
            for class_id, physical_index in enumerate(physical_indices)
        )
        court_records.append(
            CourtProjectionV3(
                court_instance_id=court.court_instance_id,
                classes=classes,
            )
        )
    return MultiCourtProjectionV3(
        camera_id=camera.camera_id,
        width=camera.width,
        height=camera.height,
        courts=tuple(court_records),
    )


def project_court_semantics_for_version(
    camera: SceneCamera,
    layout: MultiCourtLayout,
    *,
    schema_version: CourtDatasetSchemaVersion,
    near_plane_m: float = 0.01,
) -> MultiCourtProjectionAny:
    """Project from the explicit generation version, never payload shape."""
    if schema_version is CourtDatasetSchemaVersion.V1:
        return project_court_semantics(
            camera,
            layout,
            near_plane_m=near_plane_m,
        )
    if schema_version is CourtDatasetSchemaVersion.V2:
        return project_court_semantics_v2(
            camera,
            layout,
            near_plane_m=near_plane_m,
        )
    if schema_version is CourtDatasetSchemaVersion.V3:
        return project_court_semantics_v3(
            camera,
            layout,
            near_plane_m=near_plane_m,
        )
    raise TypeError("schema_version must be a CourtDatasetSchemaVersion.")


@overload
def attach_renderer_visibility(
    projection: MultiCourtProjection,
    *,
    alpha: NDArray[np.floating],
    depth: NDArray[np.floating],
    alpha_threshold: float = 0.01,
    sample_radius_px: int = 1,
) -> MultiCourtProjection: ...


@overload
def attach_renderer_visibility(
    projection: MultiCourtProjectionV2,
    *,
    alpha: NDArray[np.floating],
    depth: NDArray[np.floating],
    alpha_threshold: float = 0.01,
    sample_radius_px: int = 1,
) -> MultiCourtProjectionV2: ...


@overload
def attach_renderer_visibility(
    projection: MultiCourtProjectionV3,
    *,
    alpha: NDArray[np.floating],
    depth: NDArray[np.floating],
    alpha_threshold: float = 0.01,
    sample_radius_px: int = 1,
) -> MultiCourtProjectionV3: ...


def attach_renderer_visibility(
    projection: MultiCourtProjectionAny,
    *,
    alpha: NDArray[np.floating],
    depth: NDArray[np.floating],
    alpha_threshold: float = 0.01,
    sample_radius_px: int = 1,
) -> MultiCourtProjectionAny:
    """Attach visibility only when NHT alpha and positive depth support the pixel."""
    alpha_array = np.asarray(alpha)
    depth_array = np.asarray(depth)
    expected_shape = (projection.height, projection.width, 1)
    if alpha_array.dtype != np.float32 or depth_array.dtype != np.float32:
        raise TypeError("NHT alpha and depth arrays must have dtype float32.")
    if alpha_array.shape != expected_shape or depth_array.shape != expected_shape:
        raise ValueError(
            f"NHT alpha/depth must have shape {expected_shape}; "
            f"got {alpha_array.shape} and {depth_array.shape}."
        )
    if not np.isfinite(alpha_array).all() or not np.isfinite(depth_array).all():
        raise ValueError("NHT alpha/depth must contain only finite values.")
    if (
        np.any(alpha_array < 0.0)
        or np.any(alpha_array > 1.0)
        or np.any(depth_array < 0.0)
    ):
        raise ValueError("NHT alpha/depth values are outside their semantic ranges.")
    if not math.isfinite(alpha_threshold) or alpha_threshold < 0.0:
        raise ValueError("alpha_threshold must be finite and non-negative.")
    if isinstance(sample_radius_px, bool) or sample_radius_px < 0:
        raise ValueError("sample_radius_px must be a non-negative integer.")

    return attach_renderer_visibility_from_validated_arrays(
        projection,
        alpha=alpha_array,
        depth=depth_array,
        alpha_threshold=alpha_threshold,
        sample_radius_px=sample_radius_px,
    )


@overload
def attach_renderer_visibility_from_validated_arrays(
    projection: MultiCourtProjection,
    *,
    alpha: NDArray[np.float32],
    depth: NDArray[np.float32],
    alpha_threshold: float = 0.01,
    sample_radius_px: int = 1,
) -> MultiCourtProjection: ...


@overload
def attach_renderer_visibility_from_validated_arrays(
    projection: MultiCourtProjectionV2,
    *,
    alpha: NDArray[np.float32],
    depth: NDArray[np.float32],
    alpha_threshold: float = 0.01,
    sample_radius_px: int = 1,
) -> MultiCourtProjectionV2: ...


@overload
def attach_renderer_visibility_from_validated_arrays(
    projection: MultiCourtProjectionV3,
    *,
    alpha: NDArray[np.float32],
    depth: NDArray[np.float32],
    alpha_threshold: float = 0.01,
    sample_radius_px: int = 1,
) -> MultiCourtProjectionV3: ...


def attach_renderer_visibility_from_validated_arrays(
    projection: MultiCourtProjectionAny,
    *,
    alpha: NDArray[np.float32],
    depth: NDArray[np.float32],
    alpha_threshold: float = 0.01,
    sample_radius_px: int = 1,
) -> MultiCourtProjectionAny:
    """Sample visibility from arrays already scanned by the Court assembler.

    This explicit entrypoint avoids a second complete finite/range scan during
    production assembly.  Callers must own and document the preceding strict
    validation pass; independent consumers should use
    :func:`attach_renderer_visibility`.
    """
    alpha_array = np.asarray(alpha)
    depth_array = np.asarray(depth)
    expected_shape = (projection.height, projection.width, 1)
    if alpha_array.dtype != np.float32 or depth_array.dtype != np.float32:
        raise TypeError("Validated NHT alpha/depth arrays must have dtype float32.")
    if alpha_array.shape != expected_shape or depth_array.shape != expected_shape:
        raise ValueError("Validated NHT alpha/depth shapes changed before visibility.")
    if not math.isfinite(alpha_threshold) or alpha_threshold < 0.0:
        raise ValueError("alpha_threshold must be finite and non-negative.")
    if isinstance(sample_radius_px, bool) or sample_radius_px < 0:
        raise ValueError("sample_radius_px must be a non-negative integer.")

    if isinstance(projection, MultiCourtProjectionV2):
        return _attach_renderer_visibility_v2(
            projection,
            alpha=alpha,
            depth=depth,
            alpha_threshold=alpha_threshold,
            sample_radius_px=sample_radius_px,
        )
    if isinstance(projection, MultiCourtProjectionV3):
        return _attach_renderer_visibility_v3(
            projection,
            alpha=alpha,
            depth=depth,
            alpha_threshold=alpha_threshold,
            sample_radius_px=sample_radius_px,
        )

    courts: list[CourtProjection] = []
    for court in projection.courts:
        classes: list[SemanticClass] = []
        for semantic_class in court.classes:
            points: list[SemanticPoint] = []
            for point in semantic_class.points:
                visible = False
                if point.in_frame:
                    x = int(round(point.uv[0]))
                    y = int(round(point.uv[1]))
                    x0 = max(0, x - sample_radius_px)
                    x1 = min(projection.width, x + sample_radius_px + 1)
                    y0 = max(0, y - sample_radius_px)
                    y1 = min(projection.height, y + sample_radius_px + 1)
                    local_alpha = alpha_array[y0:y1, x0:x1, 0]
                    local_depth = depth_array[y0:y1, x0:x1, 0]
                    visible = bool(
                        np.any((local_alpha >= alpha_threshold) & (local_depth > 0.0))
                    )
                points.append(replace(point, renderer_visible=visible))
            classes.append(replace(semantic_class, points=(points[0], points[1])))
        courts.append(replace(court, classes=tuple(classes)))
    return replace(projection, courts=tuple(courts))


def _attach_renderer_visibility_v2(
    projection: MultiCourtProjectionV2,
    *,
    alpha: NDArray[np.float32],
    depth: NDArray[np.float32],
    alpha_threshold: float,
    sample_radius_px: int,
) -> MultiCourtProjectionV2:
    courts: list[CourtProjectionV2] = []
    for court in projection.courts:
        classes: list[SemanticClassV2] = []
        for semantic_class in court.classes:
            point = semantic_class.points[0]
            visible = False
            if point.in_frame:
                x = int(round(point.uv[0]))
                y = int(round(point.uv[1]))
                x0 = max(0, x - sample_radius_px)
                x1 = min(projection.width, x + sample_radius_px + 1)
                y0 = max(0, y - sample_radius_px)
                y1 = min(projection.height, y + sample_radius_px + 1)
                local_alpha = alpha[y0:y1, x0:x1, 0]
                local_depth = depth[y0:y1, x0:x1, 0]
                visible = bool(
                    np.any((local_alpha >= alpha_threshold) & (local_depth > 0.0))
                )
            classes.append(
                replace(
                    semantic_class,
                    points=(replace(point, renderer_visible=visible),),
                )
            )
        courts.append(replace(court, classes=tuple(classes)))
    return replace(projection, courts=tuple(courts))


def _attach_renderer_visibility_v3(
    projection: MultiCourtProjectionV3,
    *,
    alpha: NDArray[np.float32],
    depth: NDArray[np.float32],
    alpha_threshold: float,
    sample_radius_px: int,
) -> MultiCourtProjectionV3:
    courts: list[CourtProjectionV3] = []
    for court in projection.courts:
        classes: list[SemanticClassV2] = []
        for semantic_class in court.classes:
            point = semantic_class.points[0]
            visible = False
            if point.in_frame:
                x = int(round(point.uv[0]))
                y = int(round(point.uv[1]))
                x0 = max(0, x - sample_radius_px)
                x1 = min(projection.width, x + sample_radius_px + 1)
                y0 = max(0, y - sample_radius_px)
                y1 = min(projection.height, y + sample_radius_px + 1)
                local_alpha = alpha[y0:y1, x0:x1, 0]
                local_depth = depth[y0:y1, x0:x1, 0]
                visible = bool(
                    np.any((local_alpha >= alpha_threshold) & (local_depth > 0.0))
                )
            classes.append(
                replace(
                    semantic_class,
                    points=(replace(point, renderer_visible=visible),),
                )
            )
        courts.append(replace(court, classes=tuple(classes)))
    return replace(projection, courts=tuple(courts))


__all__ = [
    "AMBIGUOUS_CAMERA_RELATIVE_NEAR_FAR_REASON",
    "CAMERA_RELATIVE_MID_PLANE_TOLERANCE_M",
    "AmbiguousCameraRelativeNearFarError",
    "CourtProjection",
    "CourtProjectionAny",
    "CourtProjectionV2",
    "CourtProjectionV3",
    "MultiCourtProjection",
    "MultiCourtProjectionAny",
    "MultiCourtProjectionV2",
    "MultiCourtProjectionV3",
    "PHYSICAL_INDICES_BY_CLASS",
    "SEMANTIC_CLASS_NAMES",
    "SEMANTIC_CLASS_NAMES_V2",
    "SemanticClass",
    "SemanticClassV2",
    "SemanticPoint",
    "attach_renderer_visibility",
    "attach_renderer_visibility_from_validated_arrays",
    "camera_center_court_y",
    "camera_relative_physical_indices",
    "coverage_mode_from_in_frame_point_count",
    "project_court_semantics",
    "project_court_semantics_for_version",
    "project_court_semantics_v2",
    "project_court_semantics_v3",
    "scene_from_court_from_published_points",
]
