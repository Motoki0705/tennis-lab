"""Seven-class Court labels with renderer-derived point visibility."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.scene_contract import MultiCourtLayout, SceneCamera
from src.utils.schema.court import STANDARD_COURT_CONFIG, court_keypoints_3d

SEMANTIC_CLASS_NAMES: tuple[str, ...] = (
    "doubles_left",
    "doubles_right",
    "singles_left",
    "singles_right",
    "service_left",
    "service_right",
    "service_t",
)
PHYSICAL_INDICES_BY_CLASS: tuple[tuple[int, int], ...] = (
    (0, 2),
    (1, 3),
    (4, 5),
    (6, 7),
    (8, 10),
    (9, 11),
    (12, 13),
)


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
        count = self.in_frame_point_count
        if count == 14:
            return "full"
        if count >= 10:
            return "near_full"
        if count >= 4:
            return "partial"
        if count >= 1:
            return "sparse"
        return "none"

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


def attach_renderer_visibility(
    projection: MultiCourtProjection,
    *,
    alpha: NDArray[np.floating],
    depth: NDArray[np.floating],
    alpha_threshold: float = 0.01,
    sample_radius_px: int = 1,
) -> MultiCourtProjection:
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
    if np.any(alpha_array < 0.0) or np.any(alpha_array > 1.0) or np.any(depth_array < 0.0):
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


def attach_renderer_visibility_from_validated_arrays(
    projection: MultiCourtProjection,
    *,
    alpha: NDArray[np.float32],
    depth: NDArray[np.float32],
    alpha_threshold: float = 0.01,
    sample_radius_px: int = 1,
) -> MultiCourtProjection:
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
                        np.any(
                            (local_alpha >= alpha_threshold)
                            & (local_depth > 0.0)
                        )
                    )
                points.append(replace(point, renderer_visible=visible))
            classes.append(replace(semantic_class, points=(points[0], points[1])))
        courts.append(replace(court, classes=tuple(classes)))
    return replace(projection, courts=tuple(courts))


__all__ = [
    "CourtProjection",
    "MultiCourtProjection",
    "PHYSICAL_INDICES_BY_CLASS",
    "SEMANTIC_CLASS_NAMES",
    "SemanticClass",
    "SemanticPoint",
    "attach_renderer_visibility",
    "attach_renderer_visibility_from_validated_arrays",
    "project_court_semantics",
]
