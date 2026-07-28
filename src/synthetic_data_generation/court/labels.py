"""Instance-aware, near/far-symmetric court labels and seven-channel targets."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.court.layout import MultiCourtLayout
from src.synthetic_data_generation.scene_contract import SceneCamera

SYMMETRIC_KEYPOINT_CLASS_NAMES: tuple[str, ...] = (
    "doubles_left",
    "doubles_right",
    "singles_left",
    "singles_right",
    "service_left",
    "service_right",
    "service_t",
)
PHYSICAL_TO_SYMMETRIC_CLASS: tuple[int, ...] = (
    0,
    1,
    0,
    1,
    2,
    2,
    3,
    3,
    4,
    5,
    4,
    5,
    6,
    6,
)
PHYSICAL_INDICES_BY_SYMMETRIC_CLASS: tuple[tuple[int, int], ...] = (
    (0, 2),
    (1, 3),
    (4, 5),
    (6, 7),
    (8, 10),
    (9, 11),
    (12, 13),
)
_LINE_POINT_COUNT = 14


@dataclass(frozen=True)
class SymmetricCourtPoint:
    """One physical point whose near/far identity is intentionally absent."""

    uv: tuple[float, float]
    depth_scene: float
    xyz_scene: tuple[float, float, float]
    in_front: bool
    in_frame: bool
    visible: bool | None


@dataclass(frozen=True)
class SymmetricCourtClass:
    """Two unordered physical points sharing one semantic class."""

    class_id: int
    class_name: str
    points: tuple[SymmetricCourtPoint, ...]

    def __post_init__(self) -> None:
        if not 0 <= self.class_id < len(SYMMETRIC_KEYPOINT_CLASS_NAMES):
            raise ValueError("class_id is outside the seven-class schema.")
        if self.class_name != SYMMETRIC_KEYPOINT_CLASS_NAMES[self.class_id]:
            raise ValueError("class_name does not match class_id.")
        if len(self.points) != 2:
            raise ValueError("Every symmetric class must retain two physical points.")


@dataclass(frozen=True)
class CourtInstanceProjection:
    """Seven unordered point classes for one physical court instance."""

    court_instance_id: str
    classes: tuple[SymmetricCourtClass, ...]

    def __post_init__(self) -> None:
        if not self.court_instance_id:
            raise ValueError("court_instance_id must not be empty.")
        if tuple(value.class_id for value in self.classes) != tuple(range(7)):
            raise ValueError("Court classes must be ordered exactly 0..6.")

    @property
    def in_frame_point_count(self) -> int:
        """Return physical line points inside the image."""
        return sum(
            point.in_frame
            for value in self.classes
            for point in value.points
        )

    @property
    def in_frame_class_count(self) -> int:
        """Return semantic classes with at least one in-frame point."""
        return sum(
            any(point.in_frame for point in value.points)
            for value in self.classes
        )

    @property
    def coverage_bucket(self) -> str:
        """Return a deliberate full/partial supervision bucket."""
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


@dataclass(frozen=True)
class MultiCourtProjection:
    """Instance annotations plus a model target that deliberately merges them."""

    camera_id: str
    width: int
    height: int
    courts: tuple[CourtInstanceProjection, ...]

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id must not be empty.")
        if self.width <= 1 or self.height <= 1:
            raise ValueError("Projection dimensions must be greater than one.")
        ids = [court.court_instance_id for court in self.courts]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Projection court instances must be non-empty and unique.")


def project_multi_court(
    camera: SceneCamera,
    layout: MultiCourtLayout,
    *,
    near_plane_scene: float = 0.01,
) -> MultiCourtProjection:
    """Project all physical courts without assigning instance groups to a model."""
    if not np.isfinite(near_plane_scene) or near_plane_scene <= 0.0:
        raise ValueError("near_plane_scene must be finite and positive.")
    camera_to_scene = np.asarray(
        camera.camera_to_scene,
        dtype=np.float64,
    ).reshape(4, 4)
    scene_to_camera = np.linalg.inv(camera_to_scene)
    intrinsics = np.asarray(camera.intrinsics, dtype=np.float64).reshape(3, 3)
    court_records = []
    for court in layout.courts:
        points_scene = court.keypoints_scene()[:_LINE_POINT_COUNT]
        points_camera = (
            np.column_stack(
                (points_scene, np.ones(_LINE_POINT_COUNT, dtype=np.float64))
            )
            @ scene_to_camera.T
        )[:, :3]
        depth = points_camera[:, 2]
        homogeneous = points_camera @ intrinsics.T
        safe_depth = np.where(
            np.abs(depth) > np.finfo(np.float64).eps,
            depth,
            np.nan,
        )
        uv = homogeneous[:, :2] / safe_depth[:, None]
        in_front = depth > near_plane_scene
        in_frame = (
            in_front
            & np.isfinite(uv).all(axis=1)
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] < camera.width)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] < camera.height)
        )
        grouped: list[list[SymmetricCourtPoint]] = [[] for _ in range(7)]
        for physical_index, class_id in enumerate(PHYSICAL_TO_SYMMETRIC_CLASS):
            grouped[class_id].append(
                SymmetricCourtPoint(
                    uv=(float(uv[physical_index, 0]), float(uv[physical_index, 1])),
                    depth_scene=float(depth[physical_index]),
                    xyz_scene=(
                        float(points_scene[physical_index, 0]),
                        float(points_scene[physical_index, 1]),
                        float(points_scene[physical_index, 2]),
                    ),
                    in_front=bool(in_front[physical_index]),
                    in_frame=bool(in_frame[physical_index]),
                    visible=None,
                )
            )
        classes = tuple(
            SymmetricCourtClass(
                class_id=class_id,
                class_name=SYMMETRIC_KEYPOINT_CLASS_NAMES[class_id],
                points=tuple(grouped[class_id]),
            )
            for class_id in range(7)
        )
        court_records.append(
            CourtInstanceProjection(
                court_instance_id=court.court_instance_id,
                classes=classes,
            )
        )
    return MultiCourtProjection(
        camera_id=camera.camera_id,
        width=camera.width,
        height=camera.height,
        courts=tuple(court_records),
    )


def attach_visibility(
    projection: MultiCourtProjection,
    visibility_by_court: dict[str, tuple[bool, ...]],
) -> MultiCourtProjection:
    """Attach renderer-derived physical visibility without changing class IDs."""
    expected_ids = {court.court_instance_id for court in projection.courts}
    if set(visibility_by_court) != expected_ids:
        raise ValueError("visibility_by_court instance IDs differ from projection.")
    courts = []
    for court in projection.courts:
        values = visibility_by_court[court.court_instance_id]
        if len(values) != _LINE_POINT_COUNT:
            raise ValueError("Each court visibility vector must contain 14 values.")
        classes = []
        for class_record in court.classes:
            point_records = []
            physical_indices = PHYSICAL_INDICES_BY_SYMMETRIC_CLASS[
                class_record.class_id
            ]
            for point, physical_index in zip(
                class_record.points,
                physical_indices,
                strict=True,
            ):
                point_records.append(
                    replace(point, visible=bool(values[physical_index]))
                )
            classes.append(replace(class_record, points=tuple(point_records)))
        courts.append(replace(court, classes=tuple(classes)))
    return replace(projection, courts=tuple(courts))


def rescale_projection(
    projection: MultiCourtProjection,
    *,
    width: int,
    height: int,
) -> MultiCourtProjection:
    """Rescale pixel coordinates to an explicitly chosen render resolution."""
    if isinstance(width, bool) or isinstance(height, bool) or width <= 1 or height <= 1:
        raise ValueError("Projection dimensions must be integers greater than one.")
    scale_x = width / projection.width
    scale_y = height / projection.height
    courts = []
    for court in projection.courts:
        classes = []
        for class_record in court.classes:
            points = tuple(
                replace(
                    point,
                    uv=(point.uv[0] * scale_x, point.uv[1] * scale_y),
                )
                for point in class_record.points
            )
            classes.append(replace(class_record, points=points))
        courts.append(replace(court, classes=tuple(classes)))
    return replace(
        projection,
        width=width,
        height=height,
        courts=tuple(courts),
    )


def build_seven_channel_heatmaps(
    projection: MultiCourtProjection,
    *,
    sigma_px: float,
    require_renderer_visibility: bool,
) -> NDArray[np.float32]:
    """Build seven multi-peak heatmaps, merging all court instances by class.

    No court-instance grouping target is emitted. Multiple symmetric physical
    points and multiple courts contribute peaks to the same channel using
    pixelwise maximum composition.
    """
    if not np.isfinite(sigma_px) or sigma_px <= 0.0:
        raise ValueError("sigma_px must be finite and positive.")
    if require_renderer_visibility:
        unknown = [
            point
            for court in projection.courts
            for value in court.classes
            for point in value.points
            if point.in_frame and point.visible is None
        ]
        if unknown:
            raise ValueError(
                "Renderer visibility is required but remains unevaluated."
            )

    y, x = np.mgrid[0 : projection.height, 0 : projection.width]
    heatmaps: NDArray[np.float32] = np.zeros(
        (len(SYMMETRIC_KEYPOINT_CLASS_NAMES), projection.height, projection.width),
        dtype=np.float32,
    )
    denominator = 2.0 * sigma_px * sigma_px
    for court in projection.courts:
        for value in court.classes:
            for point in value.points:
                include = (
                    point.in_frame
                    and (
                        point.visible is True
                        if require_renderer_visibility
                        else True
                    )
                )
                if not include:
                    continue
                distance_squared = (
                    (x - point.uv[0]) ** 2 + (y - point.uv[1]) ** 2
                )
                peak = np.exp(-distance_squared / denominator).astype(np.float32)
                np.maximum(heatmaps[value.class_id], peak, out=heatmaps[value.class_id])
    return heatmaps


def encode_heatmap_atlas_u16(
    heatmaps: NDArray[np.floating],
) -> NDArray[np.uint16]:
    """Pack seven float heatmaps into one deterministic lossless PNG atlas."""
    values = np.asarray(heatmaps)
    if values.ndim != 3 or values.shape[0] != len(
        SYMMETRIC_KEYPOINT_CLASS_NAMES
    ):
        raise ValueError("heatmaps must have shape [7, height, width].")
    if values.shape[1] <= 1 or values.shape[2] <= 1:
        raise ValueError("Heatmap dimensions must be greater than one.")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Heatmaps must be finite and lie in [0, 1].")
    quantized = np.rint(values * np.iinfo(np.uint16).max).astype(np.uint16)
    return np.transpose(quantized, (1, 0, 2)).reshape(
        values.shape[1],
        values.shape[0] * values.shape[2],
    )


def decode_heatmap_atlas_u16(
    atlas: NDArray[np.integer],
    *,
    channel_count: int = len(SYMMETRIC_KEYPOINT_CLASS_NAMES),
) -> NDArray[np.float32]:
    """Decode a horizontal uint16 heatmap atlas back to ``[C,H,W]``."""
    values = np.asarray(atlas)
    if values.dtype != np.uint16 or values.ndim != 2:
        raise ValueError("Heatmap atlas must be a uint16 [height, channels*width] array.")
    if (
        isinstance(channel_count, bool)
        or channel_count <= 0
        or values.shape[1] % channel_count != 0
    ):
        raise ValueError("Heatmap atlas width must divide by channel_count.")
    width = values.shape[1] // channel_count
    decoded = values.reshape(values.shape[0], channel_count, width).transpose(
        1,
        0,
        2,
    )
    return np.asarray(
        decoded,
        dtype=np.float32,
    ) / np.float32(np.iinfo(np.uint16).max)
