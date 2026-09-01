"""Deterministic depth-aware rendering of exact Court V4 obstacle voxels.

The renderer treats each occupied integer cell as the closed scene-space AABB
``[cell * voxel_size, (cell + 1) * voxel_size]``. Only faces exposed by the
exact six-neighbour occupancy relation contribute geometry. Their canonical,
deduplicated boundary segments are rasterized by default; explicitly selected
solid rendering retains the exposed-face rasterizer. Input RGB is NHT's raw
premultiplied/coverage-weighted accumulation; before the obstacle overlay it is
resolved as ``rgb + background * (1 - alpha)``. A 3DGS depth is valid exactly
where ``alpha > 0`` and ``metric_depth > 0``. Invalid depth does not occlude an
obstacle surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.scene_contract import SceneCamera
from src.synthetic_data_generation.visualization.contracts import (
    CourtAABBRenderStyle,
    CourtAABBTrajectoryFilterRadiusMode,
    CourtAABBTrajectoryFilterScope,
    CourtAABBWireframeTopology,
)

_FACE_DEFINITIONS: tuple[
    tuple[tuple[int, int, int], tuple[tuple[int, int, int], ...]], ...
] = (
    ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
    ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
    ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
    ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
    ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
    ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
)
_BARYCENTRIC_EPSILON = 1.0e-10
_DEGENERATE_AREA_EPSILON = 1.0e-12
COURT_AABB_TRAJECTORY_DISTANCE_METRIC = (
    "trajectory_segment_to_closed_cell_aabb"
)


@dataclass(frozen=True, slots=True)
class CourtAABBRenderConfig:
    """Validated geometry, composition, and fail-closed resource limits."""

    voxel_size_m: float
    render_style: CourtAABBRenderStyle
    wireframe_topology: CourtAABBWireframeTopology
    near_plane_m: float
    depth_epsilon_m: float
    surface_color_rgb: tuple[float, float, float]
    surface_opacity: float
    edge_opacity: float
    edge_width_px: int
    background_color_rgb: tuple[float, float, float]
    maximum_cells: int
    maximum_surface_faces: int
    maximum_edge_segments: int
    maximum_projected_pixels: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "voxel_size_m",
            _positive_float(self.voxel_size_m, name="voxel_size_m"),
        )
        if not isinstance(self.render_style, CourtAABBRenderStyle):
            raise TypeError("render_style must be CourtAABBRenderStyle.")
        if not isinstance(self.wireframe_topology, CourtAABBWireframeTopology):
            raise TypeError(
                "wireframe_topology must be CourtAABBWireframeTopology."
            )
        object.__setattr__(
            self,
            "near_plane_m",
            _positive_float(self.near_plane_m, name="near_plane_m"),
        )
        object.__setattr__(
            self,
            "depth_epsilon_m",
            _nonnegative_float(self.depth_epsilon_m, name="depth_epsilon_m"),
        )
        object.__setattr__(
            self,
            "surface_color_rgb",
            _rgb_tuple(self.surface_color_rgb, name="surface_color_rgb"),
        )
        object.__setattr__(
            self,
            "surface_opacity",
            _unit_float(self.surface_opacity, name="surface_opacity"),
        )
        edge_opacity = _unit_float(self.edge_opacity, name="edge_opacity")
        if edge_opacity <= 0.0:
            raise ValueError("edge_opacity must be positive.")
        object.__setattr__(self, "edge_opacity", edge_opacity)
        edge_width = _positive_int(self.edge_width_px, name="edge_width_px")
        if edge_width > 64:
            raise ValueError("edge_width_px must not exceed 64.")
        object.__setattr__(self, "edge_width_px", edge_width)
        object.__setattr__(
            self,
            "background_color_rgb",
            _rgb_tuple(self.background_color_rgb, name="background_color_rgb"),
        )
        for name in (
            "maximum_cells",
            "maximum_surface_faces",
            "maximum_edge_segments",
            "maximum_projected_pixels",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name=name))


@dataclass(frozen=True, slots=True)
class CourtAABBRenderStats:
    """Immutable counters describing one complete, non-truncated render."""

    cell_count: int
    surface_face_count: int
    source_triangle_count: int
    candidate_edge_segment_count: int
    edge_segment_count: int
    suppressed_seam_segment_count: int
    near_clipped_face_count: int
    near_rejected_face_count: int
    triangle_count: int
    raster_triangle_count: int
    near_clipped_edge_segment_count: int
    near_rejected_edge_segment_count: int
    raster_edge_segment_count: int
    projected_pixel_count: int
    covered_fragment_count: int
    surface_pixel_count: int
    drawn_pixel_count: int
    occluded_pixel_count: int
    edge_pixel_count: int
    drawn_edge_pixel_count: int
    occluded_edge_pixel_count: int
    background_valid_pixel_count: int
    background_invalid_pixel_count: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _nonnegative_int(getattr(self, name), name=name)
        if self.source_triangle_count != 2 * self.surface_face_count:
            raise ValueError("source_triangle_count must equal twice surface_face_count.")
        if (
            self.candidate_edge_segment_count
            != self.edge_segment_count + self.suppressed_seam_segment_count
        ):
            raise ValueError(
                "candidate_edge_segment_count must equal drawn plus suppressed edges."
            )
        if self.drawn_pixel_count + self.occluded_pixel_count != self.surface_pixel_count:
            raise ValueError("Every surface pixel must be either drawn or occluded.")
        if (
            self.drawn_edge_pixel_count + self.occluded_edge_pixel_count
            != self.edge_pixel_count
        ):
            raise ValueError("Every edge pixel must be either drawn or occluded.")


@dataclass(frozen=True, slots=True)
class CourtAABBRenderResult:
    """A read-only uint8 RGB visualization and its immutable frame statistics."""

    camera_id: str
    rgb: NDArray[np.uint8]
    stats: CourtAABBRenderStats

    def __post_init__(self) -> None:
        if not isinstance(self.camera_id, str) or not self.camera_id:
            raise TypeError("camera_id must be a non-empty string.")
        if not isinstance(self.rgb, np.ndarray) or self.rgb.dtype != np.dtype(np.uint8):
            raise TypeError("rgb must be a uint8 numpy array.")
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError("rgb must have shape [H,W,3].")
        if not isinstance(self.stats, CourtAABBRenderStats):
            raise TypeError("stats must be CourtAABBRenderStats.")
        output = np.ascontiguousarray(self.rgb, dtype=np.uint8)
        output.setflags(write=False)
        object.__setattr__(self, "rgb", output)


@dataclass(frozen=True, slots=True, eq=False)
class PreparedCourtAABBGeometry:
    """One exact exposed surface prepared once for a streaming output."""

    cell_count: int
    voxel_size_m: float
    faces_scene_m: NDArray[np.float64]
    edge_segments_scene_m: NDArray[np.float64]
    candidate_edge_segment_count: int
    suppressed_seam_segment_count: int

    def __post_init__(self) -> None:
        cell_count = _nonnegative_int(self.cell_count, name="cell_count")
        voxel_size = _positive_float(self.voxel_size_m, name="voxel_size_m")
        faces = self.faces_scene_m
        if not isinstance(faces, np.ndarray) or faces.dtype != np.dtype(np.float64):
            raise TypeError("faces_scene_m must be a float64 numpy array.")
        if faces.ndim != 3 or faces.shape[1:] != (4, 3):
            raise ValueError("faces_scene_m must have exact shape [F,4,3].")
        if not np.isfinite(faces).all():
            raise ValueError("faces_scene_m must contain only finite metric values.")
        if faces.shape[0] > 6 * cell_count:
            raise ValueError("faces_scene_m exceeds six faces per occupied cell.")
        edges = self.edge_segments_scene_m
        if not isinstance(edges, np.ndarray) or edges.dtype != np.dtype(np.float64):
            raise TypeError("edge_segments_scene_m must be a float64 numpy array.")
        if edges.ndim != 3 or edges.shape[1:] != (2, 3):
            raise ValueError("edge_segments_scene_m must have exact shape [E,2,3].")
        if not np.isfinite(edges).all():
            raise ValueError(
                "edge_segments_scene_m must contain only finite metric values."
            )
        if edges.shape[0] > 4 * faces.shape[0]:
            raise ValueError("edge_segments_scene_m exceeds four edges per face.")
        candidate_edge_count = _nonnegative_int(
            self.candidate_edge_segment_count,
            name="candidate_edge_segment_count",
        )
        suppressed_seam_count = _nonnegative_int(
            self.suppressed_seam_segment_count,
            name="suppressed_seam_segment_count",
        )
        if candidate_edge_count > 4 * faces.shape[0]:
            raise ValueError(
                "candidate_edge_segment_count exceeds four edges per face."
            )
        if candidate_edge_count != edges.shape[0] + suppressed_seam_count:
            raise ValueError(
                "candidate_edge_segment_count must equal drawn plus suppressed edges."
            )
        prepared = np.ascontiguousarray(faces, dtype=np.float64)
        prepared.setflags(write=False)
        prepared_edges = np.ascontiguousarray(edges, dtype=np.float64)
        prepared_edges.setflags(write=False)
        object.__setattr__(self, "cell_count", cell_count)
        object.__setattr__(self, "voxel_size_m", voxel_size)
        object.__setattr__(self, "faces_scene_m", prepared)
        object.__setattr__(self, "edge_segments_scene_m", prepared_edges)
        object.__setattr__(
            self,
            "candidate_edge_segment_count",
            candidate_edge_count,
        )
        object.__setattr__(
            self,
            "suppressed_seam_segment_count",
            suppressed_seam_count,
        )

    @property
    def surface_face_count(self) -> int:
        """Return the exact exposed-quad count."""
        return int(self.faces_scene_m.shape[0])

    @property
    def edge_segment_count(self) -> int:
        """Return the exact canonical exposed-face edge count."""
        return int(self.edge_segments_scene_m.shape[0])


@dataclass(frozen=True, slots=True, eq=False)
class CourtAABBTrajectoryFilterResult:
    """Immutable display-only occupancy derived from one selected trajectory."""

    cells: NDArray[np.int64]
    scope: CourtAABBTrajectoryFilterScope
    radius_mode: CourtAABBTrajectoryFilterRadiusMode | None
    resolved_radius_m: float | None
    original_cell_count: int
    trajectory_center_count: int
    trajectory_segment_count: int
    filter_segment_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.scope, CourtAABBTrajectoryFilterScope):
            raise TypeError("scope must be CourtAABBTrajectoryFilterScope.")
        if self.radius_mode is not None and not isinstance(
            self.radius_mode,
            CourtAABBTrajectoryFilterRadiusMode,
        ):
            raise TypeError(
                "radius_mode must be CourtAABBTrajectoryFilterRadiusMode or None."
            )
        original_count = _positive_int(
            self.original_cell_count,
            name="original_cell_count",
        )
        center_count = _positive_int(
            self.trajectory_center_count,
            name="trajectory_center_count",
        )
        segment_count = _positive_int(
            self.trajectory_segment_count,
            name="trajectory_segment_count",
        )
        if segment_count != center_count:
            raise ValueError(
                "A closed trajectory must have one segment per ordered center."
            )
        filter_segment_count = _nonnegative_int(
            self.filter_segment_count,
            name="filter_segment_count",
        )
        if self.scope is CourtAABBTrajectoryFilterScope.ALL:
            if self.radius_mode is not None or self.resolved_radius_m is not None:
                raise ValueError(
                    "all trajectory filtering requires radius mode and radius None."
                )
            if filter_segment_count != 0:
                raise ValueError("all trajectory filtering uses no filter segments.")
            radius = None
        else:
            if self.radius_mode is None:
                raise ValueError("non-all trajectory filtering requires radius mode.")
            radius = _positive_float(
                self.resolved_radius_m,
                name="resolved_radius_m",
            )
            if self.scope is CourtAABBTrajectoryFilterScope.SELECTED_TRAJECTORY:
                if filter_segment_count != segment_count:
                    raise ValueError(
                        "selected_trajectory filtering must use every closed segment."
                    )
            elif filter_segment_count != min(2, segment_count):
                raise ValueError(
                    "local_swept_segments filtering must use its unique incoming and "
                    "outgoing segments."
                )
        cells = _validated_cells(self.cells, maximum_cells=original_count)
        retained_count = int(cells.shape[0])
        if retained_count == 0:
            raise ValueError("Trajectory occupancy filter retained no cells.")
        if retained_count > original_count:
            raise ValueError("Filtered occupancy exceeds its original cell count.")
        prepared = np.array(cells, dtype=np.int64, copy=True, order="C")
        prepared.setflags(write=False)
        object.__setattr__(self, "cells", prepared)
        object.__setattr__(self, "resolved_radius_m", radius)
        object.__setattr__(self, "original_cell_count", original_count)
        object.__setattr__(self, "trajectory_center_count", center_count)
        object.__setattr__(self, "trajectory_segment_count", segment_count)
        object.__setattr__(self, "filter_segment_count", filter_segment_count)

    @property
    def retained_cell_count(self) -> int:
        """Return the complete derived display-cell count."""
        return int(self.cells.shape[0])

    @property
    def removed_cell_count(self) -> int:
        """Return the number of artifact cells excluded only from display."""
        return self.original_cell_count - self.retained_cell_count

    def to_dict(self) -> dict[str, object]:
        """Return deterministic sidecar evidence for the derived display scope."""
        return {
            "scope": self.scope.value,
            "radius_mode": (
                None if self.radius_mode is None else self.radius_mode.value
            ),
            "resolved_radius_m": self.resolved_radius_m,
            "distance_metric": COURT_AABB_TRAJECTORY_DISTANCE_METRIC,
            "original_cell_count": self.original_cell_count,
            "retained_cell_count": self.retained_cell_count,
            "removed_cell_count": self.removed_cell_count,
            "trajectory_center_count": self.trajectory_center_count,
            "trajectory_segment_count": self.trajectory_segment_count,
            "filter_segment_count": self.filter_segment_count,
            "closed_trajectory": True,
            "affects_collision_authority": False,
        }


@dataclass(frozen=True, slots=True, eq=False)
class PreparedCourtAABBTrajectoryFilter:
    """Validated occupancy and closed trajectory reused by frame-local filtering."""

    cells: NDArray[np.int64]
    trajectory_centers_scene_m: NDArray[np.float64]
    cell_lower_scene_m: NDArray[np.float64]
    cell_upper_scene_m: NDArray[np.float64]
    voxel_size_m: float
    scope: CourtAABBTrajectoryFilterScope
    radius_mode: CourtAABBTrajectoryFilterRadiusMode | None
    resolved_radius_m: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.cells, np.ndarray) or self.cells.shape[0] == 0:
            raise ValueError("Prepared trajectory filter cells must not be empty.")
        cells = _validated_cells(
            self.cells,
            maximum_cells=int(self.cells.shape[0]),
        )
        centers = _validated_trajectory_centers(self.trajectory_centers_scene_m)
        voxel_size = _positive_float(self.voxel_size_m, name="voxel_size_m")
        if not isinstance(self.scope, CourtAABBTrajectoryFilterScope):
            raise TypeError("scope must be CourtAABBTrajectoryFilterScope.")
        if self.radius_mode is not None and not isinstance(
            self.radius_mode,
            CourtAABBTrajectoryFilterRadiusMode,
        ):
            raise TypeError(
                "radius_mode must be CourtAABBTrajectoryFilterRadiusMode or None."
            )
        if self.scope is CourtAABBTrajectoryFilterScope.ALL:
            if self.radius_mode is not None or self.resolved_radius_m is not None:
                raise ValueError(
                    "all trajectory filtering requires radius mode and radius None."
                )
            radius = None
        else:
            if self.radius_mode is None:
                raise ValueError("non-all trajectory filtering requires radius mode.")
            radius = _positive_float(
                self.resolved_radius_m,
                name="resolved_radius_m",
            )
        expected_lower = cells.astype(np.float64) * voxel_size
        expected_upper = expected_lower + voxel_size
        for name, value, expected in (
            ("cell_lower_scene_m", self.cell_lower_scene_m, expected_lower),
            ("cell_upper_scene_m", self.cell_upper_scene_m, expected_upper),
        ):
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.dtype(np.float64)
                or value.shape != cells.shape
                or not np.array_equal(value, expected)
            ):
                raise ValueError(f"{name} must exactly match the prepared cell AABBs.")
        prepared_cells = np.array(cells, dtype=np.int64, copy=True, order="C")
        prepared_centers = np.array(centers, dtype=np.float64, copy=True, order="C")
        prepared_lower = np.array(expected_lower, dtype=np.float64, copy=True, order="C")
        prepared_upper = np.array(expected_upper, dtype=np.float64, copy=True, order="C")
        for value in (
            prepared_cells,
            prepared_centers,
            prepared_lower,
            prepared_upper,
        ):
            value.setflags(write=False)
        object.__setattr__(self, "cells", prepared_cells)
        object.__setattr__(self, "trajectory_centers_scene_m", prepared_centers)
        object.__setattr__(self, "cell_lower_scene_m", prepared_lower)
        object.__setattr__(self, "cell_upper_scene_m", prepared_upper)
        object.__setattr__(self, "voxel_size_m", voxel_size)
        object.__setattr__(self, "resolved_radius_m", radius)

    @property
    def original_cell_count(self) -> int:
        """Return the exact artifact cell count."""
        return int(self.cells.shape[0])

    @property
    def trajectory_center_count(self) -> int:
        """Return the canonical closed-trajectory center count."""
        return int(self.trajectory_centers_scene_m.shape[0])

    @property
    def trajectory_segment_count(self) -> int:
        """Return the authority segment count for the closed trajectory."""
        return self.trajectory_center_count

    def filter(self, *, frame_index: int | None = None) -> CourtAABBTrajectoryFilterResult:
        """Return the complete display subset for one configured scope."""
        if self.scope is CourtAABBTrajectoryFilterScope.ALL:
            if frame_index is not None:
                raise ValueError("all filtering does not accept frame_index.")
            retained = self.cells
            segment_indices: tuple[int, ...] = ()
        else:
            if self.scope is CourtAABBTrajectoryFilterScope.LOCAL_SWEPT_SEGMENTS:
                if (
                    isinstance(frame_index, bool)
                    or not isinstance(frame_index, int)
                    or not 0 <= frame_index < self.trajectory_center_count
                ):
                    raise ValueError(
                        "local_swept_segments requires a canonical frame_index."
                    )
                incoming = (frame_index - 1) % self.trajectory_segment_count
                outgoing = frame_index
                segment_indices = tuple(dict.fromkeys((incoming, outgoing)))
            else:
                if frame_index is not None:
                    raise ValueError(
                        "selected_trajectory filtering does not accept frame_index."
                    )
                segment_indices = tuple(range(self.trajectory_segment_count))
            assert self.resolved_radius_m is not None
            retained = _filter_cells_by_segments(
                self.cells,
                cell_lower=self.cell_lower_scene_m,
                cell_upper=self.cell_upper_scene_m,
                trajectory_centers=self.trajectory_centers_scene_m,
                segment_indices=segment_indices,
                radius_m=self.resolved_radius_m,
            )
            if retained.shape[0] == 0:
                raise ValueError("Trajectory occupancy filter retained no cells.")
        return CourtAABBTrajectoryFilterResult(
            cells=retained,
            scope=self.scope,
            radius_mode=self.radius_mode,
            resolved_radius_m=self.resolved_radius_m,
            original_cell_count=self.original_cell_count,
            trajectory_center_count=self.trajectory_center_count,
            trajectory_segment_count=self.trajectory_segment_count,
            filter_segment_count=len(segment_indices),
        )


@dataclass(frozen=True, slots=True)
class _ProjectedSegment:
    camera_depths: NDArray[np.float64]
    pixels: NDArray[np.float64]
    sample_count: int


@dataclass(frozen=True, slots=True)
class _ProjectedTriangle:
    vertices_camera: NDArray[np.float64]
    pixels: NDArray[np.float64]
    bounds: tuple[int, int, int, int]


def filter_court_obstacle_cells_by_trajectory(
    occupancy_cells: object,
    *,
    trajectory_centers_scene_m: object,
    voxel_size_m: float,
    scope: CourtAABBTrajectoryFilterScope,
    radius_mode: CourtAABBTrajectoryFilterRadiusMode | None,
    resolved_radius_m: float | None,
    maximum_cells: int,
    frame_index: int | None = None,
) -> CourtAABBTrajectoryFilterResult:
    """Prepare and derive one exact trajectory-context display subset."""
    prepared = prepare_court_aabb_trajectory_filter(
        occupancy_cells,
        trajectory_centers_scene_m=trajectory_centers_scene_m,
        voxel_size_m=voxel_size_m,
        scope=scope,
        radius_mode=radius_mode,
        resolved_radius_m=resolved_radius_m,
        maximum_cells=maximum_cells,
    )
    return prepared.filter(frame_index=frame_index)


def prepare_court_aabb_trajectory_filter(
    occupancy_cells: object,
    *,
    trajectory_centers_scene_m: object,
    voxel_size_m: float,
    scope: CourtAABBTrajectoryFilterScope,
    radius_mode: CourtAABBTrajectoryFilterRadiusMode | None,
    resolved_radius_m: float | None,
    maximum_cells: int,
) -> PreparedCourtAABBTrajectoryFilter:
    """Validate the artifact and closed trajectory once for streaming filters.

    The ordered centers form a closed polyline. Each occupied cell is treated as
    the closed AABB ``[cell * voxel_size, (cell + 1) * voxel_size]``. Radius
    inclusion remains exact and inclusive in :meth:`filter`; preparation only
    validates and derives immutable metric bounds. The display subset never
    changes collision authority or its artifact.
    """
    if not isinstance(scope, CourtAABBTrajectoryFilterScope):
        raise TypeError("scope must be CourtAABBTrajectoryFilterScope.")
    if radius_mode is not None and not isinstance(
        radius_mode,
        CourtAABBTrajectoryFilterRadiusMode,
    ):
        raise TypeError(
            "radius_mode must be CourtAABBTrajectoryFilterRadiusMode or None."
        )
    voxel_size = _positive_float(voxel_size_m, name="voxel_size_m")
    cell_limit = _positive_int(maximum_cells, name="maximum_cells")
    cells = _validated_cells(occupancy_cells, maximum_cells=cell_limit)
    if cells.shape[0] == 0:
        raise ValueError("Trajectory occupancy filter requires non-empty cells.")
    centers = _validated_trajectory_centers(trajectory_centers_scene_m)
    if scope is CourtAABBTrajectoryFilterScope.ALL:
        if radius_mode is not None or resolved_radius_m is not None:
            raise ValueError(
                "all trajectory filtering requires radius mode and radius None."
            )
        radius = None
    else:
        if radius_mode is None:
            raise ValueError("non-all trajectory filtering requires radius mode.")
        radius = _positive_float(resolved_radius_m, name="resolved_radius_m")
    cell_lower = cells.astype(np.float64) * voxel_size
    cell_upper = cell_lower + voxel_size
    return PreparedCourtAABBTrajectoryFilter(
        cells=cells,
        trajectory_centers_scene_m=centers,
        cell_lower_scene_m=cell_lower,
        cell_upper_scene_m=cell_upper,
        voxel_size_m=voxel_size,
        scope=scope,
        radius_mode=radius_mode,
        resolved_radius_m=radius,
    )


def _filter_cells_by_segments(
    cells: NDArray[np.int64],
    *,
    cell_lower: NDArray[np.float64],
    cell_upper: NDArray[np.float64],
    trajectory_centers: NDArray[np.float64],
    segment_indices: tuple[int, ...],
    radius_m: float,
) -> NDArray[np.int64]:
    """Return cells within an inclusive exact distance of selected segments."""
    radius_squared = radius_m * radius_m
    retained_mask = np.zeros((cells.shape[0],), dtype=np.bool_)
    segment_ends = np.roll(trajectory_centers, shift=-1, axis=0)
    for segment_index in segment_indices:
        start = trajectory_centers[segment_index]
        end = segment_ends[segment_index]
        expanded_lower = np.nextafter(
            np.minimum(start, end) - radius_m,
            -math.inf,
        )
        expanded_upper = np.nextafter(
            np.maximum(start, end) + radius_m,
            math.inf,
        )
        candidates = np.flatnonzero(
            ~retained_mask
            & np.all(cell_upper >= expanded_lower, axis=1)
            & np.all(cell_lower <= expanded_upper, axis=1)
        )
        for index in candidates:
            if (
                segment_aabb_distance_squared(
                    start,
                    end,
                    lower=cell_lower[index],
                    upper=cell_upper[index],
                )
                <= radius_squared
            ):
                retained_mask[index] = True
    return cells[retained_mask]


def segment_aabb_distance_squared(
    start: NDArray[np.float64],
    end: NDArray[np.float64],
    *,
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> float:
    """Return exact squared distance from a closed segment to a closed AABB."""
    direction = end - start
    breakpoints = {0.0, 1.0}
    for axis in range(3):
        if abs(float(direction[axis])) <= 1.0e-15:
            continue
        for boundary in (lower[axis], upper[axis]):
            value = float((boundary - start[axis]) / direction[axis])
            if 0.0 < value < 1.0:
                breakpoints.add(value)
    ordered = sorted(breakpoints)

    def distance_squared(parameter: float) -> float:
        point = start + direction * parameter
        delta = np.maximum(np.maximum(lower - point, point - upper), 0.0)
        return float(delta @ delta)

    best = min(distance_squared(value) for value in ordered)
    for first, second in zip(ordered, ordered[1:], strict=False):
        midpoint = (first + second) / 2.0
        point = start + direction * midpoint
        coefficients: list[tuple[float, float]] = []
        for axis in range(3):
            if point[axis] < lower[axis]:
                coefficients.append(
                    (-float(direction[axis]), float(lower[axis] - start[axis]))
                )
            elif point[axis] > upper[axis]:
                coefficients.append(
                    (float(direction[axis]), float(start[axis] - upper[axis]))
                )
        denominator = sum(slope * slope for slope, _intercept in coefficients)
        if denominator <= 0.0:
            continue
        optimum = (
            -sum(slope * intercept for slope, intercept in coefficients) / denominator
        )
        if first <= optimum <= second:
            best = min(best, distance_squared(optimum))
    return best


def extract_exposed_voxel_faces(
    occupancy_cells: object,
    *,
    voxel_size_m: float,
    maximum_cells: int,
    maximum_surface_faces: int,
) -> NDArray[np.float64]:
    """Return deterministic outward-wound exposed quads with shape ``[F,4,3]``.

    Cells must already be unique and strictly lexicographically sorted.  The
    face order is cell order followed by ``-x,+x,-y,+y,-z,+z``.
    """
    voxel_size = _positive_float(voxel_size_m, name="voxel_size_m")
    cell_limit = _positive_int(maximum_cells, name="maximum_cells")
    face_limit = _positive_int(maximum_surface_faces, name="maximum_surface_faces")
    cells = _validated_cells(occupancy_cells, maximum_cells=cell_limit)
    occupied = {tuple(int(item) for item in row) for row in cells}
    faces: list[NDArray[np.float64]] = []
    for row in cells:
        cell = tuple(int(item) for item in row)
        for neighbour_offset, corners in _FACE_DEFINITIONS:
            neighbour = tuple(
                cell[axis] + neighbour_offset[axis] for axis in range(3)
            )
            if neighbour in occupied:
                continue
            if len(faces) >= face_limit:
                raise ValueError(
                    "exposed voxel surface exceeds maximum_surface_faces="
                    f"{face_limit}; rendering was not truncated."
                )
            face = np.asarray(
                [
                    [
                        (cell[axis] + corner[axis]) * voxel_size
                        for axis in range(3)
                    ]
                    for corner in corners
                ],
                dtype=np.float64,
            )
            if not np.isfinite(face).all():
                raise ValueError("occupancy cell bounds must be finite in scene metres.")
            faces.append(face)
    result = (
        np.stack(faces, axis=0)
        if faces
        else np.empty((0, 4, 3), dtype=np.float64)
    )
    result.setflags(write=False)
    return result


def extract_canonical_exposed_face_edges(
    faces_scene_m: object,
    *,
    wireframe_topology: CourtAABBWireframeTopology,
    maximum_edge_segments: int,
) -> NDArray[np.float64]:
    """Return sorted filtered exposed-face edges with shape ``[E,2,3]``."""
    edges, _, _ = _extract_canonical_exposed_face_edge_geometry(
        faces_scene_m,
        wireframe_topology=wireframe_topology,
        maximum_edge_segments=maximum_edge_segments,
    )
    return edges


def _extract_canonical_exposed_face_edge_geometry(
    faces_scene_m: object,
    *,
    wireframe_topology: CourtAABBWireframeTopology,
    maximum_edge_segments: int,
) -> tuple[NDArray[np.float64], int, int]:
    """Return filtered edges, candidate count, and suppressed seam count."""
    limit = _positive_int(maximum_edge_segments, name="maximum_edge_segments")
    if not isinstance(wireframe_topology, CourtAABBWireframeTopology):
        raise TypeError(
            "wireframe_topology must be CourtAABBWireframeTopology."
        )
    if not isinstance(faces_scene_m, np.ndarray) or faces_scene_m.dtype != np.dtype(
        np.float64
    ):
        raise TypeError("faces_scene_m must be a float64 numpy array.")
    if faces_scene_m.ndim != 3 or faces_scene_m.shape[1:] != (4, 3):
        raise ValueError("faces_scene_m must have exact shape [F,4,3].")
    if not np.isfinite(faces_scene_m).all():
        raise ValueError("faces_scene_m must contain only finite values.")
    edge_normals: dict[
        tuple[tuple[float, float, float], tuple[float, float, float]],
        list[tuple[float, float, float]],
    ] = {}
    for face in faces_scene_m:
        cross = np.cross(face[1] - face[0], face[2] - face[0])
        normal_length = float(np.linalg.norm(cross))
        if not math.isfinite(normal_length) or normal_length <= 0.0:
            raise ValueError("Exposed face must have a finite nondegenerate normal.")
        normal = cast(
            tuple[float, float, float],
            tuple(float(value / normal_length) for value in cross),
        )
        for start_index, end_index in ((0, 1), (1, 2), (2, 3), (3, 0)):
            start = cast(tuple[float, float, float], tuple(float(v) for v in face[start_index]))
            end = cast(tuple[float, float, float], tuple(float(v) for v in face[end_index]))
            if start == end:
                raise ValueError("Exposed face edges must have distinct endpoints.")
            edge = (start, end) if start < end else (end, start)
            edge_normals.setdefault(edge, []).append(normal)
    ordered_candidates = sorted(edge_normals)
    if wireframe_topology is CourtAABBWireframeTopology.BOUNDARY:
        ordered = [
            edge
            for edge in ordered_candidates
            if not (
                len(edge_normals[edge]) == 2
                and len(set(edge_normals[edge])) == 1
            )
        ]
    else:
        ordered = ordered_candidates
    if len(ordered) > limit:
        raise ValueError(
            "filtered exposed-face edges exceed maximum_edge_segments="
            f"{limit}; rendering was not truncated."
        )
    candidate_count = len(ordered_candidates)
    suppressed_count = candidate_count - len(ordered)
    result = (
        np.asarray(ordered, dtype=np.float64)
        if ordered
        else np.empty((0, 2, 3), dtype=np.float64)
    )
    result.setflags(write=False)
    return result, candidate_count, suppressed_count


def prepare_court_obstacle_aabbs(
    occupancy_cells: object,
    *,
    config: CourtAABBRenderConfig,
) -> PreparedCourtAABBGeometry:
    """Validate cells and extract their exact exposed surfaces once."""
    if not isinstance(config, CourtAABBRenderConfig):
        raise TypeError("config must be CourtAABBRenderConfig.")
    cells = _validated_cells(occupancy_cells, maximum_cells=config.maximum_cells)
    faces = extract_exposed_voxel_faces(
        cells,
        voxel_size_m=config.voxel_size_m,
        maximum_cells=config.maximum_cells,
        maximum_surface_faces=config.maximum_surface_faces,
    )
    if config.render_style is CourtAABBRenderStyle.WIREFRAME:
        edges, candidate_edge_count, suppressed_seam_count = (
            _extract_canonical_exposed_face_edge_geometry(
                faces,
                wireframe_topology=config.wireframe_topology,
                maximum_edge_segments=config.maximum_edge_segments,
            )
        )
    else:
        edges = np.empty((0, 2, 3), dtype=np.float64)
        candidate_edge_count = 0
        suppressed_seam_count = 0
    edges.setflags(write=False)
    return PreparedCourtAABBGeometry(
        cell_count=int(cells.shape[0]),
        voxel_size_m=config.voxel_size_m,
        faces_scene_m=faces,
        edge_segments_scene_m=edges,
        candidate_edge_segment_count=candidate_edge_count,
        suppressed_seam_segment_count=suppressed_seam_count,
    )


def render_court_obstacle_aabbs(
    *,
    rgb: object,
    alpha: object,
    metric_depth: object,
    camera: SceneCamera,
    occupancy_cells: object,
    config: CourtAABBRenderConfig,
) -> CourtAABBRenderResult:
    """Depth-composite the configured exact occupancy style over one 3DGS frame.

    Fragment depths are perspective-correct camera-Z values. A fragment is
    drawn when background depth is invalid or when
    ``fragment_z <= metric_depth + depth_epsilon_m``.
    """
    if not isinstance(config, CourtAABBRenderConfig):
        raise TypeError("config must be CourtAABBRenderConfig.")
    geometry = prepare_court_obstacle_aabbs(occupancy_cells, config=config)
    return render_prepared_court_obstacle_aabbs(
        rgb=rgb,
        alpha=alpha,
        metric_depth=metric_depth,
        camera=camera,
        geometry=geometry,
        config=config,
    )


def render_prepared_court_obstacle_aabbs(
    *,
    rgb: object,
    alpha: object,
    metric_depth: object,
    camera: SceneCamera,
    geometry: PreparedCourtAABBGeometry,
    config: CourtAABBRenderConfig,
) -> CourtAABBRenderResult:
    """Depth-composite one prevalidated exact surface without rebuilding it."""
    if not isinstance(camera, SceneCamera):
        raise TypeError("camera must be a SceneCamera.")
    if not isinstance(geometry, PreparedCourtAABBGeometry):
        raise TypeError("geometry must be PreparedCourtAABBGeometry.")
    if not isinstance(config, CourtAABBRenderConfig):
        raise TypeError("config must be CourtAABBRenderConfig.")
    if geometry.voxel_size_m != config.voxel_size_m:
        raise ValueError("Prepared AABB geometry voxel size disagrees with config.")
    if geometry.cell_count > config.maximum_cells:
        raise ValueError("Prepared AABB geometry exceeds maximum_cells.")
    if geometry.surface_face_count > config.maximum_surface_faces:
        raise ValueError("Prepared AABB geometry exceeds maximum_surface_faces.")
    if geometry.edge_segment_count > config.maximum_edge_segments:
        raise ValueError("Prepared AABB geometry exceeds maximum_edge_segments.")
    rgb_array, alpha_array, depth_array = _validated_frame_arrays(
        rgb=rgb,
        alpha=alpha,
        metric_depth=metric_depth,
        camera=camera,
    )
    faces_scene = geometry.faces_scene_m
    edges_scene = geometry.edge_segments_scene_m

    camera_from_scene = camera.camera_to_scene.inverse().matrix()
    intrinsic = np.asarray(camera.intrinsics, dtype=np.float64).reshape(3, 3)
    near_clipped_faces = 0
    near_rejected_faces = 0
    triangle_count = 0
    raster_triangle_count = 0
    near_clipped_edges = 0
    near_rejected_edges = 0
    raster_edge_count = 0
    projected_pixel_count = 0
    covered_fragment_count = 0
    z_buffer = np.full((camera.height, camera.width), np.inf, dtype=np.float64)

    if config.render_style is CourtAABBRenderStyle.SOLID:
        faces_camera = (
            faces_scene @ camera_from_scene[:3, :3].T + camera_from_scene[:3, 3]
        )
        projected_triangles: list[_ProjectedTriangle] = []
        for face_camera in faces_camera:
            was_clipped = bool(np.any(face_camera[:, 2] < config.near_plane_m))
            polygon = _clip_polygon_to_near_plane(face_camera, config.near_plane_m)
            if polygon.shape[0] < 3:
                near_rejected_faces += 1
                continue
            if was_clipped:
                near_clipped_faces += 1
            triangles = _triangulate_fan(polygon)
            triangle_count += len(triangles)
            for triangle in triangles:
                projected_triangle = _project_triangle(
                    triangle,
                    intrinsic=intrinsic,
                    width=camera.width,
                    height=camera.height,
                )
                if projected_triangle is None:
                    continue
                x_min, x_max, y_min, y_max = projected_triangle.bounds
                candidate_pixels = (x_max - x_min + 1) * (y_max - y_min + 1)
                projected_pixel_count += candidate_pixels
                _enforce_projected_pixel_limit(
                    projected_pixel_count,
                    maximum_projected_pixels=config.maximum_projected_pixels,
                )
                projected_triangles.append(projected_triangle)
        raster_triangle_count = len(projected_triangles)
        for projected_triangle_item in projected_triangles:
            covered_fragment_count += _rasterize_triangle(
                projected_triangle_item,
                z_buffer=z_buffer,
            )
    else:
        edges_camera = (
            edges_scene @ camera_from_scene[:3, :3].T + camera_from_scene[:3, 3]
        )
        projected_edges: list[_ProjectedSegment] = []
        for edge_camera in edges_camera:
            was_clipped = bool(np.any(edge_camera[:, 2] < config.near_plane_m))
            clipped_edge = _clip_segment_to_near_plane(
                edge_camera,
                config.near_plane_m,
            )
            if clipped_edge is None:
                near_rejected_edges += 1
                continue
            if was_clipped:
                near_clipped_edges += 1
            projected_edge = _project_segment(
                clipped_edge,
                intrinsic=intrinsic,
                width=camera.width,
                height=camera.height,
                edge_width_px=config.edge_width_px,
            )
            if projected_edge is None:
                continue
            candidate_pixels = (
                projected_edge.sample_count
                * config.edge_width_px
                * config.edge_width_px
            )
            projected_pixel_count += candidate_pixels
            _enforce_projected_pixel_limit(
                projected_pixel_count,
                maximum_projected_pixels=config.maximum_projected_pixels,
            )
            projected_edges.append(projected_edge)
        raster_edge_count = len(projected_edges)
        for projected_edge_item in projected_edges:
            covered_fragment_count += _rasterize_segment(
                projected_edge_item,
                z_buffer=z_buffer,
                edge_width_px=config.edge_width_px,
            )

    surface = np.isfinite(z_buffer)
    alpha_plane = alpha_array[..., 0]
    depth_plane = depth_array[..., 0]
    background_valid = (alpha_plane > 0.0) & (depth_plane > 0.0)
    drawn = surface & (
        ~background_valid
        | (z_buffer <= depth_plane.astype(np.float64) + config.depth_epsilon_m)
    )
    occluded = surface & ~drawn

    background_color = np.asarray(config.background_color_rgb, dtype=np.float32)
    composed = rgb_array + background_color[None, None, :] * (
        np.float32(1.0) - alpha_array
    )
    if np.any(drawn):
        surface_color = np.asarray(config.surface_color_rgb, dtype=np.float32)
        opacity = np.float32(
            config.surface_opacity
            if config.render_style is CourtAABBRenderStyle.SOLID
            else config.edge_opacity
        )
        composed[drawn] = (
            surface_color * opacity
            + composed[drawn] * (np.float32(1.0) - opacity)
        )
    output = np.rint(np.clip(composed, 0.0, 1.0) * np.float32(255.0)).astype(
        np.uint8
    )
    surface_pixels = int(np.count_nonzero(surface))
    edge_pixels = (
        surface_pixels
        if config.render_style is CourtAABBRenderStyle.WIREFRAME
        else 0
    )
    drawn_pixels = int(np.count_nonzero(drawn))
    occluded_pixels = int(np.count_nonzero(occluded))
    stats = CourtAABBRenderStats(
        cell_count=geometry.cell_count,
        surface_face_count=int(faces_scene.shape[0]),
        source_triangle_count=2 * int(faces_scene.shape[0]),
        candidate_edge_segment_count=geometry.candidate_edge_segment_count,
        edge_segment_count=int(edges_scene.shape[0]),
        suppressed_seam_segment_count=geometry.suppressed_seam_segment_count,
        near_clipped_face_count=near_clipped_faces,
        near_rejected_face_count=near_rejected_faces,
        triangle_count=triangle_count,
        raster_triangle_count=raster_triangle_count,
        near_clipped_edge_segment_count=near_clipped_edges,
        near_rejected_edge_segment_count=near_rejected_edges,
        raster_edge_segment_count=raster_edge_count,
        projected_pixel_count=projected_pixel_count,
        covered_fragment_count=covered_fragment_count,
        surface_pixel_count=surface_pixels,
        drawn_pixel_count=drawn_pixels,
        occluded_pixel_count=occluded_pixels,
        edge_pixel_count=edge_pixels,
        drawn_edge_pixel_count=(drawn_pixels if edge_pixels else 0),
        occluded_edge_pixel_count=(occluded_pixels if edge_pixels else 0),
        background_valid_pixel_count=int(np.count_nonzero(background_valid)),
        background_invalid_pixel_count=int(background_valid.size - np.count_nonzero(background_valid)),
    )
    return CourtAABBRenderResult(camera_id=camera.camera_id, rgb=output, stats=stats)


def _validated_cells(
    value: object,
    *,
    maximum_cells: int,
) -> NDArray[np.int64]:
    if not isinstance(value, np.ndarray):
        raise TypeError("occupancy_cells must be a numpy array.")
    if value.dtype != np.dtype(np.int64):
        raise TypeError("occupancy_cells must have exact dtype int64.")
    if value.ndim != 2 or value.shape[1:] != (3,):
        raise ValueError("occupancy_cells must have exact shape [N,3].")
    if value.shape[0] > maximum_cells:
        raise ValueError(
            f"occupancy_cells exceeds maximum_cells={maximum_cells}; "
            "rendering was not truncated."
        )
    previous: tuple[int, int, int] | None = None
    for row in value:
        current = (int(row[0]), int(row[1]), int(row[2]))
        if previous is not None and current <= previous:
            raise ValueError(
                "occupancy_cells must be unique and strictly lexicographically sorted."
            )
        previous = current
    return cast(NDArray[np.int64], value)


def _validated_trajectory_centers(value: object) -> NDArray[np.float64]:
    if not isinstance(value, np.ndarray):
        raise TypeError("trajectory_centers_scene_m must be a numpy array.")
    if value.dtype != np.dtype(np.float64):
        raise TypeError("trajectory_centers_scene_m must have exact dtype float64.")
    if value.ndim != 2 or value.shape[1:] != (3,):
        raise ValueError("trajectory_centers_scene_m must have exact shape [N,3].")
    if value.shape[0] == 0:
        raise ValueError("trajectory_centers_scene_m must not be empty.")
    if not np.isfinite(value).all():
        raise ValueError("trajectory_centers_scene_m must contain only finite values.")
    return cast(NDArray[np.float64], value)


def _validated_frame_arrays(
    *,
    rgb: object,
    alpha: object,
    metric_depth: object,
    camera: SceneCamera,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.dtype(np.float32):
        raise TypeError("rgb must be a float32 numpy array.")
    expected_rgb_shape = (camera.height, camera.width, 3)
    if rgb.shape != expected_rgb_shape:
        raise ValueError(f"rgb must have camera-exact shape {expected_rgb_shape}.")
    if not isinstance(alpha, np.ndarray) or alpha.dtype != np.dtype(np.float32):
        raise TypeError("alpha must be a float32 numpy array.")
    expected_plane_shape = (camera.height, camera.width, 1)
    if alpha.shape != expected_plane_shape:
        raise ValueError(f"alpha must have camera-exact shape {expected_plane_shape}.")
    if not isinstance(metric_depth, np.ndarray) or metric_depth.dtype != np.dtype(np.float32):
        raise TypeError("metric_depth must be a float32 numpy array.")
    if metric_depth.shape != expected_plane_shape:
        raise ValueError(
            f"metric_depth must have camera-exact shape {expected_plane_shape}."
        )
    if not np.isfinite(rgb).all() or np.any(rgb < 0.0) or np.any(rgb > 1.0):
        raise ValueError("rgb must contain only finite values in [0,1].")
    if not np.isfinite(alpha).all() or np.any(alpha < 0.0) or np.any(alpha > 1.0):
        raise ValueError("alpha must contain only finite values in [0,1].")
    if not np.isfinite(metric_depth).all() or np.any(metric_depth < 0.0):
        raise ValueError("metric_depth must contain only finite nonnegative values.")
    return (
        cast(NDArray[np.float32], rgb),
        cast(NDArray[np.float32], alpha),
        cast(NDArray[np.float32], metric_depth),
    )


def _enforce_projected_pixel_limit(
    projected_pixel_count: int,
    *,
    maximum_projected_pixels: int,
) -> None:
    if projected_pixel_count > maximum_projected_pixels:
        raise ValueError(
            "projected obstacle raster exceeds maximum_projected_pixels="
            f"{maximum_projected_pixels}; rendering was not truncated."
        )


def _clip_segment_to_near_plane(
    segment: NDArray[np.float64],
    near_plane: float,
) -> NDArray[np.float64] | None:
    start_inside = bool(segment[0, 2] >= near_plane)
    end_inside = bool(segment[1, 2] >= near_plane)
    if not start_inside and not end_inside:
        return None
    clipped = segment.copy()
    if start_inside != end_inside:
        interpolation = (near_plane - segment[0, 2]) / (
            segment[1, 2] - segment[0, 2]
        )
        intersection = segment[0] + interpolation * (segment[1] - segment[0])
        intersection[2] = near_plane
        clipped[0 if not start_inside else 1] = intersection
    return clipped


def _project_segment(
    segment: NDArray[np.float64],
    *,
    intrinsic: NDArray[np.float64],
    width: int,
    height: int,
    edge_width_px: int,
) -> _ProjectedSegment | None:
    homogeneous = segment @ intrinsic.T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    if not np.isfinite(pixels).all():
        raise ValueError("projected obstacle edge pixels must be finite.")
    offsets = _edge_brush_offsets(edge_width_px)
    clipped = _clip_line_to_rectangle(
        pixels,
        x_min=float(-int(np.max(offsets))) - 0.5,
        x_max=float(width - 1 - int(np.min(offsets))) + 0.5,
        y_min=float(-int(np.max(offsets))) - 0.5,
        y_max=float(height - 1 - int(np.min(offsets))) + 0.5,
    )
    if clipped is None:
        return None
    clipped_pixels, start_parameter, end_parameter = clipped
    inverse_depths = np.reciprocal(segment[:, 2])
    clipped_inverse_depths = np.asarray(
        (
            (1.0 - start_parameter) * inverse_depths[0]
            + start_parameter * inverse_depths[1],
            (1.0 - end_parameter) * inverse_depths[0]
            + end_parameter * inverse_depths[1],
        ),
        dtype=np.float64,
    )
    camera_depths = np.reciprocal(clipped_inverse_depths)
    maximum_delta = float(np.max(np.abs(clipped_pixels[1] - clipped_pixels[0])))
    sample_count = max(1, int(math.ceil(maximum_delta)) + 1)
    return _ProjectedSegment(
        camera_depths=camera_depths,
        pixels=clipped_pixels,
        sample_count=sample_count,
    )


def _clip_line_to_rectangle(
    pixels: NDArray[np.float64],
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[NDArray[np.float64], float, float] | None:
    start = pixels[0]
    delta = pixels[1] - start
    lower_parameter = 0.0
    upper_parameter = 1.0
    for coordinate, change, lower, upper in (
        (float(start[0]), float(delta[0]), x_min, x_max),
        (float(start[1]), float(delta[1]), y_min, y_max),
    ):
        if change == 0.0:
            if coordinate < lower or coordinate > upper:
                return None
            continue
        first = (lower - coordinate) / change
        second = (upper - coordinate) / change
        entry = min(first, second)
        exit_ = max(first, second)
        lower_parameter = max(lower_parameter, entry)
        upper_parameter = min(upper_parameter, exit_)
        if lower_parameter > upper_parameter:
            return None
    clipped = np.stack(
        (
            start + lower_parameter * delta,
            start + upper_parameter * delta,
        ),
        axis=0,
    )
    return cast(NDArray[np.float64], clipped), lower_parameter, upper_parameter


def _edge_brush_offsets(edge_width_px: int) -> NDArray[np.int64]:
    start = -(edge_width_px // 2)
    return np.arange(start, start + edge_width_px, dtype=np.int64)


def _rasterize_segment(
    segment: _ProjectedSegment,
    *,
    z_buffer: NDArray[np.float64],
    edge_width_px: int,
) -> int:
    if segment.sample_count == 1:
        parameters: NDArray[np.float64] = np.zeros((1,), dtype=np.float64)
    else:
        parameters = np.linspace(
            0.0,
            1.0,
            num=segment.sample_count,
            dtype=np.float64,
        )
    projected = (
        segment.pixels[0][None, :] * (1.0 - parameters[:, None])
        + segment.pixels[1][None, :] * parameters[:, None]
    )
    centres = np.floor(projected + 0.5).astype(np.int64)
    offsets = _edge_brush_offsets(edge_width_px)
    offset_y, offset_x = np.meshgrid(offsets, offsets, indexing="ij")
    flat_offset_x = offset_x.reshape(-1)
    flat_offset_y = offset_y.reshape(-1)
    x_coordinates = (centres[:, 0, None] + flat_offset_x[None, :]).reshape(-1)
    y_coordinates = (centres[:, 1, None] + flat_offset_y[None, :]).reshape(-1)
    pixel_delta = segment.pixels[1] - segment.pixels[0]
    squared_length = float(np.dot(pixel_delta, pixel_delta))
    if squared_length == 0.0:
        depths = np.full(
            x_coordinates.shape,
            float(np.min(segment.camera_depths)),
            dtype=np.float64,
        )
    else:
        closest_parameters = np.clip(
            (
                (x_coordinates.astype(np.float64) - segment.pixels[0, 0])
                * pixel_delta[0]
                + (y_coordinates.astype(np.float64) - segment.pixels[0, 1])
                * pixel_delta[1]
            )
            / squared_length,
            0.0,
            1.0,
        )
        inverse_depths = (
            (1.0 - closest_parameters) / segment.camera_depths[0]
            + closest_parameters / segment.camera_depths[1]
        )
        depths = np.reciprocal(inverse_depths)
    valid = (
        (x_coordinates >= 0)
        & (x_coordinates < z_buffer.shape[1])
        & (y_coordinates >= 0)
        & (y_coordinates < z_buffer.shape[0])
    )
    if not np.any(valid):
        return 0
    np.minimum.at(
        z_buffer,
        (y_coordinates[valid], x_coordinates[valid]),
        depths[valid],
    )
    return int(np.count_nonzero(valid))


def _clip_polygon_to_near_plane(
    polygon: NDArray[np.float64],
    near_plane: float,
) -> NDArray[np.float64]:
    vertices = [np.asarray(vertex, dtype=np.float64) for vertex in polygon]
    clipped: list[NDArray[np.float64]] = []
    if not vertices:
        return np.empty((0, 3), dtype=np.float64)
    previous = vertices[-1]
    previous_inside = bool(previous[2] >= near_plane)
    for current in vertices:
        current_inside = bool(current[2] >= near_plane)
        if current_inside != previous_inside:
            denominator = current[2] - previous[2]
            interpolation = (near_plane - previous[2]) / denominator
            intersection = previous + interpolation * (current - previous)
            intersection[2] = near_plane
            clipped.append(intersection)
        if current_inside:
            clipped.append(current.copy())
        previous = current
        previous_inside = current_inside
    clipped = _deduplicate_polygon(clipped)
    if len(clipped) < 3:
        return np.empty((0, 3), dtype=np.float64)
    return cast(NDArray[np.float64], np.stack(clipped, axis=0))


def _deduplicate_polygon(
    vertices: list[NDArray[np.float64]],
) -> list[NDArray[np.float64]]:
    result: list[NDArray[np.float64]] = []
    for vertex in vertices:
        if not result or not np.array_equal(vertex, result[-1]):
            result.append(vertex)
    if len(result) > 1 and np.array_equal(result[0], result[-1]):
        result.pop()
    return result


def _triangulate_fan(
    polygon: NDArray[np.float64],
) -> tuple[NDArray[np.float64], ...]:
    return tuple(
        np.stack((polygon[0], polygon[index], polygon[index + 1]), axis=0)
        for index in range(1, polygon.shape[0] - 1)
    )


def _project_triangle(
    triangle: NDArray[np.float64],
    *,
    intrinsic: NDArray[np.float64],
    width: int,
    height: int,
) -> _ProjectedTriangle | None:
    homogeneous = triangle @ intrinsic.T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    if not np.isfinite(pixels).all():
        raise ValueError("projected obstacle pixels must be finite.")
    area = _signed_double_area(pixels)
    if abs(area) <= _DEGENERATE_AREA_EPSILON:
        return None
    minimum = np.min(pixels, axis=0)
    maximum = np.max(pixels, axis=0)
    if (
        maximum[0] < 0.0
        or maximum[1] < 0.0
        or minimum[0] > width - 1
        or minimum[1] > height - 1
    ):
        return None
    x_min = int(math.ceil(max(0.0, float(minimum[0]))))
    x_max = int(math.floor(min(float(width - 1), float(maximum[0]))))
    y_min = int(math.ceil(max(0.0, float(minimum[1]))))
    y_max = int(math.floor(min(float(height - 1), float(maximum[1]))))
    if x_min > x_max or y_min > y_max:
        return None
    return _ProjectedTriangle(
        vertices_camera=triangle,
        pixels=pixels,
        bounds=(x_min, x_max, y_min, y_max),
    )


def _signed_double_area(pixels: NDArray[np.float64]) -> float:
    return float(
        (pixels[1, 0] - pixels[0, 0]) * (pixels[2, 1] - pixels[0, 1])
        - (pixels[1, 1] - pixels[0, 1]) * (pixels[2, 0] - pixels[0, 0])
    )


def _rasterize_triangle(
    triangle: _ProjectedTriangle,
    *,
    z_buffer: NDArray[np.float64],
) -> int:
    x_min, x_max, y_min, y_max = triangle.bounds
    x_coordinates: NDArray[np.float64] = np.arange(
        x_min,
        x_max + 1,
        dtype=np.float64,
    )
    y_coordinates: NDArray[np.float64] = np.arange(
        y_min,
        y_max + 1,
        dtype=np.float64,
    )
    pixel_x, pixel_y = np.meshgrid(x_coordinates, y_coordinates)
    pixels = triangle.pixels
    denominator = (
        (pixels[1, 1] - pixels[2, 1]) * (pixels[0, 0] - pixels[2, 0])
        + (pixels[2, 0] - pixels[1, 0]) * (pixels[0, 1] - pixels[2, 1])
    )
    lambda_zero = (
        (pixels[1, 1] - pixels[2, 1]) * (pixel_x - pixels[2, 0])
        + (pixels[2, 0] - pixels[1, 0]) * (pixel_y - pixels[2, 1])
    ) / denominator
    lambda_one = (
        (pixels[2, 1] - pixels[0, 1]) * (pixel_x - pixels[2, 0])
        + (pixels[0, 0] - pixels[2, 0]) * (pixel_y - pixels[2, 1])
    ) / denominator
    lambda_two = np.float64(1.0) - lambda_zero - lambda_one
    inside = (
        (lambda_zero >= -_BARYCENTRIC_EPSILON)
        & (lambda_one >= -_BARYCENTRIC_EPSILON)
        & (lambda_two >= -_BARYCENTRIC_EPSILON)
    )
    covered = int(np.count_nonzero(inside))
    if covered == 0:
        return 0
    inverse_depth = (
        lambda_zero / triangle.vertices_camera[0, 2]
        + lambda_one / triangle.vertices_camera[1, 2]
        + lambda_two / triangle.vertices_camera[2, 2]
    )
    fragment_depth = np.reciprocal(inverse_depth)
    current = z_buffer[y_min : y_max + 1, x_min : x_max + 1]
    update = inside & (fragment_depth < current)
    current[update] = fragment_depth[update]
    return covered


def _rgb_tuple(value: object, *, name: str) -> tuple[float, float, float]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError(f"{name} must be a tuple of exactly three numbers.")
    result = tuple(_unit_float(item, name=name) for item in value)
    return cast(tuple[float, float, float], result)


def _positive_float(value: object, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _nonnegative_float(value: object, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _unit_float(value: object, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be in [0,1].")
    return result


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


__all__ = [
    "COURT_AABB_TRAJECTORY_DISTANCE_METRIC",
    "CourtAABBRenderConfig",
    "CourtAABBRenderResult",
    "CourtAABBRenderStats",
    "CourtAABBTrajectoryFilterResult",
    "PreparedCourtAABBGeometry",
    "PreparedCourtAABBTrajectoryFilter",
    "extract_canonical_exposed_face_edges",
    "extract_exposed_voxel_faces",
    "filter_court_obstacle_cells_by_trajectory",
    "prepare_court_aabb_trajectory_filter",
    "prepare_court_obstacle_aabbs",
    "render_court_obstacle_aabbs",
    "render_prepared_court_obstacle_aabbs",
    "segment_aabb_distance_squared",
]
