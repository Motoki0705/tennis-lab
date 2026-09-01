"""Deterministic depth-aware rendering of exact Court V4 obstacle voxels.

The renderer treats each occupied integer cell as the closed scene-space AABB
``[cell * voxel_size, (cell + 1) * voxel_size]``.  Only faces exposed by the
exact six-neighbour occupancy relation are rasterized. Input RGB is NHT's raw
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


@dataclass(frozen=True, slots=True)
class CourtAABBRenderConfig:
    """Validated geometry, composition, and fail-closed resource limits."""

    voxel_size_m: float
    near_plane_m: float
    depth_epsilon_m: float
    surface_color_rgb: tuple[float, float, float]
    surface_opacity: float
    background_color_rgb: tuple[float, float, float]
    maximum_cells: int
    maximum_surface_faces: int
    maximum_projected_pixels: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "voxel_size_m",
            _positive_float(self.voxel_size_m, name="voxel_size_m"),
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
        object.__setattr__(
            self,
            "background_color_rgb",
            _rgb_tuple(self.background_color_rgb, name="background_color_rgb"),
        )
        for name in (
            "maximum_cells",
            "maximum_surface_faces",
            "maximum_projected_pixels",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name=name))


@dataclass(frozen=True, slots=True)
class CourtAABBRenderStats:
    """Immutable counters describing one complete, non-truncated render."""

    cell_count: int
    surface_face_count: int
    source_triangle_count: int
    near_clipped_face_count: int
    near_rejected_face_count: int
    triangle_count: int
    raster_triangle_count: int
    projected_pixel_count: int
    covered_fragment_count: int
    surface_pixel_count: int
    drawn_pixel_count: int
    occluded_pixel_count: int
    background_valid_pixel_count: int
    background_invalid_pixel_count: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _nonnegative_int(getattr(self, name), name=name)
        if self.source_triangle_count != 2 * self.surface_face_count:
            raise ValueError("source_triangle_count must equal twice surface_face_count.")
        if self.drawn_pixel_count + self.occluded_pixel_count != self.surface_pixel_count:
            raise ValueError("Every surface pixel must be either drawn or occluded.")


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
        prepared = np.ascontiguousarray(faces, dtype=np.float64)
        prepared.setflags(write=False)
        object.__setattr__(self, "cell_count", cell_count)
        object.__setattr__(self, "voxel_size_m", voxel_size)
        object.__setattr__(self, "faces_scene_m", prepared)

    @property
    def surface_face_count(self) -> int:
        """Return the exact exposed-quad count."""
        return int(self.faces_scene_m.shape[0])


@dataclass(frozen=True, slots=True)
class _ProjectedTriangle:
    vertices_camera: NDArray[np.float64]
    pixels: NDArray[np.float64]
    bounds: tuple[int, int, int, int]


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
    return PreparedCourtAABBGeometry(
        cell_count=int(cells.shape[0]),
        voxel_size_m=config.voxel_size_m,
        faces_scene_m=faces,
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
    """Depth-composite exact exposed obstacle-voxel faces over one 3DGS frame.

    Triangle depths are camera-Z values evaluated perspective-correctly at
    zero-based pixel centres.  A surface is drawn when background depth is
    invalid or when ``surface_z <= metric_depth + depth_epsilon_m``.
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
    rgb_array, alpha_array, depth_array = _validated_frame_arrays(
        rgb=rgb,
        alpha=alpha,
        metric_depth=metric_depth,
        camera=camera,
    )
    faces_scene = geometry.faces_scene_m

    camera_from_scene = camera.camera_to_scene.inverse().matrix()
    faces_camera = (
        faces_scene @ camera_from_scene[:3, :3].T + camera_from_scene[:3, 3]
    )
    intrinsic = np.asarray(camera.intrinsics, dtype=np.float64).reshape(3, 3)
    projected: list[_ProjectedTriangle] = []
    near_clipped_faces = 0
    near_rejected_faces = 0
    triangle_count = 0
    projected_pixel_count = 0

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
            if projected_pixel_count > config.maximum_projected_pixels:
                raise ValueError(
                    "projected obstacle raster exceeds maximum_projected_pixels="
                    f"{config.maximum_projected_pixels}; rendering was not truncated."
                )
            projected.append(projected_triangle)

    z_buffer = np.full((camera.height, camera.width), np.inf, dtype=np.float64)
    covered_fragment_count = 0
    for projected_triangle_item in projected:
        covered_fragment_count += _rasterize_triangle(
            projected_triangle_item,
            z_buffer=z_buffer,
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
        opacity = np.float32(config.surface_opacity)
        composed[drawn] = (
            surface_color * opacity
            + composed[drawn] * (np.float32(1.0) - opacity)
        )
    output = np.rint(np.clip(composed, 0.0, 1.0) * np.float32(255.0)).astype(
        np.uint8
    )
    surface_pixels = int(np.count_nonzero(surface))
    stats = CourtAABBRenderStats(
        cell_count=geometry.cell_count,
        surface_face_count=int(faces_scene.shape[0]),
        source_triangle_count=2 * int(faces_scene.shape[0]),
        near_clipped_face_count=near_clipped_faces,
        near_rejected_face_count=near_rejected_faces,
        triangle_count=triangle_count,
        raster_triangle_count=len(projected),
        projected_pixel_count=projected_pixel_count,
        covered_fragment_count=covered_fragment_count,
        surface_pixel_count=surface_pixels,
        drawn_pixel_count=int(np.count_nonzero(drawn)),
        occluded_pixel_count=int(np.count_nonzero(occluded)),
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
    "CourtAABBRenderConfig",
    "CourtAABBRenderResult",
    "CourtAABBRenderStats",
    "PreparedCourtAABBGeometry",
    "extract_exposed_voxel_faces",
    "prepare_court_obstacle_aabbs",
    "render_court_obstacle_aabbs",
    "render_prepared_court_obstacle_aabbs",
]
