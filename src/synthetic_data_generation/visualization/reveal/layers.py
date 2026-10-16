"""Procedural RGBA layers drawn by tennis-lab: SfM points and capture cameras."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from src.synthetic_data_generation.scene_contract import SceneCamera
from src.synthetic_data_generation.visualization.reveal.viewpoint import RevealCamera
from src.utils.rendering.camera_geometry import camera_frustum_segments

# Camera-Z depth below which a projected vertex is discarded, in scene units.
_MINIMUM_DEPTH = 1.0e-6
# Overlay lines are drawn at this supersampling factor and box-filtered down.
_LINE_SUPERSAMPLING = 4


@dataclass(frozen=True, slots=True)
class ProjectedPoints:
    """Visible, depth-resolved point splats in a fixed reveal order.

    ``pixel_index[i]`` and ``colors[i]`` belong to the point at reveal rank
    ``rank[i]``. Only the nearest point per pixel is retained, so drawing any
    reveal prefix is a scatter without further depth tests.
    """

    width: int
    height: int
    pixel_index: NDArray[np.intp]
    colors: NDArray[np.float32]
    rank: NDArray[np.intp]
    point_count: int


def project_points(
    camera: RevealCamera,
    points_scene: NDArray[np.float32],
    reveal_order: NDArray[np.intp],
    *,
    radius_px: int,
) -> ProjectedPoints:
    """Z-buffer ``(N, 6)`` XYZRGB points as square splats of ``radius_px``.

    Each output pixel keeps the nearest splat; a later reveal rank never hides
    an earlier nearer point because occlusion is resolved once for all ranks.
    """
    points = np.asarray(points_scene)
    order = np.asarray(reveal_order)
    if sorted(order.tolist()) != list(range(len(points))):
        raise ValueError("reveal_order must be a permutation of the points.")
    ranks: NDArray[np.intp] = np.empty(len(points), dtype=np.intp)
    ranks[order] = np.arange(len(points))
    pixels, depth = camera.project(points[:, :3].astype(np.float64))
    source = np.flatnonzero(depth > _MINIMUM_DEPTH)
    centre = np.floor(pixels[source]).astype(np.int64)
    offsets = np.arange(-(radius_px - 1), radius_px)
    candidates_pixel: list[NDArray[np.int64]] = []
    candidates_source: list[NDArray[np.intp]] = []
    for dy in offsets:
        for dx in offsets:
            x = centre[:, 0] + dx
            y = centre[:, 1] + dy
            inside = (x >= 0) & (x < camera.width) & (y >= 0) & (y < camera.height)
            candidates_pixel.append(y[inside] * camera.width + x[inside])
            candidates_source.append(source[inside])
    pixel = np.concatenate(candidates_pixel)
    owner = np.concatenate(candidates_source)
    nearest_first = np.lexsort((depth[owner], pixel))
    pixel = pixel[nearest_first]
    owner = owner[nearest_first]
    first: NDArray[np.bool_] = np.ones(len(pixel), dtype=bool)
    first[1:] = pixel[1:] != pixel[:-1]
    pixel = pixel[first]
    owner = owner[first]
    return ProjectedPoints(
        width=camera.width,
        height=camera.height,
        pixel_index=pixel.astype(np.intp),
        colors=points[owner, 3:6].astype(np.float32),
        rank=ranks[owner],
        point_count=len(points),
    )


def draw_points(
    projected: ProjectedPoints, fraction: float
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return premultiplied RGB and alpha of the first ``fraction`` of points."""
    rgb: NDArray[np.float32] = np.zeros(
        (projected.height * projected.width, 3), dtype=np.float32
    )
    alpha: NDArray[np.float32] = np.zeros(
        projected.height * projected.width, dtype=np.float32
    )
    shown = projected.rank < round(fraction * projected.point_count)
    rgb[projected.pixel_index[shown]] = projected.colors[shown]
    alpha[projected.pixel_index[shown]] = 1.0
    return (
        rgb.reshape(projected.height, projected.width, 3),
        alpha.reshape(projected.height, projected.width, 1),
    )


@dataclass(frozen=True, slots=True)
class ProjectedCameras:
    """Capture-camera geometry projected into the reveal image, in capture order."""

    width: int
    height: int
    trajectory_px: NDArray[np.float64]
    trajectory_visible: NDArray[np.bool_]
    frustum_camera_index: NDArray[np.intp]
    frustum_segments_px: NDArray[np.float64]
    frustum_visible: NDArray[np.bool_]
    camera_count: int


def project_cameras(
    camera: RevealCamera,
    capture_cameras: tuple[SceneCamera, ...],
    *,
    frustum_depth_scene: float,
    frustum_stride: int,
) -> ProjectedCameras:
    """Project the capture trajectory and every ``frustum_stride``-th frustum."""
    centres = np.stack([item.camera_to_scene.matrix()[:3, 3] for item in capture_cameras])
    trajectory_px, trajectory_depth = camera.project(centres)
    indices: NDArray[np.intp] = np.arange(
        0, len(capture_cameras), frustum_stride, dtype=np.intp
    )
    segments = np.stack(
        [
            camera_frustum_segments(
                np.asarray(capture_cameras[index].intrinsics).reshape(3, 3),
                (capture_cameras[index].width, capture_cameras[index].height),
                capture_cameras[index].camera_to_scene.matrix(),
                depth=frustum_depth_scene,
            )
            for index in indices
        ]
    )
    segment_px, segment_depth = camera.project(segments.reshape(-1, 3))
    segment_visible = (segment_depth > _MINIMUM_DEPTH).reshape(len(indices), 8, 2).all(axis=2)
    return ProjectedCameras(
        width=camera.width,
        height=camera.height,
        trajectory_px=trajectory_px,
        trajectory_visible=trajectory_depth > _MINIMUM_DEPTH,
        frustum_camera_index=indices,
        frustum_segments_px=segment_px.reshape(len(indices), 8, 2, 2),
        frustum_visible=segment_visible,
        camera_count=len(capture_cameras),
    )


def draw_cameras(
    projected: ProjectedCameras,
    fraction: float,
    *,
    rgb: tuple[float, float, float],
    line_width_px: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return premultiplied RGB and alpha of the first ``fraction`` of cameras."""
    scale = _LINE_SUPERSAMPLING
    canvas = Image.new("L", (projected.width * scale, projected.height * scale), 0)
    draw = ImageDraw.Draw(canvas)
    shown = round(fraction * projected.camera_count)
    width = line_width_px * scale

    def line(start: NDArray[np.float64], end: NDArray[np.float64]) -> None:
        draw.line(
            ((start[0] + 0.5) * scale, (start[1] + 0.5) * scale, (end[0] + 0.5) * scale, (end[1] + 0.5) * scale),
            fill=255,
            width=width,
        )

    for index in range(1, shown):
        if projected.trajectory_visible[index - 1] and projected.trajectory_visible[index]:
            line(projected.trajectory_px[index - 1], projected.trajectory_px[index])
    for row, camera_index in enumerate(projected.frustum_camera_index):
        if camera_index >= shown:
            break
        for segment, visible in zip(
            projected.frustum_segments_px[row], projected.frustum_visible[row], strict=True
        ):
            if visible:
                line(segment[0], segment[1])
    coverage = canvas.resize((projected.width, projected.height), Image.Resampling.BOX)
    alpha = np.asarray(coverage, dtype=np.float32)[..., None] / 255.0
    color = np.asarray(rgb, dtype=np.float32)
    return alpha * color, alpha
