"""Fixed colorblind-safe camera and overview publication figures."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import findfont
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.synthetic_data_generation.scene_contract import MultiCourtLayout
from src.synthetic_data_generation.visualization.publication.cameras import (
    PublicationCameraCollection,
)
from src.utils.rendering.camera_geometry import (
    camera_coverage_segments,
    camera_trajectory_points,
    camera_trajectory_segments,
    camera_view_direction_segments,
)
from src.utils.schema.court import (
    COURT_SKELETON,
    STANDARD_COURT_CONFIG,
    court_keypoints_3d,
)

CAMERA_COVERAGE_METRIC_SCHEMA = "publication_camera_coverage_v1"
OVERVIEW_LAYOUT_SCHEMA = "publication_overview_layout_v1"
_BLUE = "#0072B2"
_ORANGE = "#D55E00"
_GREEN = "#009E73"
_MAGENTA = "#CC79A7"
_BLACK = "#222222"


def camera_collection_metrics(
    collection: PublicationCameraCollection,
) -> Mapping[str, object]:
    """Compute exact-order camera coverage metrics in metric scene units."""
    centres = camera_trajectory_points(collection.camera_to_metric_scene)
    segments = camera_trajectory_segments(collection.camera_to_metric_scene)
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    return {
        "schema": CAMERA_COVERAGE_METRIC_SCHEMA,
        "owner": collection.owner,
        "camera_count": len(collection.camera_ids),
        "trajectory_segment_count": len(segments),
        "trajectory_length_metres": float(np.sum(lengths)),
        "maximum_adjacent_displacement_metres": (
            0.0 if len(lengths) == 0 else float(np.max(lengths))
        ),
        "centre_bounds_metric_scene": [
            [float(item) for item in np.min(centres, axis=0)],
            [float(item) for item in np.max(centres, axis=0)],
        ],
    }


def render_camera_figure(
    collection: PublicationCameraCollection,
    layout: MultiCourtLayout,
    output: Path,
    *,
    size: tuple[int, int],
    frustum_depth_metres: float,
    line_width: float,
    font_size: int,
) -> None:
    """Render every explicit camera, frustum, orientation, trajectory, and court."""
    figure = Figure(figsize=(size[0] / 100.0, size[1] / 100.0), dpi=100)
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1, projection="3d")
    frusta = camera_coverage_segments(
        collection.intrinsics,
        collection.image_sizes,
        collection.camera_to_metric_scene,
        depth=frustum_depth_metres,
    )
    trajectory = camera_trajectory_segments(collection.camera_to_metric_scene)
    directions = camera_view_direction_segments(
        collection.camera_to_metric_scene,
        length=frustum_depth_metres * 0.65,
    )
    court_segments = _metric_court_segments(layout)
    axis.add_collection3d(
        Line3DCollection(court_segments, colors=_BLACK, linewidths=line_width + 0.5)
    )
    axis.add_collection3d(
        Line3DCollection(
            frusta.reshape(-1, 2, 3), colors=_BLUE, linewidths=line_width, alpha=0.55
        )
    )
    if len(trajectory):
        axis.add_collection3d(
            Line3DCollection(trajectory, colors=_ORANGE, linewidths=line_width + 0.8)
        )
    axis.add_collection3d(
        Line3DCollection(directions, colors=_GREEN, linewidths=line_width + 0.2)
    )
    centres = camera_trajectory_points(collection.camera_to_metric_scene)
    axis.scatter(
        centres[:, 0], centres[:, 1], centres[:, 2], c=_BLUE, s=18, depthshade=False
    )
    for camera_id, centre in zip(collection.camera_ids, centres, strict=True):
        axis.text(
            float(centre[0]),
            float(centre[1]),
            float(centre[2]),
            camera_id,
            fontsize=max(5, font_size - 3),
            color=_BLACK,
        )
    _set_metric_3d_bounds(
        axis, np.concatenate((centres, court_segments.reshape(-1, 3)))
    )
    logical = (
        "captured reconstruction"
        if collection.logical_scene_id is None
        else collection.logical_scene_id
    )
    axis.set_title(
        f"{collection.owner.upper()} camera coverage — {logical}\n"
        "OpenCV +z view direction; metric scene coordinates",
        fontsize=font_size + 2,
    )
    axis.set_xlabel("scene x (m)", fontsize=font_size)
    axis.set_ylabel("scene y (m)", fontsize=font_size)
    axis.set_zlabel("scene z (m)", fontsize=font_size)
    axis.view_init(elev=28, azim=-55)
    figure.subplots_adjust(left=0.02, right=0.96, bottom=0.03, top=0.90)
    _save_canvas_png(canvas, output, size=size)


def render_camera_comparison_figure(
    blcs: PublicationCameraCollection,
    plcs: PublicationCameraCollection,
    layout: MultiCourtLayout,
    output: Path,
    *,
    size: tuple[int, int],
    frustum_depth_metres: float,
    line_width: float,
    font_size: int,
) -> None:
    """Render BLCS and PLCS camera geometries on one shared metric axis."""
    figure = Figure(figsize=(size[0] / 100.0, size[1] / 100.0), dpi=100)
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1, projection="3d")
    court_segments = _metric_court_segments(layout)
    axis.add_collection3d(
        Line3DCollection(court_segments, colors=_BLACK, linewidths=line_width + 0.7)
    )
    all_points = [court_segments.reshape(-1, 3)]
    for collection, color, label in (
        (blcs, _BLUE, "BLCS"),
        (plcs, _MAGENTA, "PLCS"),
    ):
        frusta = camera_coverage_segments(
            collection.intrinsics,
            collection.image_sizes,
            collection.camera_to_metric_scene,
            depth=frustum_depth_metres,
        )
        directions = camera_view_direction_segments(
            collection.camera_to_metric_scene,
            length=frustum_depth_metres * 0.65,
        )
        centres = camera_trajectory_points(collection.camera_to_metric_scene)
        all_points.extend((frusta.reshape(-1, 3), centres))
        axis.add_collection3d(
            Line3DCollection(
                frusta.reshape(-1, 2, 3),
                colors=color,
                linewidths=line_width,
                alpha=0.55,
            )
        )
        axis.add_collection3d(
            Line3DCollection(directions, colors=color, linewidths=line_width + 0.2)
        )
        axis.scatter(
            centres[:, 0],
            centres[:, 1],
            centres[:, 2],
            c=color,
            s=26,
            label=f"{label} ({len(collection.camera_ids)} cameras)",
            depthshade=False,
        )
    _set_metric_3d_bounds(axis, np.concatenate(all_points))
    axis.set_title(
        "BLCS vs PLCS camera layout — shared metric scene / OpenCV axes",
        fontsize=font_size + 2,
    )
    axis.set_xlabel("scene x (m)", fontsize=font_size)
    axis.set_ylabel("scene y (m)", fontsize=font_size)
    axis.set_zlabel("scene z (m)", fontsize=font_size)
    axis.legend(loc="upper left", fontsize=font_size)
    axis.view_init(elev=28, azim=-55)
    figure.subplots_adjust(left=0.02, right=0.96, bottom=0.03, top=0.91)
    _save_canvas_png(canvas, output, size=size)


def overview_panel_bounds(
    size: tuple[int, int],
) -> tuple[tuple[str, tuple[int, int, int, int]], ...]:
    """Return the fixed six-panel pixel bounds, all strictly within the canvas."""
    width, height = size
    if width < 600 or height < 400:
        raise ValueError("Overview size must be at least 600x400 pixels.")
    margin_x = max(20, width // 50)
    gap_x = max(12, width // 100)
    header = max(48, height // 12)
    footer = max(58, height // 11)
    gap_y = max(12, height // 80)
    panel_width = (width - 2 * margin_x - 2 * gap_x) // 3
    panel_height = (height - header - footer - gap_y) // 2
    labels = (
        "Court dataset",
        "BLCS dataset",
        "PLCS dataset",
        "Alignment evidence",
        "Captured cameras",
        "BLCS / PLCS cameras",
    )
    bounds: list[tuple[str, tuple[int, int, int, int]]] = []
    for index, label in enumerate(labels):
        row, column = divmod(index, 3)
        left = margin_x + column * (panel_width + gap_x)
        top = header + row * (panel_height + gap_y)
        bounds.append((label, (left, top, left + panel_width, top + panel_height)))
    if any(
        left < 0
        or top < 0
        or right > width
        or bottom > height
        or left >= right
        or top >= bottom
        for _, (left, top, right, bottom) in bounds
    ):
        raise ValueError("Overview panel bounds leave the configured canvas.")
    return tuple(bounds)


def render_publication_overview(
    bundle_root: Path,
    output: Path,
    *,
    size: tuple[int, int],
    scene_id: str,
    alignment_metrics: Mapping[str, object],
    camera_metrics: Mapping[str, Mapping[str, object]],
    font_size: int,
) -> tuple[Mapping[str, object], ...]:
    """Compose the fixed overview solely from already-rendered validated panels."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Publication staging artifact already exists: {output}")
    panel_sources = (
        bundle_root / "dataset-court.gif",
        bundle_root / "dataset-blcs.gif",
        bundle_root / "dataset-plcs.gif",
        bundle_root / "alignment-heatmap-court.png",
        bundle_root / "captured-camera-trajectory.png",
        bundle_root / "camera-layout-comparison.png",
    )
    if any(path.is_symlink() or not path.is_file() for path in panel_sources):
        raise FileNotFoundError("Overview requires every upstream rendered panel.")
    bounds = overview_panel_bounds(size)
    canvas = Image.new("RGB", size, "#FFFFFF")
    draw = ImageDraw.Draw(canvas)
    font_path = findfont("DejaVu Sans")
    title_font = ImageFont.truetype(font_path, max(font_size + 7, 16))
    label_font = ImageFont.truetype(font_path, max(font_size, 11))
    metric_font = ImageFont.truetype(font_path, max(font_size - 2, 9))
    title = f"Synthetic-data publication overview — scene {scene_id}"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    draw.text(
        ((size[0] - (title_bbox[2] - title_bbox[0])) // 2, 12),
        title,
        fill=_BLACK,
        font=title_font,
    )
    mapping: list[Mapping[str, object]] = []
    for (label, (left, top, right, bottom)), source in zip(
        bounds, panel_sources, strict=True
    ):
        with Image.open(source) as image:
            image.seek(0)
            rgb = image.convert("RGB")
            label_height = max(24, font_size + 9)
            available = (right - left - 8, bottom - top - label_height - 8)
            fitted = ImageOps.contain(rgb, available, Image.Resampling.LANCZOS)
        paste_left = left + (right - left - fitted.width) // 2
        paste_top = (
            top + label_height + (bottom - top - label_height - fitted.height) // 2
        )
        canvas.paste(fitted, (paste_left, paste_top))
        draw.rectangle((left, top, right - 1, bottom - 1), outline=_BLUE, width=2)
        draw.rectangle((left, top, right - 1, top + label_height), fill="#EAF4FA")
        draw.text((left + 8, top + 4), label, fill=_BLACK, font=label_font)
        mapping.append(
            {
                "panel": label,
                "source_artifact": source.name,
                "bounds_pixels": [left, top, right, bottom],
            }
        )
    footer_top = max(bottom for _, (_, _, _, bottom) in bounds) + 8
    metric_line = (
        "Agreement: mean court-line probability "
        f"{_numeric_metric(alignment_metrics, 'court_line_mean_probability'):.3f}; "
        "evidence q95 distance "
        f"{_numeric_metric(alignment_metrics, 'projected_evidence_nearest_court_q95_metres'):.3f} m.  "
        "Cameras: captured "
        f"{_integer_metric(camera_metrics['reconstruction'], 'camera_count')}, "
        f"BLCS {_integer_metric(camera_metrics['blcs'], 'camera_count')}, "
        f"PLCS {_integer_metric(camera_metrics['plcs'], 'camera_count')}."
    )
    metric_bbox = draw.textbbox((0, 0), metric_line, font=metric_font)
    if metric_bbox[2] > size[0] - 24:
        raise ValueError(
            "Overview quantitative summary exceeds the fixed canvas width."
        )
    draw.text((12, footer_top), metric_line, fill=_BLACK, font=metric_font)
    canvas.save(output, format="PNG", optimize=False, compress_level=9)
    with Image.open(output) as reopened:
        if reopened.format != "PNG" or reopened.size != size:
            raise ValueError("Overview media metadata differs after reopening.")
    return tuple(mapping)


def _metric_court_segments(layout: MultiCourtLayout) -> NDArray[np.float64]:
    local = np.asarray(
        court_keypoints_3d(STANDARD_COURT_CONFIG)[:14].detach().cpu().numpy(),
        dtype=np.float64,
    )
    segments: list[NDArray[np.float64]] = []
    for court in layout.courts:
        points = court.scene_from_court.apply(local)
        segments.extend(
            points[[first, second]]
            for first, second in COURT_SKELETON
            if first < len(points) and second < len(points)
        )
    if not segments:
        raise ValueError("Metric layout contains no renderable court segments.")
    return np.stack(segments)


def _set_metric_3d_bounds(axis: Axes3D, points: NDArray[np.float64]) -> None:
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("3-D figure bounds require finite (N, 3) points.")
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    span = np.maximum(upper - lower, 1.0)
    centre = (lower + upper) / 2.0
    half = float(np.max(span)) * 0.58
    axis.set_xlim(float(centre[0] - half), float(centre[0] + half))
    axis.set_ylim(float(centre[1] - half), float(centre[1] + half))
    axis.set_zlim(float(centre[2] - half), float(centre[2] + half))
    axis.set_box_aspect((1.0, 1.0, 1.0))


def _numeric_metric(metrics: Mapping[str, object], name: str) -> float:
    value = metrics[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Metric {name!r} must be numeric.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Metric {name!r} must be finite.")
    return result


def _integer_metric(metrics: Mapping[str, object], name: str) -> int:
    value = metrics[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Metric {name!r} must be an integer.")
    return value


def _save_canvas_png(
    canvas: FigureCanvasAgg,
    output: Path,
    *,
    size: tuple[int, int],
) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Publication staging artifact already exists: {output}")
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    if rgba.shape != (size[1], size[0], 4):
        raise ValueError("Publication PNG canvas dimensions changed.")
    Image.fromarray(rgba[..., :3], mode="RGB").save(
        output,
        format="PNG",
        optimize=False,
        compress_level=9,
    )


__all__ = [
    "CAMERA_COVERAGE_METRIC_SCHEMA",
    "OVERVIEW_LAYOUT_SCHEMA",
    "camera_collection_metrics",
    "overview_panel_bounds",
    "render_camera_comparison_figure",
    "render_camera_figure",
    "render_publication_overview",
]
