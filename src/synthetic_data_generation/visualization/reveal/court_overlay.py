"""Accepted-alignment overlay: court lines and court axes over the NHT render.

The geometry is the canonical CourtKP20 ground skeleton placed by every
accepted court's ``scene_from_court`` and the alignment's metric/NHT
similarity, i.e. exactly what dataset labels are projected from.

The overlay is drawn on top without a depth test. The NHT depth is an
alpha-weighted expected depth that haze pulls towards the camera (on B00 it
lies 5-16 % short along the court lines), so it cannot separate a real
occluder from a semi-transparent one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from src.synthetic_data_generation.visualization.reveal.contracts import (
    RevealAlignmentStyle,
)
from src.synthetic_data_generation.visualization.reveal.timeline import smoothstep
from src.synthetic_data_generation.visualization.reveal.viewpoint import RevealCamera
from src.utils.schema.court import (
    COURT_SKELETON,
    NUM_GROUND_COURT_KP,
    STANDARD_COURT_CONFIG,
    court_keypoints_3d,
)

# Strokes are sampled at this spacing so that the parts behind the camera
# are clipped and lines grow smoothly, in metres.
_SAMPLE_SPACING_M = 0.05
# Samples nearer than this to the camera plane are dropped, in metres.
_NEAR_PLANE_M = 0.1
# Court lines finish growing at this fraction of the alignment phase; court
# axes and labels then grow in from ``_AXES_START`` to the end of the phase.
_LINE_GROWTH_END = 0.7
_AXES_START = 0.5
# Opacity of the dark outline that keeps lines legible on bright ground.
_OUTLINE_OPACITY = 0.6
_LABEL_LIFT_M = 0.6
_SUPERSAMPLING = 4
_AXIS_RGB: tuple[tuple[float, float, float], ...] = (
    (0.95, 0.27, 0.25),
    (0.35, 0.90, 0.35),
    (0.30, 0.58, 1.00),
)


@dataclass(frozen=True, slots=True)
class OverlayStroke:
    """One straight 3D stroke sampled in NHT scene units.

    ``growth[i]`` is the fraction of the stroke length at sample ``i``; a
    stroke grown to ``g`` shows the samples with ``growth <= g``.
    """

    samples_scene: NDArray[np.float64]
    growth: NDArray[np.float64]
    rgb: tuple[float, float, float]
    is_axis: bool


@dataclass(frozen=True, slots=True)
class CourtOverlayGeometry:
    """Strokes and court-id label anchors of every accepted court."""

    strokes: tuple[OverlayStroke, ...]
    labels: tuple[tuple[str, NDArray[np.float64]], ...]
    units_per_metre: float


def _stroke(
    scene_from_court: NDArray[np.float64],
    start_court: NDArray[np.float64],
    end_court: NDArray[np.float64],
    *,
    rgb: tuple[float, float, float],
    is_axis: bool,
) -> OverlayStroke:
    length = float(np.linalg.norm(end_court - start_court))
    count = max(2, math.ceil(length / _SAMPLE_SPACING_M) + 1)
    growth = np.linspace(0.0, 1.0, count)
    court = start_court + growth[:, None] * (end_court - start_court)
    scene = court @ scene_from_court[:3, :3].T + scene_from_court[:3, 3]
    return OverlayStroke(samples_scene=scene, growth=growth, rgb=rgb, is_axis=is_axis)


def court_overlay_geometry(
    scene_from_court: Mapping[str, NDArray[np.float64]],
    *,
    line_rgb: tuple[float, float, float],
    axis_length_m: float,
    units_per_metre: float,
) -> CourtOverlayGeometry:
    """Build the ground-line and axes strokes of every court, in court-id order.

    ``scene_from_court`` maps each accepted court id to its similarity from
    court metres to NHT scene units.
    """
    if not scene_from_court:
        raise ValueError("The court overlay needs at least one accepted court.")
    keypoints = np.asarray(
        court_keypoints_3d(STANDARD_COURT_CONFIG)[:NUM_GROUND_COURT_KP].numpy(),
        dtype=np.float64,
    )
    edges = tuple(
        (first, second)
        for first, second in COURT_SKELETON
        if first < NUM_GROUND_COURT_KP and second < NUM_GROUND_COURT_KP
    )
    strokes: list[OverlayStroke] = []
    labels: list[tuple[str, NDArray[np.float64]]] = []
    for court_id in sorted(scene_from_court):
        transform = np.asarray(scene_from_court[court_id], dtype=np.float64)
        strokes.extend(
            _stroke(transform, keypoints[first], keypoints[second], rgb=line_rgb, is_axis=False)
            for first, second in edges
        )
        origin = np.zeros(3)
        strokes.extend(
            _stroke(transform, origin, axis_length_m * np.eye(3)[axis], rgb=rgb, is_axis=True)
            for axis, rgb in enumerate(_AXIS_RGB)
        )
        anchor = np.asarray((0.0, 0.0, axis_length_m + _LABEL_LIFT_M))
        labels.append((court_id, transform[:3, :3] @ anchor + transform[:3, 3]))
    return CourtOverlayGeometry(
        strokes=tuple(strokes), labels=tuple(labels), units_per_metre=units_per_metre
    )


def overlay_reveal(progress: float) -> tuple[float, float]:
    """Return (court-line growth, axes/label reveal) at alignment ``progress``."""
    lines = min(max(progress / _LINE_GROWTH_END, 0.0), 1.0)
    axes = smoothstep((progress - _AXES_START) / (1.0 - _AXES_START))
    return lines, axes


def _runs(shown: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Return maximal ``(start, stop)`` runs of consecutive shown samples.

    ``stop`` is inclusive and every run holds at least two samples.
    """
    links = shown[:-1] & shown[1:]
    edges = np.diff(np.concatenate(([0], links.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]


def draw_court_overlay(
    camera: RevealCamera,
    geometry: CourtOverlayGeometry,
    progress: float,
    *,
    style: RevealAlignmentStyle,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return the premultiplied RGB and alpha of the overlay at ``progress``."""
    line_growth, axes_reveal = overlay_reveal(progress)
    scale = _SUPERSAMPLING
    size = (camera.width * scale, camera.height * scale)
    outline = Image.new("L", size, 0)
    core = Image.new("RGB", size, (0, 0, 0))
    coverage = Image.new("L", size, 0)
    outline_draw = ImageDraw.Draw(outline)
    core_draw = ImageDraw.Draw(core)
    coverage_draw = ImageDraw.Draw(coverage)
    core_width = style.line_width_px * scale
    outline_width = (style.line_width_px + 2 * style.outline_px) * scale
    near = _NEAR_PLANE_M * geometry.units_per_metre

    polylines: list[tuple[OverlayStroke, list[tuple[float, float]]]] = []
    for stroke in geometry.strokes:
        grown = axes_reveal if stroke.is_axis else line_growth
        if grown <= 0.0:
            continue
        pixels, depth = camera.project(stroke.samples_scene)
        shown = (depth > near) & (stroke.growth <= grown)
        polylines.extend(
            (
                stroke,
                [((x + 0.5) * scale, (y + 0.5) * scale) for x, y in pixels[start : stop + 1]],
            )
            for start, stop in _runs(shown)
        )
    if style.outline_px > 0:
        for _, points in polylines:
            outline_draw.line(
                points, fill=round(255 * _OUTLINE_OPACITY), width=outline_width, joint="curve"
            )
    for stroke, points in polylines:
        core_draw.line(
            points,
            fill=tuple(round(255 * channel) for channel in stroke.rgb),
            width=core_width,
            joint="curve",
        )
        coverage_draw.line(points, fill=255, width=core_width, joint="curve")

    if axes_reveal > 0.0:
        font = ImageFont.load_default(size=style.label_font_px * scale)
        value = round(255 * axes_reveal)
        stroke_width = max(1, style.outline_px) * scale
        for text, anchor_scene in geometry.labels:
            pixels, depth = camera.project(anchor_scene[None])
            x, y = pixels[0]
            if depth[0] <= near or not (0 <= x < camera.width and 0 <= y < camera.height):
                continue
            position = ((x + 0.5) * scale, (y + 0.5) * scale)
            core_draw.text(
                position,
                text,
                font=font,
                anchor="mb",
                fill=(value, value, value),
                stroke_width=stroke_width,
                stroke_fill=(0, 0, 0),
            )
            coverage_draw.text(
                position,
                text,
                font=font,
                anchor="mb",
                fill=value,
                stroke_width=stroke_width,
                stroke_fill=value,
            )

    shape = (camera.width, camera.height)
    box = Image.Resampling.BOX
    shadow = np.asarray(outline.resize(shape, box), dtype=np.float32)[..., None] / 255.0
    rgb = np.asarray(core.resize(shape, box), dtype=np.float32) / 255.0
    alpha = np.asarray(coverage.resize(shape, box), dtype=np.float32)[..., None] / 255.0
    # Black outline under the coloured core: premultiplied colour is the core's.
    return rgb, alpha + shadow * (1.0 - alpha)
