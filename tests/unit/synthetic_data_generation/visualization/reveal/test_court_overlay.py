from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.synthetic_data_generation.visualization.reveal.contracts import (
    RevealAlignmentStyle,
)
from src.synthetic_data_generation.visualization.reveal.court_overlay import (
    _runs,
    court_overlay_geometry,
    draw_court_overlay,
    overlay_reveal,
)
from src.synthetic_data_generation.visualization.reveal.viewpoint import (
    RevealCamera,
    look_at_camera_to_frame,
)
from src.utils.schema.court import COURT_SKELETON, HALF_DOUBLES_WIDTH, HALF_LENGTH

STYLE = RevealAlignmentStyle(
    line_rgb=(0.0, 1.0, 1.0),
    line_width_px=2,
    outline_px=1,
    axis_length_m=3.0,
    label_font_px=12,
)
GROUND_EDGES = sum(1 for a, b in COURT_SKELETON if a < 14 and b < 14)


def _transform(x: float) -> NDArray[np.float64]:
    matrix = np.eye(4)
    matrix[:3, 3] = (x, 0.0, 0.0)
    return matrix


def _overhead_camera(width: int = 160, height: int = 240) -> RevealCamera:
    """Look straight down at the court-frame origin from 40 m."""
    pose = look_at_camera_to_frame(
        np.asarray((0.0, 0.0, 40.0)), np.zeros(3), np.asarray((0.0, 1.0, 0.0))
    )
    focal = 360.0  # 9 px per metre at 40 m: the court fills the frame
    return RevealCamera(
        width=width,
        height=height,
        intrinsics=np.asarray(
            ((focal, 0.0, (width - 1) / 2), (0.0, focal, (height - 1) / 2), (0, 0, 1.0))
        ),
        camera_to_scene=pose,
    )


def test_geometry_places_ground_lines_and_axes_of_every_court() -> None:
    geometry = court_overlay_geometry(
        {"court-001": _transform(-15.0), "court-000": _transform(0.0)},
        line_rgb=(0.0, 1.0, 1.0),
        axis_length_m=3.0,
        units_per_metre=1.0,
    )

    assert [label for label, _ in geometry.labels] == ["court-000", "court-001"]
    lines = [stroke for stroke in geometry.strokes if not stroke.is_axis]
    axes = [stroke for stroke in geometry.strokes if stroke.is_axis]
    assert len(lines) == 2 * GROUND_EDGES and len(axes) == 6
    # Only ground lines: the net is not part of the fitted alignment.
    assert all(np.allclose(stroke.samples_scene[:, 2], 0.0) for stroke in lines)
    court_001 = np.concatenate([stroke.samples_scene for stroke in lines[GROUND_EDGES:]])
    assert court_001[:, 0].min() == pytest.approx(-15.0 - HALF_DOUBLES_WIDTH)
    assert court_001[:, 1].max() == pytest.approx(HALF_LENGTH)
    np.testing.assert_allclose(axes[2].samples_scene[-1], (0.0, 0.0, 3.0))
    np.testing.assert_allclose(geometry.labels[1][1][:2], (-15.0, 0.0))


def test_geometry_applies_the_similarity_scale() -> None:
    transform = 0.5 * np.eye(4)
    transform[3, 3] = 1.0
    geometry = court_overlay_geometry(
        {"court-000": transform}, line_rgb=(1, 1, 1), axis_length_m=2.0, units_per_metre=0.5
    )

    axis_x = next(stroke for stroke in geometry.strokes if stroke.is_axis)
    np.testing.assert_allclose(axis_x.samples_scene[-1], (1.0, 0.0, 0.0))


def test_overlay_reveal_grows_lines_then_axes() -> None:
    assert overlay_reveal(0.0) == (0.0, 0.0)
    lines, axes = overlay_reveal(0.35)
    assert lines == pytest.approx(0.5) and axes == 0.0
    assert overlay_reveal(0.7)[0] == 1.0
    assert overlay_reveal(1.0) == (1.0, 1.0)


def test_runs_split_on_hidden_samples() -> None:
    shown = np.asarray([True, True, True, False, True, False, True, True])

    assert _runs(shown) == [(0, 2), (6, 7)]
    assert _runs(np.zeros(3, dtype=bool)) == []


def test_drawn_court_covers_its_projected_outline() -> None:
    camera = _overhead_camera()
    geometry = court_overlay_geometry(
        {"court-000": np.eye(4)}, line_rgb=(0.0, 1.0, 1.0), axis_length_m=3.0, units_per_metre=1.0
    )

    rgb, alpha = draw_court_overlay(camera, geometry, 1.0, style=STYLE)
    _, nothing = draw_court_overlay(camera, geometry, 0.0, style=STYLE)

    assert nothing.max() == 0.0
    corner_px, _ = camera.project(np.asarray([[HALF_DOUBLES_WIDTH, HALF_LENGTH, 0.0]]))
    x, y = np.floor(corner_px[0]).astype(int)
    assert alpha[y - 1 : y + 2, x - 1 : x + 2].max() > 0.5
    # Pixels away from the court stay empty.
    assert alpha[5, 5, 0] == 0.0
    # A doubles-sideline pixel carries premultiplied cyan.
    side_px, _ = camera.project(np.asarray([[HALF_DOUBLES_WIDTH, HALF_LENGTH / 2, 0.0]]))
    x, y = np.floor(side_px[0]).astype(int)
    window = (slice(y - 1, y + 2), slice(x - 1, x + 2))
    peak = np.unravel_index(np.argmax(alpha[window][..., 0]), (3, 3))
    pixel = rgb[window][peak]
    assert pixel[0] == 0.0 and pixel[1] == pytest.approx(pixel[2]) and pixel[1] > 0.5
    assert np.all(rgb <= alpha + 1e-6)


def test_partial_growth_draws_less_than_the_full_court() -> None:
    camera = _overhead_camera()
    geometry = court_overlay_geometry(
        {"court-000": np.eye(4)}, line_rgb=(0.0, 1.0, 1.0), axis_length_m=3.0, units_per_metre=1.0
    )

    _, half = draw_court_overlay(camera, geometry, 0.35, style=STYLE)
    _, full = draw_court_overlay(camera, geometry, 0.7, style=STYLE)

    assert 0.0 < half.sum() < full.sum()


def test_strokes_behind_the_camera_are_clipped() -> None:
    # Camera at the net centre, 1 m up, looking along +y: the near half of the
    # court lies behind it and must not wrap into the image.
    pose = look_at_camera_to_frame(
        np.asarray((0.0, 0.0, 1.0)), np.asarray((0.0, 10.0, 0.0)), np.asarray((0.0, 0.0, 1.0))
    )
    camera = RevealCamera(
        width=64,
        height=48,
        intrinsics=np.asarray(((30.0, 0.0, 31.5), (0.0, 30.0, 23.5), (0.0, 0.0, 1.0))),
        camera_to_scene=pose,
    )
    geometry = court_overlay_geometry(
        {"court-000": np.eye(4)}, line_rgb=(1, 1, 1), axis_length_m=3.0, units_per_metre=1.0
    )

    _, alpha = draw_court_overlay(camera, geometry, 0.7, style=STYLE)

    # Ground lines ahead project below the horizon only (rows >= centre).
    assert alpha[:20].max() == 0.0
    assert alpha[24:].max() > 0.0
