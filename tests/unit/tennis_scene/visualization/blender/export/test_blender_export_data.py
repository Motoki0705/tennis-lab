"""Tests for PC2 caches, bounded gap filling and bundle helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.tennis_scene.visualization.blender.export.bundle import court_geometry
from src.tennis_scene.visualization.blender.export.gaps import fill_short_gaps
from src.tennis_scene.visualization.blender.export.pointcache import read_pc2, write_pc2
from src.utils.schema.court import X_MAX, X_MIN, Y_MAX, Y_MIN, net_height_at_x


def test_pc2_round_trip_preserves_frames_and_header(tmp_path: Path) -> None:
    frames = np.random.default_rng(0).normal(size=(5, 7, 3)).astype(np.float32)
    path = tmp_path / "mesh.pc2"

    write_pc2(path, frames, start_frame=3.0)
    loaded, start, rate = read_pc2(path)

    np.testing.assert_array_equal(loaded, frames)
    assert (start, rate) == (3.0, 1.0)
    assert path.stat().st_size == 32 + frames.nbytes


def test_pc2_rejects_non_finite_vertices(tmp_path: Path) -> None:
    frames: np.ndarray = np.zeros((2, 3, 3), dtype=np.float32)
    frames[1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        write_pc2(tmp_path / "bad.pc2", frames, start_frame=0.0)


def test_fill_short_gaps_interpolates_only_bounded_short_runs() -> None:
    values = np.arange(10, dtype=np.float32)[:, None] * [1.0, 2.0]
    values[[0, 3, 4, 6, 7, 8, 9]] = -999.0  # garbage on invalid frames
    valid: np.ndarray = np.ones(10, dtype=bool)
    valid[[0, 3, 4, 6, 7, 8, 9]] = False
    valid[9] = True
    values[9] = [9.0, 18.0]

    result = fill_short_gaps(values, valid, max_gap_frames=2)

    # leading run (0) stays invalid; 3-4 (len 2) filled; 6-8 (len 3) too long.
    assert result.filled_frames == (3, 4)
    assert result.unfilled_frames == (0, 6, 7, 8)
    np.testing.assert_allclose(result.values[3:5], [[3.0, 6.0], [4.0, 8.0]])
    assert np.isnan(result.values[[0, 6, 7, 8]]).all()
    np.testing.assert_array_equal(result.valid, [False, 1, 1, 1, 1, 1, 0, 0, 0, 1])


def test_fill_short_gaps_zero_limit_disables_filling() -> None:
    values = np.array([[0.0], [0.0], [2.0]], dtype=np.float32)
    valid = np.array([True, False, True])

    result = fill_short_gaps(values, valid, max_gap_frames=0)

    assert result.filled_frames == ()
    assert result.unfilled_frames == (1,)


def test_fill_short_gaps_rejects_non_finite_valid_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        fill_short_gaps(
            np.array([[np.nan]], dtype=np.float32), np.array([True]), max_gap_frames=1
        )


def test_court_geometry_lists_ground_lines_inside_runoff() -> None:
    court = court_geometry()

    lines = np.asarray(court["ground_lines"])
    assert lines.shape == (13, 2, 3)
    assert (lines[..., 2] == 0.0).all()
    assert court["runoff_xy"] == [X_MIN, X_MAX, Y_MIN, Y_MAX]
    top = np.asarray(court["net"]["top"])
    assert top[0, 0] == -court["net"]["post_x"]
    assert top[:, 1].min() == pytest.approx(net_height_at_x(0.0))
