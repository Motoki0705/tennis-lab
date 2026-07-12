"""Tests for src/tennis_scene/clip_studio/render.py (layout + drawing smoke)."""

import numpy as np
import pytest

from src.tennis_scene.clip_studio.render import (
    KEY_GUIDE_LINES,
    compute_layout,
    render_studio,
)
from src.tennis_scene.clip_studio.state import ClipStudioState


class TestComputeLayout:
    def test_two_cameras_side_by_side(self, studio_state: ClipStudioState) -> None:
        layout = compute_layout(studio_state, canvas_width=800)
        assert (layout.grid_cols, layout.grid_rows) == (2, 1)
        x0, y0, w0, _ = layout.tile_rect(0)
        x1, _, _, _ = layout.tile_rect(1)
        assert x1 > x0 + w0 - 1
        assert layout.canvas_height > layout.timeline.y

    def test_tile_index_at(self, studio_state: ClipStudioState) -> None:
        layout = compute_layout(studio_state, canvas_width=800)
        x, y, w, h = layout.tile_rect(1)
        assert layout.tile_index_at(x + w // 2, y + h // 2) == 1
        assert layout.tile_index_at(0, layout.canvas_height - 1) is None

    def test_clip_row_hit(self, studio_state: ClipStudioState) -> None:
        layout = compute_layout(studio_state, canvas_width=800)
        rx, ry, rw, rh = layout.clip_row_rect()
        assert layout.in_clip_row(rx + rw // 2, ry + rh // 2)
        assert not layout.in_clip_row(rx + rw // 2, ry - 1)

    def test_timeline_maps_view_window(self, studio_state: ClipStudioState) -> None:
        studio_state.zoom_view(0.5, anchor_sec=5.0)
        layout = compute_layout(studio_state, canvas_width=800)
        assert layout.timeline.view_start_sec == studio_state.view_start_sec
        assert layout.timeline.view_end_sec == studio_state.view_end_sec

    def test_too_narrow_canvas_raises(self, studio_state: ClipStudioState) -> None:
        with pytest.raises(ValueError, match="canvas_width"):
            compute_layout(studio_state, canvas_width=100)


class TestRenderStudio:
    def test_renders_canvas_with_frames(self, studio_state: ClipStudioState) -> None:
        frames: list[np.ndarray] = [
            np.full((36, 64, 3), 120, dtype=np.uint8),
            np.full((36, 64, 3), 90, dtype=np.uint8),
        ]
        canvas = render_studio(studio_state, frames, canvas_width=800)
        layout = compute_layout(studio_state, canvas_width=800)
        assert canvas.shape == (layout.canvas_height, layout.canvas_width, 3)
        assert canvas.dtype == np.uint8
        # tile content is visible (not all background)
        x, y, w, h = layout.tile_rect(0)
        assert int(canvas[y : y + h, x : x + w].max()) >= 120

    def test_renders_missing_frames_as_placeholder(
        self, studio_state: ClipStudioState
    ) -> None:
        canvas = render_studio(studio_state, [None, None], canvas_width=800)
        assert canvas.dtype == np.uint8

    def test_renders_persistent_key_guide(
        self, studio_state: ClipStudioState
    ) -> None:
        canvas = render_studio(studio_state, [None, None], canvas_width=800)
        layout = compute_layout(studio_state, canvas_width=800)
        guide = canvas[layout.footer_y + 36 :, :]

        assert len(KEY_GUIDE_LINES) == 3
        assert int(guide.max()) > 24

    def test_renders_marks_clips_and_help(self, studio_state: ClipStudioState) -> None:
        studio_state.seek(2.5)
        studio_state.set_mark_in()
        studio_state.seek(3.5)
        studio_state.set_mark_out()
        studio_state.selected_clip = 0
        canvas = render_studio(
            studio_state, [None, None], canvas_width=800, status="msg", show_help=True
        )
        assert canvas.dtype == np.uint8

    def test_frames_length_mismatch_raises(self, studio_state: ClipStudioState) -> None:
        with pytest.raises(ValueError, match="must match cameras"):
            render_studio(studio_state, [None], canvas_width=800)
