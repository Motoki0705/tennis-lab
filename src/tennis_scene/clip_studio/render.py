"""Canvas rendering for the clip studio GUI.

Pure drawing: takes the editing state plus already-decoded preview frames and
returns a BGR canvas. The layout is exposed separately
(:func:`compute_layout`) so the event layer can hit-test mouse positions
against the exact same geometry that was rendered.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.clip_studio.imaging import letterbox_frame
from src.tennis_scene.clip_studio.state import ClipStudioState
from src.tennis_scene.clip_studio.timeline import TimelineGeometry, format_timecode

MARGIN = 8
TILE_GAP = 6
TILE_HEADER_HEIGHT = 20
RULER_HEIGHT = 18
CLIP_ROW_HEIGHT = 22
CAMERA_ROW_HEIGHT = 9
FOOTER_HEIGHT = 40

COLOR_BACKGROUND = (24, 24, 24)
COLOR_TILE_BG = (12, 12, 12)
COLOR_BORDER = (70, 70, 70)
COLOR_BORDER_SELECTED = (0, 170, 255)
COLOR_TEXT = (215, 215, 215)
COLOR_TEXT_DIM = (140, 140, 140)
COLOR_COVERAGE = (95, 95, 95)
COLOR_COVERAGE_SELECTED = (0, 130, 190)
COLOR_CLIP = (190, 120, 50)
COLOR_CLIP_SELECTED = (255, 190, 90)
COLOR_MARK_IN = (90, 200, 90)
COLOR_MARK_OUT = (70, 70, 220)
COLOR_PLAYHEAD = (240, 240, 240)

_TICK_STEPS_SEC = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0)

HELP_LINES = (
    "space play/pause    ,/. step 1 frame    </> jump 1s    Home/End extent edges",
    "1-9 select camera   [/] offset -/+1 frame   {/} offset -/+10 frames   a audio auto-sync",
    "i/o set in/out mark   c create clip   Tab cycle clips   x delete selected clip",
    "z/Z zoom in/out   f fit timeline   drag timeline: scrub   click clip row: select",
    "s save project   e export selected clip   E export all clips   h toggle help   q quit",
)


@dataclass(frozen=True)
class StudioLayout:
    """Pixel layout of the studio canvas for a given state/canvas width."""

    canvas_width: int
    canvas_height: int
    grid_cols: int
    grid_rows: int
    tile_width: int
    tile_height: int
    timeline: TimelineGeometry
    footer_y: int

    def tile_rect(self, camera_index: int) -> tuple[int, int, int, int]:
        """(x, y, w, h) of a camera tile."""
        col = camera_index % self.grid_cols
        row = camera_index // self.grid_cols
        x = MARGIN + col * (self.tile_width + TILE_GAP)
        y = MARGIN + row * (self.tile_height + TILE_GAP)
        return (x, y, self.tile_width, self.tile_height)

    def tile_index_at(self, x: int, y: int) -> int | None:
        """Camera index under a canvas pixel, or None."""
        for camera_index in range(self.grid_cols * self.grid_rows):
            tx, ty, tw, th = self.tile_rect(camera_index)
            if tx <= x < tx + tw and ty <= y < ty + th:
                return camera_index
        return None

    def clip_row_rect(self) -> tuple[int, int, int, int]:
        timeline = self.timeline
        return (timeline.x, timeline.y + RULER_HEIGHT, timeline.width, CLIP_ROW_HEIGHT)

    def in_clip_row(self, x: int, y: int) -> bool:
        rx, ry, rw, rh = self.clip_row_rect()
        return rx <= x < rx + rw and ry <= y < ry + rh


def compute_layout(state: ClipStudioState, canvas_width: int = 1600) -> StudioLayout:
    """Deterministic layout for ``state`` at the given canvas width."""
    if canvas_width < 320:
        raise ValueError(f"canvas_width must be >= 320, got {canvas_width}")
    num_cameras = state.num_cameras
    grid_cols = max(1, math.ceil(math.sqrt(num_cameras)))
    grid_rows = math.ceil(num_cameras / grid_cols)

    tile_width = (canvas_width - 2 * MARGIN - (grid_cols - 1) * TILE_GAP) // grid_cols
    max_aspect = max(info.height / info.width for info in state.infos)
    tile_height = TILE_HEADER_HEIGHT + round(tile_width * max_aspect)

    timeline_height = RULER_HEIGHT + CLIP_ROW_HEIGHT + num_cameras * CAMERA_ROW_HEIGHT + 6
    timeline_y = MARGIN + grid_rows * tile_height + (grid_rows - 1) * TILE_GAP + TILE_GAP
    timeline = TimelineGeometry(
        x=MARGIN,
        y=timeline_y,
        width=canvas_width - 2 * MARGIN,
        height=timeline_height,
        view_start_sec=state.view_start_sec,
        view_end_sec=state.view_end_sec,
    )
    footer_y = timeline_y + timeline_height + 4
    canvas_height = footer_y + FOOTER_HEIGHT
    return StudioLayout(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        tile_width=tile_width,
        tile_height=tile_height,
        timeline=timeline,
        footer_y=footer_y,
    )


def render_studio(
    state: ClipStudioState,
    frames: Sequence[NDArray[np.uint8] | None],
    *,
    canvas_width: int = 1600,
    status: str = "",
    show_help: bool = False,
) -> NDArray[np.uint8]:
    """Render the full studio canvas (tiles + timeline + footer)."""
    if len(frames) != state.num_cameras:
        raise ValueError(
            f"frames length {len(frames)} must match cameras {state.num_cameras}"
        )
    layout = compute_layout(state, canvas_width)
    canvas: NDArray[np.uint8] = np.full(
        (layout.canvas_height, layout.canvas_width, 3),
        COLOR_BACKGROUND,
        dtype=np.uint8,
    )
    frame_indices = state.frame_indices_at(state.playhead_sec)
    for camera_index, frame in enumerate(frames):
        _draw_tile(canvas, state, layout, camera_index, frame, frame_indices[camera_index])
    _draw_timeline(canvas, state, layout)
    _draw_footer(canvas, state, layout, status)
    if show_help:
        _draw_help_overlay(canvas, layout)
    return canvas


def _put_text(
    canvas: NDArray[np.uint8],
    text: str,
    org: tuple[int, int],
    *,
    scale: float = 0.42,
    color: tuple[int, int, int] = COLOR_TEXT,
) -> None:
    cv2.putText(
        canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA
    )


def _draw_tile(
    canvas: NDArray[np.uint8],
    state: ClipStudioState,
    layout: StudioLayout,
    camera_index: int,
    frame: NDArray[np.uint8] | None,
    frame_index: int | None,
) -> None:
    x, y, width, height = layout.tile_rect(camera_index)
    view = state.source_views()[camera_index]
    image_height = height - TILE_HEADER_HEIGHT
    image_top = y + TILE_HEADER_HEIGHT

    canvas[y : y + height, x : x + width] = COLOR_TILE_BG
    if frame is not None:
        fitted, _ = letterbox_frame(frame, width, image_height)
        canvas[image_top : image_top + image_height, x : x + width] = fitted
    else:
        _put_text(
            canvas,
            "no coverage at playhead",
            (x + 10, image_top + image_height // 2),
            color=COLOR_TEXT_DIM,
        )

    if frame_index is not None:
        local_sec = frame_index / view.info.fps
        position = f"f {frame_index}/{view.info.frame_count}  {format_timecode(local_sec)}"
    else:
        position = "out of range"
    header = (
        f"[{camera_index + 1}] {view.camera_id}  "
        f"off {view.offset_sec:+.3f}s  {position}"
    )
    _put_text(canvas, header, (x + 6, y + 14))

    selected = camera_index == state.selected_camera
    border = COLOR_BORDER_SELECTED if selected else COLOR_BORDER
    cv2.rectangle(canvas, (x, y), (x + width - 1, y + height - 1), border, 1 if not selected else 2)


def _select_tick_step(view_span_sec: float, width_px: int) -> float:
    px_per_sec = width_px / view_span_sec
    for step in _TICK_STEPS_SEC:
        if step * px_per_sec >= 70:
            return step
    return _TICK_STEPS_SEC[-1]


def _draw_timeline(
    canvas: NDArray[np.uint8], state: ClipStudioState, layout: StudioLayout
) -> None:
    timeline = layout.timeline
    x0, y0 = timeline.x, timeline.y
    x1 = timeline.x + timeline.width
    canvas[y0 : y0 + timeline.height, x0:x1] = (34, 34, 34)

    # Ruler ticks and labels.
    step = _select_tick_step(timeline.view_span_sec, timeline.width)
    first_tick = math.ceil(timeline.view_start_sec / step) * step
    tick = first_tick
    while tick <= timeline.view_end_sec:
        tick_x = timeline.sec_to_x(tick)
        cv2.line(canvas, (tick_x, y0 + 2), (tick_x, y0 + RULER_HEIGHT - 4), COLOR_TEXT_DIM, 1)
        _put_text(
            canvas,
            format_timecode(tick),
            (tick_x + 3, y0 + RULER_HEIGHT - 5),
            scale=0.34,
            color=COLOR_TEXT_DIM,
        )
        tick += step

    # Clip row.
    clip_x, clip_y, clip_w, clip_h = layout.clip_row_rect()
    canvas[clip_y : clip_y + clip_h, clip_x : clip_x + clip_w] = (42, 42, 42)
    for clip_index, clip in enumerate(state.project.clips):
        if clip.end_sec < timeline.view_start_sec or clip.start_sec > timeline.view_end_sec:
            continue
        left = max(timeline.sec_to_x(clip.start_sec), clip_x)
        right = min(timeline.sec_to_x(clip.end_sec), clip_x + clip_w - 1)
        if right <= left:
            continue
        selected = clip_index == state.selected_clip
        color = COLOR_CLIP_SELECTED if selected else COLOR_CLIP
        cv2.rectangle(canvas, (left, clip_y + 2), (right, clip_y + clip_h - 3), color, -1)
        if right - left > 44:
            _put_text(
                canvas, clip.name, (left + 4, clip_y + clip_h - 8),
                scale=0.36, color=(20, 20, 20),
            )

    # Per-camera coverage rows.
    rows_top = clip_y + clip_h + 2
    for camera_index, view in enumerate(state.source_views()):
        row_y = rows_top + camera_index * CAMERA_ROW_HEIGHT
        start_sec, end_sec = view.coverage_sec
        left = max(timeline.sec_to_x(start_sec), x0)
        right = min(timeline.sec_to_x(end_sec), x1 - 1)
        selected = camera_index == state.selected_camera
        color = COLOR_COVERAGE_SELECTED if selected else COLOR_COVERAGE
        if right > left:
            cv2.rectangle(
                canvas, (left, row_y + 1), (right, row_y + CAMERA_ROW_HEIGHT - 2), color, -1
            )

    # In/out marks.
    strip_bottom = y0 + timeline.height - 2
    for mark_sec, color in (
        (state.mark_in_sec, COLOR_MARK_IN),
        (state.mark_out_sec, COLOR_MARK_OUT),
    ):
        if mark_sec is None:
            continue
        if not timeline.view_start_sec <= mark_sec <= timeline.view_end_sec:
            continue
        mark_x = timeline.sec_to_x(mark_sec)
        cv2.line(canvas, (mark_x, clip_y), (mark_x, strip_bottom), color, 1)

    # Playhead.
    if timeline.view_start_sec <= state.playhead_sec <= timeline.view_end_sec:
        playhead_x = timeline.sec_to_x(state.playhead_sec)
        cv2.line(canvas, (playhead_x, y0), (playhead_x, strip_bottom), COLOR_PLAYHEAD, 1)
        triangle = np.array(
            [
                [playhead_x - 4, y0],
                [playhead_x + 4, y0],
                [playhead_x, y0 + 6],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(canvas, [triangle], COLOR_PLAYHEAD)


def _draw_footer(
    canvas: NDArray[np.uint8],
    state: ClipStudioState,
    layout: StudioLayout,
    status: str,
) -> None:
    line1_y = layout.footer_y + 14
    line2_y = layout.footer_y + 32
    play_flag = "PLAY" if state.playing else "PAUSE"
    dirty_flag = "*unsaved" if state.dirty else "saved"
    marks = (
        f"in {format_timecode(state.mark_in_sec) if state.mark_in_sec is not None else '-'}"
        f"  out {format_timecode(state.mark_out_sec) if state.mark_out_sec is not None else '-'}"
    )
    summary = (
        f"{play_flag}  t {format_timecode(state.playhead_sec)}  {marks}  "
        f"clips {len(state.project.clips)}  [{dirty_flag}]"
    )
    _put_text(canvas, summary, (MARGIN, line1_y))
    footer = status if status else "h: help"
    _put_text(canvas, footer, (MARGIN, line2_y), color=COLOR_TEXT_DIM)


def _draw_help_overlay(canvas: NDArray[np.uint8], layout: StudioLayout) -> None:
    box_width = min(760, layout.canvas_width - 2 * MARGIN)
    box_height = 20 + 20 * len(HELP_LINES)
    x0, y0 = MARGIN + 12, MARGIN + 12
    overlay = canvas[y0 : y0 + box_height, x0 : x0 + box_width].astype(np.float32)
    overlay = overlay * 0.25 + np.array((16.0, 16.0, 16.0), dtype=np.float32) * 0.75
    canvas[y0 : y0 + box_height, x0 : x0 + box_width] = overlay.astype(np.uint8)
    cv2.rectangle(
        canvas, (x0, y0), (x0 + box_width - 1, y0 + box_height - 1), COLOR_BORDER, 1
    )
    for line_index, line in enumerate(HELP_LINES):
        _put_text(canvas, line, (x0 + 10, y0 + 24 + 20 * line_index), scale=0.4)


__all__ = ["StudioLayout", "compute_layout", "render_studio", "HELP_LINES"]
