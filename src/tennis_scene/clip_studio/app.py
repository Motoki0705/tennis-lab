"""Interactive cv2 application for multi-camera sync and clip editing.

Wires keyboard/mouse events to :class:`ClipStudioState`, fetches preview
frames through :class:`PreviewSourcePool`, and redraws via
:func:`render_studio` only when something visible changed (state or frames),
so the app idles cheaply on long videos. Heavy operations (audio sync,
export) run synchronously with a status line update; the project JSON is
saved explicitly with ``s`` and always autosaved on quit.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.clip_studio.audio_sync import estimate_audio_offsets
from src.tennis_scene.clip_studio.export import ExportSettings, export_clips
from src.tennis_scene.clip_studio.project import ClipStudioProject
from src.tennis_scene.clip_studio.render import (
    StudioLayout,
    compute_layout,
    render_studio,
)
from src.tennis_scene.clip_studio.sources import PreviewSource, PreviewSourcePool
from src.tennis_scene.clip_studio.state import ClipStudioState
from src.utils.configuration import PathResolver

LOGGER = logging.getLogger(__name__)

KEY_ESC = 27
KEY_TAB = 9
KEY_LEFT = 65361
KEY_UP = 65362
KEY_RIGHT = 65363
KEY_DOWN = 65364
KEY_HOME = 65360
KEY_END = 65367

STATUS_DURATION_SEC = 5.0


@dataclass
class ClipStudioAppConfig:
    """Runtime configuration of the GUI application."""

    project_path: Path
    resolver: PathResolver
    export: ExportSettings
    canvas_width: int
    tile_width: int
    cache_frames: int
    seek_grab_threshold: int
    window_name: str
    audio_sample_rate: int
    audio_envelope_rate: float
    audio_max_seconds: float | None
    zoom_step: float


class ClipStudioApp:
    """Own the cv2 window, preview sources and the editing state."""

    def __init__(self, config: ClipStudioAppConfig, project: ClipStudioProject) -> None:
        self.config = config
        self.pool = PreviewSourcePool(
            [
                PreviewSource(
                    source.path,
                    tile_width=config.tile_width,
                    cache_frames=config.cache_frames,
                    seek_grab_threshold=config.seek_grab_threshold,
                )
                for source in project.sources
            ]
        )
        self.state = ClipStudioState(project, self.pool.infos)
        self._status = ""
        self._status_until = 0.0
        self._show_help = False
        self._dragging = False
        self._layout: StudioLayout = compute_layout(self.state, config.canvas_width)
        self._last_render_key: tuple[Any, ...] | None = None

    # ------------------------------------------------------------------ loop
    def run(self) -> None:
        """Run the event loop until the user quits; autosaves on exit."""
        cv2.namedWindow(self.config.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.config.window_name, self._on_mouse)
        last_tick = time.monotonic()
        try:
            while True:
                now = time.monotonic()
                if self.state.playing:
                    self.state.advance(now - last_tick)
                last_tick = now
                self._render_if_needed()
                key = cv2.waitKeyEx(15)
                if key != -1 and not self._handle_key(key):
                    break
        finally:
            self._save_project(reason="autosave on quit")
            self.pool.close()
            cv2.destroyWindow(self.config.window_name)

    def _current_status(self) -> str:
        if self._status and time.monotonic() < self._status_until:
            return self._status
        return ""

    def _set_status(self, message: str) -> None:
        LOGGER.info(message)
        self._status = message
        self._status_until = time.monotonic() + STATUS_DURATION_SEC

    def _render_if_needed(self, *, force: bool = False) -> None:
        state = self.state
        frame_indices = state.frame_indices_at(state.playhead_sec)
        render_key: tuple[Any, ...] = (
            tuple(frame_indices),
            round(state.playhead_sec, 4),
            state.playing,
            state.selected_camera,
            state.selected_clip,
            state.mark_in_sec,
            state.mark_out_sec,
            tuple(source.offset_sec for source in state.project.sources),
            tuple(
                (clip.name, clip.start_sec, clip.end_sec)
                for clip in state.project.clips
            ),
            state.dirty,
            round(state.view_start_sec, 4),
            round(state.view_end_sec, 4),
            self._current_status(),
            self._show_help,
        )
        if not force and render_key == self._last_render_key:
            return
        frames: list[NDArray[np.uint8] | None] = self.pool.fetch(frame_indices)
        self._layout = compute_layout(state, self.config.canvas_width)
        canvas = render_studio(
            state,
            frames,
            canvas_width=self.config.canvas_width,
            status=self._current_status(),
            show_help=self._show_help,
        )
        cv2.imshow(self.config.window_name, canvas)
        self._last_render_key = render_key

    def _render_now(self, message: str) -> None:
        """Show a status immediately before a blocking operation."""
        self._set_status(message)
        self._render_if_needed(force=True)
        cv2.waitKey(1)

    # ------------------------------------------------------------------ keys
    def _handle_key(self, key: int) -> bool:
        """Dispatch one key event; returns False to quit."""
        state = self.state
        try:
            if key in {ord("q"), KEY_ESC}:
                return False
            elif key == ord(" "):
                state.toggle_play()
            elif key in {ord(","), KEY_LEFT}:
                state.step_frames(-1)
            elif key in {ord("."), KEY_RIGHT}:
                state.step_frames(1)
            elif key == ord("<"):
                state.seek(state.playhead_sec - 1.0)
            elif key == ord(">"):
                state.seek(state.playhead_sec + 1.0)
            elif key == KEY_HOME:
                state.seek(state.extent_sec()[0])
            elif key == KEY_END:
                state.seek(state.extent_sec()[1])
            elif ord("1") <= key <= ord("9"):
                camera_index = key - ord("1")
                if camera_index < state.num_cameras:
                    state.select_camera(camera_index)
            elif key in {KEY_UP, KEY_DOWN}:
                step = -1 if key == KEY_UP else 1
                state.select_camera((state.selected_camera + step) % state.num_cameras)
            elif key in {ord("["), ord("]"), ord("{"), ord("}")}:
                frame_sec = 1.0 / state.reference_fps()
                delta = {
                    ord("["): -frame_sec,
                    ord("]"): frame_sec,
                    ord("{"): -10 * frame_sec,
                    ord("}"): 10 * frame_sec,
                }[key]
                offset = state.nudge_selected_offset(delta)
                camera_id = state.project.sources[state.selected_camera].camera_id
                self._set_status(f"{camera_id} offset -> {offset:+.3f}s")
            elif key == ord("a"):
                self._run_audio_sync()
            elif key == ord("i"):
                state.set_mark_in()
            elif key == ord("o"):
                state.set_mark_out()
            elif key == ord("c"):
                clip = state.make_clip_from_marks()
                self._set_status(
                    f"created {clip.name} [{clip.start_sec:.3f}s, {clip.end_sec:.3f}s)"
                )
            elif key == KEY_TAB:
                state.cycle_selected_clip(1)
            elif key == ord("x"):
                clip = state.delete_selected_clip()
                self._set_status(f"deleted {clip.name}")
            elif key == ord("z"):
                state.zoom_view(1.0 / self.config.zoom_step)
            elif key == ord("Z"):
                state.zoom_view(self.config.zoom_step)
            elif key == ord("f"):
                state.fit_view()
            elif key == ord("s"):
                self._save_project(reason="saved")
            elif key == ord("e"):
                self._export(all_clips=False)
            elif key == ord("E"):
                self._export(all_clips=True)
            elif key == ord("h"):
                self._show_help = not self._show_help
        except (ValueError, RuntimeError) as error:
            self._set_status(f"error: {error}")
        return True

    # ----------------------------------------------------------------- mouse
    def _on_mouse(self, event: int, x: int, y: int, flags: int, _param: object) -> None:
        layout = self._layout
        timeline = layout.timeline
        if event == cv2.EVENT_LBUTTONDOWN:
            if layout.in_clip_row(x, y):
                selected = self.state.select_clip_at(timeline.x_to_sec(x))
                if selected is not None:
                    self._set_status(
                        f"selected {self.state.project.clips[selected].name}"
                    )
            elif timeline.contains(x, y):
                self._dragging = True
                self.state.seek(timeline.x_to_sec(x))
            else:
                camera_index = layout.tile_index_at(x, y)
                if camera_index is not None and camera_index < self.state.num_cameras:
                    self.state.select_camera(camera_index)
        elif event == cv2.EVENT_MOUSEMOVE and self._dragging:
            self.state.seek(timeline.x_to_sec(x))
        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging = False
        elif event == cv2.EVENT_MOUSEWHEEL and timeline.contains(x, y):
            factor = 1.0 / self.config.zoom_step if flags > 0 else self.config.zoom_step
            self.state.zoom_view(factor, anchor_sec=timeline.x_to_sec(x))

    # ------------------------------------------------------------ operations
    def _save_project(self, *, reason: str) -> None:
        self.state.project.save(self.config.project_path, self.config.resolver)
        self.state.mark_saved()
        self._set_status(f"{reason}: {self.config.project_path}")

    def _run_audio_sync(self) -> None:
        state = self.state
        reference = state.selected_camera
        self._render_now(
            f"audio sync: decoding audio (reference {state.project.sources[reference].camera_id}) ..."
        )
        result = estimate_audio_offsets(
            [source.path for source in state.project.sources],
            reference_index=reference,
            reference_offset_sec=state.project.sources[reference].offset_sec,
            sample_rate=self.config.audio_sample_rate,
            envelope_rate=self.config.audio_envelope_rate,
            max_seconds=self.config.audio_max_seconds,
        )
        state.apply_offsets(result.offsets_sec)
        confidences = ", ".join(
            f"{source.camera_id}={confidence:.2f}"
            for source, confidence in zip(
                state.project.sources, result.confidences, strict=True
            )
        )
        self._set_status(f"audio sync applied (confidence: {confidences})")

    def _export(self, *, all_clips: bool) -> None:
        state = self.state
        state.playing = False
        if all_clips:
            clip_names = None
        else:
            if state.selected_clip is None:
                raise ValueError("no clip selected (Tab or click a clip first)")
            clip_names = [state.project.clips[state.selected_clip].name]
        label = "all clips" if clip_names is None else clip_names[0]
        self._render_now(f"exporting {label} ...")
        results = export_clips(
            state.project,
            self.config.export,
            infos=self.pool.infos,
            clip_names=clip_names,
        )
        self._set_status(
            f"exported {len(results)} clip(s) to {self.config.export.output_dir}"
        )


__all__ = ["ClipStudioApp", "ClipStudioAppConfig"]
