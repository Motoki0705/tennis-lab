"""Editing state machine for the clip studio GUI.

Pure logic (no cv2 / no decoding) so every editing operation is unit-testable.
The GUI layer translates key/mouse events into calls on
:class:`ClipStudioState` and renders the result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.tennis_scene.clip_studio.project import Clip, ClipStudioProject
from src.tennis_scene.clip_studio.timeline import (
    source_coverage_sec,
    source_frame_index,
    timeline_extent_sec,
)
from src.utils.video import VideoInfo


@dataclass(frozen=True)
class SourceView:
    """A project source joined with its probed video metadata."""

    camera_id: str
    offset_sec: float
    info: VideoInfo

    @property
    def duration_sec(self) -> float:
        return float(self.info.frame_count) / float(self.info.fps)

    @property
    def coverage_sec(self) -> tuple[float, float]:
        return source_coverage_sec(
            offset_sec=self.offset_sec, duration_sec=self.duration_sec
        )


class ClipStudioState:
    """Mutable editing state over a validated project.

    Invariants:
        - ``playhead_sec`` stays inside the global extent.
        - The view window stays inside the global extent and never collapses
          below ``MIN_VIEW_SPAN_SEC``.
        - ``dirty`` is set by every project mutation and cleared by
          :meth:`mark_saved`.
    """

    MIN_VIEW_SPAN_SEC = 0.25

    def __init__(
        self, project: ClipStudioProject, infos: Sequence[VideoInfo]
    ) -> None:
        errors = project.validate()
        if errors:
            raise ValueError(f"Invalid project: {errors}")
        if len(infos) != len(project.sources):
            raise ValueError(
                f"infos length {len(infos)} must match sources {len(project.sources)}"
            )
        for index, info in enumerate(infos):
            if info.fps <= 0 or info.frame_count <= 0:
                raise ValueError(
                    f"source {index} has invalid metadata: fps={info.fps}, "
                    f"frame_count={info.frame_count}"
                )
        self.project = project
        self.infos = list(infos)

        extent = self.extent_sec()
        self.playhead_sec: float = extent[0]
        self.view_start_sec: float = extent[0]
        self.view_end_sec: float = extent[1]
        self.selected_camera: int = 0
        self.selected_clip: int | None = None
        self.mark_in_sec: float | None = None
        self.mark_out_sec: float | None = None
        self.playing: bool = False
        self.dirty: bool = False

    # ------------------------------------------------------------------ views
    @property
    def num_cameras(self) -> int:
        return len(self.project.sources)

    def source_views(self) -> list[SourceView]:
        return [
            SourceView(
                camera_id=source.camera_id,
                offset_sec=source.offset_sec,
                info=info,
            )
            for source, info in zip(self.project.sources, self.infos, strict=True)
        ]

    def extent_sec(self) -> tuple[float, float]:
        return timeline_extent_sec(view.coverage_sec for view in self.source_views())

    def reference_fps(self) -> float:
        """FPS used for frame stepping: the selected camera's fps."""
        return float(self.infos[self.selected_camera].fps)

    def frame_indices_at(self, global_sec: float) -> list[int | None]:
        """Nearest source frame index per camera (None outside coverage)."""
        return [
            source_frame_index(
                global_sec,
                offset_sec=source.offset_sec,
                fps=info.fps,
                frame_count=info.frame_count,
            )
            for source, info in zip(self.project.sources, self.infos, strict=True)
        ]

    # --------------------------------------------------------------- playback
    def seek(self, global_sec: float) -> None:
        start, end = self.extent_sec()
        self.playhead_sec = min(max(global_sec, start), end)

    def step_frames(self, frames: int) -> None:
        self.seek(self.playhead_sec + frames / self.reference_fps())

    def toggle_play(self) -> None:
        self.playing = not self.playing

    def advance(self, delta_sec: float) -> None:
        """Advance the playhead during playback; stops at the extent end."""
        if delta_sec < 0:
            raise ValueError(f"delta_sec must be non-negative, got {delta_sec}")
        end = self.extent_sec()[1]
        target = self.playhead_sec + delta_sec
        if target >= end:
            self.playhead_sec = end
            self.playing = False
        else:
            self.playhead_sec = target

    # ------------------------------------------------------------------- sync
    def select_camera(self, camera_index: int) -> None:
        if not 0 <= camera_index < self.num_cameras:
            raise ValueError(
                f"camera_index {camera_index} out of range [0, {self.num_cameras})"
            )
        self.selected_camera = camera_index

    def nudge_selected_offset(self, delta_sec: float) -> float:
        """Shift the selected camera's sync offset; returns the new offset."""
        source = self.project.sources[self.selected_camera]
        source.offset_sec += delta_sec
        self.dirty = True
        return source.offset_sec

    def apply_offsets(self, offsets_sec: Sequence[float]) -> None:
        """Replace all sync offsets (e.g. from audio auto-sync)."""
        if len(offsets_sec) != self.num_cameras:
            raise ValueError(
                f"offsets length {len(offsets_sec)} must match cameras "
                f"{self.num_cameras}"
            )
        for source, offset in zip(self.project.sources, offsets_sec, strict=True):
            source.offset_sec = float(offset)
        self.dirty = True
        # Offsets move the extent; keep playhead and view legal.
        self.seek(self.playhead_sec)
        self._clamp_view()

    # ------------------------------------------------------------------ clips
    def set_mark_in(self) -> None:
        self.mark_in_sec = self.playhead_sec

    def set_mark_out(self) -> None:
        self.mark_out_sec = self.playhead_sec

    def clear_marks(self) -> None:
        self.mark_in_sec = None
        self.mark_out_sec = None

    def make_clip_from_marks(self) -> Clip:
        """Create a clip from the in/out marks and select it."""
        if self.mark_in_sec is None or self.mark_out_sec is None:
            raise ValueError("both in (i) and out (o) marks are required")
        if self.mark_out_sec <= self.mark_in_sec:
            raise ValueError(
                f"mark out ({self.mark_out_sec:.3f}s) must be after "
                f"mark in ({self.mark_in_sec:.3f}s)"
            )
        clip = Clip(
            name=self.project.next_clip_name(),
            start_sec=self.mark_in_sec,
            end_sec=self.mark_out_sec,
        )
        self.project.clips.append(clip)
        self.selected_clip = len(self.project.clips) - 1
        self.dirty = True
        return clip

    def delete_selected_clip(self) -> Clip:
        if self.selected_clip is None:
            raise ValueError("no clip selected")
        clip = self.project.clips.pop(self.selected_clip)
        self.selected_clip = None
        self.dirty = True
        return clip

    def clip_index_at(self, global_sec: float) -> int | None:
        for index, clip in enumerate(self.project.clips):
            if clip.contains(global_sec):
                return index
        return None

    def select_clip_at(self, global_sec: float) -> int | None:
        self.selected_clip = self.clip_index_at(global_sec)
        return self.selected_clip

    def cycle_selected_clip(self, direction: int = 1) -> int | None:
        """Select the next/previous clip in start-time order."""
        if not self.project.clips:
            self.selected_clip = None
            return None
        order = sorted(
            range(len(self.project.clips)),
            key=lambda index: self.project.clips[index].start_sec,
        )
        if self.selected_clip is None:
            position = 0 if direction >= 0 else len(order) - 1
        else:
            position = (order.index(self.selected_clip) + direction) % len(order)
        self.selected_clip = order[position]
        return self.selected_clip

    # ------------------------------------------------------------------- view
    def zoom_view(self, factor: float, anchor_sec: float | None = None) -> None:
        """Scale the view span by ``factor`` around ``anchor_sec``."""
        if factor <= 0:
            raise ValueError(f"factor must be positive, got {factor}")
        start, end = self.extent_sec()
        full_span = end - start
        anchor = self.playhead_sec if anchor_sec is None else anchor_sec
        anchor = min(max(anchor, start), end)
        old_span = self.view_end_sec - self.view_start_sec
        new_span = min(max(old_span * factor, self.MIN_VIEW_SPAN_SEC), full_span)
        ratio = (
            (anchor - self.view_start_sec) / old_span if old_span > 0 else 0.5
        )
        self.view_start_sec = anchor - ratio * new_span
        self.view_end_sec = self.view_start_sec + new_span
        self._clamp_view()

    def pan_view(self, delta_sec: float) -> None:
        self.view_start_sec += delta_sec
        self.view_end_sec += delta_sec
        self._clamp_view()

    def fit_view(self) -> None:
        self.view_start_sec, self.view_end_sec = self.extent_sec()

    def _clamp_view(self) -> None:
        start, end = self.extent_sec()
        span = min(self.view_end_sec - self.view_start_sec, end - start)
        span = max(span, min(self.MIN_VIEW_SPAN_SEC, end - start))
        if self.view_start_sec < start:
            self.view_start_sec = start
        if self.view_start_sec + span > end:
            self.view_start_sec = end - span
        self.view_end_sec = self.view_start_sec + span

    # ---------------------------------------------------------------- persist
    def mark_saved(self) -> None:
        self.dirty = False


__all__ = ["ClipStudioState", "SourceView"]
