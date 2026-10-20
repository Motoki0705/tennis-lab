"""Synchronized raw-proposal / Codex-annotation presentation, using CPU only."""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tasks.annotation.migration.legacy_format.contracts import BallFrameAnnotation
from src.tasks.base.visualization.presentation import style
from src.tasks.base.visualization.presentation.export import render_presentation
from src.utils.paths import PROJECT_ROOT

from .annotation_correction_config import AnnotationCorrectionConfig
from .annotation_correction_frames import open_frame_input
from .annotation_correction_inputs import (
    CHANGES,
    AnnotationComparison,
    Candidate,
    Correction,
    class_counts,
    load_comparison,
    prepared_clips,
    rank_clips,
)
from .annotation_correction_inputs import classify_frame as classify_frame

RGB = tuple[int, int, int]
Rect = tuple[int, int, int, int]
_COLORS = {
    Correction.UNREVIEWED: style.CORRECTION_UNREVIEWED,
    Correction.RERANK: style.CORRECTION_RERANK,
    Correction.MOVED: style.CORRECTION_MOVED,
    Correction.ADDED: style.CORRECTION_ADDED,
    Correction.REMOVED: style.CORRECTION_REMOVED,
}


def _rgb(value: str) -> RGB:
    return (int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16))


@dataclass(frozen=True)
class ComparisonLayout:
    width: int
    height: int
    left: Rect
    right: Rect
    strip: Rect


def comparison_layout(width: int, height: int) -> ComparisonLayout:
    if (
        width < 160
        or height < 90
        or width % 2
        or height % 2
        or width * 9 != height * 16
    ):
        raise ValueError("comparison resolution must be even 16:9, at least 160x90")
    scale = width / 1920
    gap = 2 * round(16 * scale)
    panel_width = (width - gap) // 2
    panel_height = round(panel_width * 9 / 16)
    top = (height - panel_height) // 2
    return ComparisonLayout(
        width,
        height,
        (0, top, panel_width, panel_height),
        (panel_width + gap, top, panel_width, panel_height),
        (0, top + panel_height + round(32 * scale), width, max(1, round(36 * scale))),
    )


def playback_frames(
    corrections: Sequence[Correction], start: int, stop: int, hold_on_change_frames: int
) -> tuple[int, ...]:
    if not 0 <= start < stop <= len(corrections):
        raise ValueError("frames must be a nonempty half-open interval inside the clip")
    if type(hold_on_change_frames) is not int or hold_on_change_frames < 0:
        raise ValueError("hold_on_change_frames must be a non-negative integer")
    return tuple(
        source
        for source in range(start, stop)
        for _ in range(
            1 + (hold_on_change_frames if corrections[source] in CHANGES else 0)
        )
    )


def draw_comparison_frame(
    image: NDArray[np.uint8],
    candidates: Sequence[Candidate],
    annotation: BallFrameAnnotation,
    corrections: Sequence[Correction],
    *,
    source_frame: int,
    event_seconds: float,
    layout: ComparisonLayout,
    source_size: tuple[int, int],
) -> NDArray[np.uint8]:
    """Draw two letterboxed panels and the full-clip classification strip.

    An event starts when the frame class changes (also at the interval start).
    Animation follows output time, so explicitly duplicated frames animate too.
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be RGB uint8")
    height, width = image.shape[:2]
    source_width, source_height = source_size
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source coordinate dimensions must be positive")
    _, _, panel_width, panel_height = layout.left
    scale = min(panel_width / width, panel_height / height)
    scaled_width, scaled_height = round(width * scale), round(height * scale)
    offset_x = (panel_width - scaled_width) // 2
    offset_y = (panel_height - scaled_height) // 2
    panel: NDArray[np.uint8] = np.full(
        (panel_height, panel_width, 3), _rgb(style.BACKGROUND), dtype=np.uint8
    )
    panel[offset_y : offset_y + scaled_height, offset_x : offset_x + scaled_width] = (
        cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
    )
    left, right = panel.copy(), panel.copy()

    def point(center: Sequence[float]) -> tuple[int, int]:
        return (
            round(offset_x + center[0] * scaled_width / source_width),
            round(offset_y + center[1] * scaled_height / source_height),
        )

    radius = max(2, round(panel_width * 0.009))
    thickness = max(1, round(layout.width / 960))
    green, gray = _rgb(style.NEW), _rgb(style.FIXED)
    white = (255, 255, 255)
    for candidate in candidates[1:3]:
        overlay = left.copy()
        cv2.circle(
            overlay,
            point(candidate.center_px),
            max(1, radius // 3),
            white,
            -1,
            cv2.LINE_AA,
        )
        cv2.addWeighted(
            overlay, candidate.score, left, 1 - candidate.score, 0, dst=left
        )
    if candidates:
        cv2.circle(
            left, point(candidates[0].center_px), radius, white, thickness, cv2.LINE_AA
        )
    kind = corrections[source_frame]
    if annotation.reviewed:
        if candidates and kind != Correction.AGREE:
            origin = point(candidates[0].center_px)
            cv2.circle(right, origin, radius, gray, thickness, cv2.LINE_AA)
            if kind == Correction.REMOVED:
                red = _rgb(style.CORRECTION_REMOVED)
                for sign in (-1, 1):
                    cv2.line(
                        right,
                        (origin[0] - radius, origin[1] - sign * radius),
                        (origin[0] + radius, origin[1] + sign * radius),
                        red,
                        thickness,
                        cv2.LINE_AA,
                    )
            progress = min(1.0, max(0.0, event_seconds / 0.3))
            for ball in annotation.balls:
                if ball.center_px is None or progress == 0:
                    continue
                target = point(ball.center_px)
                end = (
                    round(origin[0] + (target[0] - origin[0]) * progress),
                    round(origin[1] + (target[1] - origin[1]) * progress),
                )
                if math.dist(origin, end) > radius:
                    cv2.arrowedLine(
                        right,
                        origin,
                        end,
                        green,
                        thickness,
                        cv2.LINE_AA,
                        tipLength=0.15,
                    )
        for ball in annotation.balls:
            if ball.center_px is None:
                continue
            center = point(ball.center_px)
            if ball.status == "visible":
                cv2.circle(right, center, radius, green, -1, cv2.LINE_AA)
            elif ball.status == "occluded":
                for angle in range(0, 360, 45):
                    cv2.ellipse(
                        right,
                        center,
                        (radius, radius),
                        0,
                        angle,
                        angle + 24,
                        green,
                        thickness,
                        cv2.LINE_AA,
                    )
            elif ball.status == "interpolated":
                cv2.circle(right, center, radius, green, thickness, cv2.LINE_AA)
    if kind in _COLORS and 0 <= event_seconds < 0.4:
        for target_panel in (left, right):
            cv2.rectangle(
                target_panel,
                (0, 0),
                (panel_width - 1, panel_height - 1),
                _rgb(_COLORS[kind]),
                thickness,
            )
    output: NDArray[np.uint8] = np.full(
        (layout.height, layout.width, 3), _rgb(style.BACKGROUND), dtype=np.uint8
    )
    for rect, contents in ((layout.left, left), (layout.right, right)):
        x, y, w, h = rect
        output[y : y + h, x : x + w] = contents
    x, y, w, h = layout.strip
    palette = np.array(
        [_rgb(_COLORS.get(correction, style.BACKGROUND)) for correction in corrections],
        dtype=np.uint8,
    )
    indices: NDArray[np.int64] = np.arange(w, dtype=np.int64) * len(corrections) // w
    output[y : y + h, x : x + w] = palette[indices]
    cursor = min(w - 1, int((source_frame + 0.5) * w / len(corrections)))
    cv2.line(output, (x + cursor, y), (x + cursor, y + h - 1), white, thickness)
    return output


def render_comparison(
    clip: AnnotationComparison,
    config: AnnotationCorrectionConfig,
    *,
    manifest: dict[str, Any],
) -> Path:
    if config.frame_range is None:
        raise ValueError("render requires a frame interval")
    output_dir = config.run_dir / "presentation"
    if output_dir.exists():
        raise FileExistsError(f"output_dir already exists: {output_dir}")
    start, stop = config.frame_range
    sources = playback_frames(
        clip.corrections, start, stop, config.hold_on_change_frames
    )
    frame_input = open_frame_input(
        clip,
        source=config.frame_source,
        store_dir=config.store_dir,
    )
    timeline = frame_input.timeline
    if Fraction(clip.prepared.manifest.nominal_fps) != timeline.rate or any(
        duration * timeline.time_base != 1 / timeline.rate
        for duration in timeline.durations
    ):
        raise ValueError(
            "native-FPS presentation requires constant source frame durations"
        )
    fps = float(timeline.rate)
    layout = comparison_layout(config.width, config.height)
    decoder = frame_input.decode(start, stop)
    image: NDArray[np.uint8] | None = None
    previous_source = -1
    event_start = 0
    previous_kind: Correction | None = None

    def draw_frame(index: int) -> NDArray[np.uint8]:
        nonlocal image, previous_source, event_start, previous_kind
        source = sources[index]
        if source != previous_source:
            image = next(decoder)
            previous_source = source
        kind = clip.corrections[source]
        if kind != previous_kind:
            event_start, previous_kind = index, kind
        if image is None:
            raise RuntimeError("decoder did not provide the source frame")
        return draw_comparison_frame(
            image,
            clip.cache.frames[str(source)].candidates,
            clip.annotation.frames[source],
            clip.corrections,
            source_frame=source,
            event_seconds=(index - event_start) / fps,
            layout=layout,
            source_size=(clip.prepared.manifest.width, clip.prepared.manifest.height),
        )

    stills = {"first": 0, "last": len(sources) - 1}
    for index, source in enumerate(sources):
        kind = clip.corrections[source]
        if kind in CHANGES:
            stills.setdefault(kind.value, index)
    manifest = {
        **manifest,
        "clip_id": clip.annotation.clip_id,
        "inputs": clip.input_manifest(),
        "frame_input": frame_input.metadata,
        "frames": {"start": start, "stop": stop},
        "class_counts": class_counts(clip.corrections[start:stop]),
        "full_clip_class_counts": class_counts(clip.corrections),
        "radius_source_px": clip.radius_px,
        "layout": {
            "resolution": [layout.width, layout.height],
            "left": layout.left,
            "right": layout.right,
            "strip": layout.strip,
        },
        "playback": {
            "fps": fps,
            "native_fps": str(timeline.rate),
            "source_frame_count": stop - start,
            "output_frame_count": len(sources),
            "hold_on_change_frames": config.hold_on_change_frames,
            "duplicated_frames": len(sources) - (stop - start),
            "output_to_source_frame": sources,
        },
    }
    try:
        render_presentation(
            draw_frame,
            frame_count=len(sources),
            fps=fps,
            output_dir=output_dir,
            name="annotation_correction",
            gif_width=config.gif_width,
            gif_fps=config.gif_fps,
            stills=stills,
            manifest=manifest,
        )
    finally:
        decoder.close()
    return output_dir


def run_visualization(config: AnnotationCorrectionConfig) -> int:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    manifest: dict[str, Any] = {
        "schema": "ball_annotation_correction_presentation.v1",
        "git_commit": commit,
        "config": config.resolved_config,
        "provenance": {
            "left_proposals": "measured",
            "right_codex_annotations": "measured",
            "correction_strip": "measured",
            "arrows_and_borders": "schematic",
        },
        "annotation_semantics": "Codex reviewed annotations; proposals are search hints, not automatic corrections",
        "campaign_dir": str(config.campaign_dir),
    }
    if config.mode == "rank":
        rows = rank_clips(
            config.annotation_root, config.campaign_dir, radius_px=config.radius_px
        )
        output_dir = config.run_dir / "presentation"
        output_dir.mkdir(parents=True, exist_ok=False)
        ranking = {
            "schema": "ball_annotation_correction_ranking.v1",
            "radius_px": config.radius_px,
            "clips": rows,
        }
        (output_dir / "ranking.json").write_text(
            json.dumps(ranking, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"Ranked {len(rows)} clips: {output_dir / 'ranking.json'}")
    else:
        if config.clip_id is None:
            raise ValueError("render requires clip_id")
        index = prepared_clips(config.annotation_root)
        if config.clip_id not in index:
            raise ValueError(f"no prepared manifest for {config.clip_id}")
        clip = load_comparison(
            config.annotation_root,
            config.campaign_dir,
            config.clip_id,
            radius_px=config.radius_px,
            prepared=index[config.clip_id],
        )
        output_dir = render_comparison(clip, config, manifest=manifest)
        print(f"Rendered annotation comparison: {output_dir}")
    return 0
