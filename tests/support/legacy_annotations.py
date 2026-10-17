"""Construct synthetic retained-format artifacts without restoring old entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.tasks.annotation.migration.legacy_format.contracts import (
    BALL_SCHEMA_VERSION,
    PLAYER_SCHEMA_VERSION,
    SCHEMA_VERSION,
    Annotation,
    BallAnnotation,
    BallFrameAnnotation,
    ClipManifest,
    FrameAnnotation,
    PlayerAnnotation,
    PlayerFrameAnnotation,
    SupportedAnnotation,
    annotation_clip_id,
)
from src.tasks.annotation.migration.legacy_format.geometry import (
    BallCapableAnnotation,
    interpolation_values,
)

KIT_VERSION = "5.0.0"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def make_template(
    manifest: ClipManifest, target: str = "combined"
) -> SupportedAnnotation:
    common: dict[str, Any] = {
        "clip_id": annotation_clip_id(manifest),
        "width": manifest.width,
        "height": manifest.height,
        "frame_count": len(manifest.frames),
        "status": "partial",
        "issues": [],
    }
    if target == "ball":
        return BallAnnotation(
            schema_version=BALL_SCHEMA_VERSION,
            **common,
            frames=[
                BallFrameAnnotation(
                    frame_index=frame.frame_index,
                    reviewed=False,
                    balls=[],
                    interpolation_break=False,
                    notes="",
                )
                for frame in manifest.frames
            ],
        )
    if target == "player":
        return PlayerAnnotation(
            schema_version=PLAYER_SCHEMA_VERSION,
            **common,
            frames=[
                PlayerFrameAnnotation(
                    frame_index=frame.frame_index,
                    reviewed=False,
                    players=[],
                    notes="",
                )
                for frame in manifest.frames
            ],
        )
    if target != "combined":
        raise ValueError(f"unsupported annotation target: {target}")
    return Annotation(
        schema_version=SCHEMA_VERSION,
        **common,
        frames=[
            FrameAnnotation(
                frame_index=frame.frame_index,
                reviewed=False,
                players=[],
                balls=[],
                interpolation_break=False,
                notes="",
            )
            for frame in manifest.frames
        ],
    )


def interpolate_ball(
    annotation: BallCapableAnnotation,
    manifest: ClipManifest,
    track_id: str,
    start: int,
    stop: int,
) -> BallCapableAnnotation:
    values = interpolation_values(annotation, manifest, track_id, start, stop)
    result: BallCapableAnnotation = annotation.model_copy(deep=True)
    for frame in result.frames:
        if frame.frame_index not in values:
            continue
        ball = next(ball for ball in frame.balls if ball.track_id == track_id)
        if ball.center_px is not None or ball.status == "out_of_frame":
            raise ValueError(
                "interpolation must not overwrite a position or fill an out-of-frame ball"
            )
        ball.center_px = values[frame.frame_index]
        ball.status = "interpolated"
        ball.interpolation_frames = [start, stop]
    return result
