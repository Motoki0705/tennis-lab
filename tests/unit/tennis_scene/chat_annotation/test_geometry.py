from __future__ import annotations

import pytest

from src.tennis_scene.chat_annotation.runtime.contracts import (
    Annotation,
    Ball,
    ClipManifest,
)
from src.tennis_scene.chat_annotation.runtime.geometry import interpolate_ball
from src.tennis_scene.chat_annotation.runtime.validation import validate_annotation


def ball_interval(annotation: Annotation, start: int = 1, stop: int = 3) -> None:
    for index in range(start, stop + 1):
        visible = index in (start, stop)
        annotation.frames[index].balls = [
            Ball(
                track_id="ball_1",
                center_px=([100, 100] if index == start else [104, 102])
                if visible
                else None,
                status="visible" if visible else "unresolved",
                interpolation_frames=None,
            )
        ]


def test_interpolation_uses_pts_and_is_reproducible(
    manifest: ClipManifest, annotation: Annotation
) -> None:
    manifest.time_base = "1/60"
    timestamps = [0, 2, 3, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    for index, frame in enumerate(manifest.frames):
        frame.source_pts = frame.clip_pts = timestamps[index]
        frame.duration_pts = (
            timestamps[index + 1] - timestamps[index]
            if index + 1 < len(timestamps)
            else 2
        )
    ball_interval(annotation)
    result = interpolate_ball(annotation, manifest, "ball_1", 1, 3)
    point = result.frames[2].balls[0]
    assert point.center_px == [101, 100.5]
    assert point.status == "interpolated" and point.interpolation_frames == [1, 3]
    assert annotation.frames[2].balls[0].center_px is None
    assert not validate_annotation(result, manifest).errors
    point.center_px = [500, 600]
    assert validate_annotation(result, manifest).errors


@pytest.mark.parametrize("boundary", [1, 2, 3])
def test_interpolation_refuses_event_boundaries_including_endpoints(
    manifest: ClipManifest, annotation: Annotation, boundary: int
) -> None:
    ball_interval(annotation)
    annotation.frames[boundary].interpolation_break = True
    with pytest.raises(ValueError, match="cannot cross"):
        interpolate_ball(annotation, manifest, "ball_1", 1, 3)


def test_interpolation_refuses_unreviewed_frames(
    manifest: ClipManifest, annotation: Annotation
) -> None:
    ball_interval(annotation)
    annotation.frames[2].reviewed = False
    with pytest.raises(ValueError, match="inspect every"):
        interpolate_ball(annotation, manifest, "ball_1", 1, 3)


def test_interpolation_refuses_long_gaps_absent_tracks_and_overwrites(
    manifest: ClipManifest, annotation: Annotation
) -> None:
    ball_interval(annotation, 1, 6)
    with pytest.raises(ValueError, match="time limit"):
        interpolate_ball(annotation, manifest, "ball_1", 1, 6)
    ball_interval(annotation)
    annotation.frames[2].balls[0].center_px = [101, 100]
    with pytest.raises(ValueError, match="overwrite"):
        interpolate_ball(annotation, manifest, "ball_1", 1, 3)
    annotation.frames[2].balls[0].center_px = None
    annotation.frames[2].balls[0].status = "out_of_frame"
    with pytest.raises(ValueError, match="out-of-frame"):
        interpolate_ball(annotation, manifest, "ball_1", 1, 3)
    annotation.frames[2].balls = []
    with pytest.raises(ValueError, match="explicitly present"):
        interpolate_ball(annotation, manifest, "ball_1", 1, 3)


@pytest.mark.parametrize("bounds", [(-1, 3), (1, 99), (3, 1), (1, 2)])
def test_interpolation_rejects_invalid_ranges(
    manifest: ClipManifest, annotation: Annotation, bounds: tuple[int, int]
) -> None:
    with pytest.raises(ValueError, match="endpoints"):
        interpolate_ball(annotation, manifest, "ball_1", *bounds)
