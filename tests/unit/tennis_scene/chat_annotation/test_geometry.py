from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from src.tennis_scene.chat_annotation.runtime.contracts import (
    Annotation,
    Ball,
    ClipManifest,
    CourtPoint,
    CourtSample,
    FrameRange,
)
from src.tennis_scene.chat_annotation.runtime.geometry import (
    complete_court,
    court_mode,
    interpolate_ball,
)
from src.tennis_scene.chat_annotation.runtime.validation import validate_annotation


def sample(
    definition: dict[str, Any],
    frame_index: int,
    offset: float = 0,
    anchors: tuple[int, ...] = (0, 1, 2, 3),
) -> CourtSample:
    xy = np.asarray(definition["points_xyz"])[:, :2]
    matrix = np.asarray([[70, 3, 960 + offset], [2, -30, 600], [0.002, 0.003, 1]])
    projected = np.column_stack((xy, np.ones(20))) @ matrix.T
    projected = projected[:, :2] / projected[:, 2, None]
    projected[:, 1] -= np.asarray(definition["points_xyz"])[:, 2] * 80
    return CourtSample(
        frame_index=frame_index,
        orientation="known",
        orientation_note="fixed fixture orientation",
        points=[
            CourtPoint(
                index=i,
                name=name,
                point_px=projected[i].tolist() if i in anchors else None,
                visibility="visible" if i in anchors else "unresolved",
                source="observed" if i in anchors else "unresolved",
                source_frames=[frame_index] if i in anchors else [],
                anchor_indices=[],
            )
            for i, name in enumerate(definition["names"])
        ],
    )


def test_ground_completion_preserves_observations_and_never_projects_net(
    manifest: ClipManifest, definition: dict[str, Any], annotation: Annotation
) -> None:
    original = sample(definition, 0)
    result = complete_court(original, definition, manifest)
    assert result.points[:4] == original.points[:4]
    assert all(
        point.source == "homography" and point.point_px is not None
        for point in result.points[4:15]
    )
    assert all(point.point_px is None for point in result.points[15:])
    assert original.points[4].point_px is None
    annotation.court_samples = [result]
    assert not validate_annotation(annotation, manifest, "c" * 64, definition).errors
    assert result.points[5].point_px is not None
    result.points[5].point_px[0] += 2
    assert any(
        "reproduced" in error
        for error in validate_annotation(
            annotation, manifest, "c" * 64, definition
        ).errors
    )


def test_homography_rejects_degenerate_and_inconsistent_anchors(
    manifest: ClipManifest, definition: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="collinear"):
        complete_court(
            sample(definition, 0, anchors=(0, 4, 6, 1)), definition, manifest
        )
    with pytest.raises(ValueError, match="four"):
        complete_court(sample(definition, 0, anchors=(0, 1, 2)), definition, manifest)
    points = sample(definition, 0, anchors=(0, 1, 2, 3, 8))
    assert points.points[8].point_px is not None
    points.points[8].point_px[0] += 100
    with pytest.raises(ValueError, match="residual"):
        complete_court(points, definition, manifest)


def test_static_requires_endpoints_and_whole_clip_motion_review(
    manifest: ClipManifest, definition: dict[str, Any], annotation: Annotation
) -> None:
    annotation.court_samples = [
        complete_court(sample(definition, i), definition, manifest) for i in (0, 6, 11)
    ]
    annotation.camera_motion = "none"
    assert court_mode(annotation, manifest) == "dynamic"
    annotation.camera_review_ranges = [FrameRange(start=0, stop=12)]
    assert court_mode(annotation, manifest) == "static"
    annotation.court_samples[-1] = complete_court(
        sample(definition, 11, offset=8), definition, manifest
    )
    assert court_mode(annotation, manifest) == "dynamic"
    annotation.court_samples[-1] = complete_court(
        sample(definition, 11), definition, manifest
    )
    annotation.frames[4].events = ["cut"]
    assert court_mode(annotation, manifest) == "dynamic"
    annotation.frames[4].events = []
    annotation.camera_motion = "moving"
    assert court_mode(annotation, manifest) == "dynamic"


def ball_interval(annotation: Annotation, start: int = 1, stop: int = 3) -> None:
    for index in range(start, stop + 1):
        visible = index in (start, stop)
        annotation.frames[index].balls = [
            Ball(
                track_id="ball_1",
                center_px=([100, 100] if index == start else [104, 102])
                if visible
                else None,
                status="visible" if visible else None,
                missing_reason=None if visible else "unresolved",
                source_frames=[index] if visible else [],
            )
        ]


def test_interpolation_uses_pts_and_is_reproducible(
    manifest: ClipManifest, annotation: Annotation, definition: dict[str, Any]
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
    assert point.status == "interpolated" and point.source_frames == [1, 3]
    assert annotation.frames[2].balls[0].center_px is None
    assert not validate_annotation(result, manifest, "c" * 64, definition).errors
    point.center_px = [500, 600]
    assert validate_annotation(result, manifest, "c" * 64, definition).errors


@pytest.mark.parametrize("event", ["hit", "bounce", "cut", "play_start", "play_end"])
def test_interpolation_refuses_event_boundaries(
    manifest: ClipManifest, annotation: Annotation, event: str
) -> None:
    ball_interval(annotation)
    annotation.frames[2].events = [event]  # type: ignore[list-item]
    with pytest.raises(ValueError, match="cannot cross"):
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
    annotation.frames[2].balls = []
    with pytest.raises(ValueError, match="explicitly present"):
        interpolate_ball(annotation, manifest, "ball_1", 1, 3)


@pytest.mark.parametrize("point_index", range(15, 20))
def test_known_net_disagreement_prevents_static_reuse(
    manifest: ClipManifest,
    annotation: Annotation,
    definition: dict[str, Any],
    point_index: int,
) -> None:
    annotation.court_samples = [
        sample(definition, i, anchors=tuple(range(20))) for i in (0, 6, 11)
    ]
    annotation.camera_motion = "none"
    annotation.camera_review_ranges = [FrameRange(start=0, stop=12)]
    annotation.court_mode = "static"
    for frame in annotation.frames:
        frame.court_reference_frame = 0
    assert court_mode(annotation, manifest) == "static"
    assert (
        validate_annotation(annotation, manifest, "c" * 64, definition).status
        == "completed"
    )
    point = annotation.court_samples[-1].points[point_index]
    assert point.point_px is not None
    point.point_px[0] += 30
    assert court_mode(annotation, manifest) == "dynamic"
    report = validate_annotation(annotation, manifest, "c" * 64, definition)
    assert report.status == "failed" and report.errors


def test_unresolved_visibility_cannot_carry_a_position(
    manifest: ClipManifest, annotation: Annotation, definition: dict[str, Any]
) -> None:
    observed = sample(definition, 0, anchors=tuple(range(20)))
    observed.points[0].visibility = "unresolved"
    with pytest.raises(ValidationError, match="unresolved visibility"):
        CourtSample.model_validate(observed.model_dump())
    annotation.court_samples = [observed]
    assert (
        validate_annotation(annotation, manifest, "c" * 64, definition).status
        == "failed"
    )
