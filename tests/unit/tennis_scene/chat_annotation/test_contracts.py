from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.tennis_scene.chat_annotation.runtime.contracts import (
    Annotation,
    Ball,
    ClipManifest,
    FrameRange,
    Player,
    annotation_clip_id,
    make_template,
    read_json,
)
from src.tennis_scene.chat_annotation.runtime.validation import validate_annotation


@pytest.mark.parametrize(
    "text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}']
)
def test_json_rejects_silent_ambiguity(tmp_path: Path, text: str) -> None:
    file = tmp_path / "bad.json"
    file.write_text(text)
    with pytest.raises(ValueError):
        read_json(file)


def test_confirmed_empty_scene_differs_from_unreviewed(
    manifest: ClipManifest, annotation: Annotation
) -> None:
    assert validate_annotation(annotation, manifest).status == "completed"
    pending = make_template(manifest)
    report = validate_annotation(pending, manifest)
    assert report.status == "partial" and report.reviewed_frames == 0
    pending.status = "completed"
    assert validate_annotation(pending, manifest).errors


def test_context_frames_are_required_annotations(manifest: ClipManifest) -> None:
    manifest.target_range = FrameRange(start=1, stop=11)
    manifest.frames[0].is_target = manifest.frames[-1].is_target = False
    manifest = ClipManifest.model_validate(manifest.model_dump())
    result = make_template(manifest)
    assert [f.frame_index for f in result.frames] == list(range(12))
    assert validate_annotation(result, manifest).target_frames == 12
    result.frames = result.frames[1:-1]
    assert validate_annotation(result, manifest).errors


@pytest.mark.parametrize("change", ["missing", "duplicate", "order", "foreign"])
def test_frame_coverage_rejects_gaps_duplicates_reordering_and_foreign_frames(
    manifest: ClipManifest, annotation: Annotation, change: str
) -> None:
    if change == "missing":
        annotation.frames.pop(2)
    elif change == "duplicate":
        annotation.frames[2] = annotation.frames[1]
    elif change == "order":
        annotation.frames.reverse()
    else:
        annotation.frames[-1].frame_index = 99
    assert validate_annotation(annotation, manifest).status == "failed"


def test_identity_uses_unique_input_filename_and_dimensions(
    manifest: ClipManifest, annotation: Annotation
) -> None:
    other = manifest.model_copy(update={"filename": "other__run__clip.mp4"})
    assert other.clip_id == manifest.clip_id
    assert annotation_clip_id(other) != annotation_clip_id(manifest)
    assert validate_annotation(annotation, other).errors
    annotation.width += 1
    assert validate_annotation(annotation, manifest).errors


def test_amodal_bbox_can_extend_outside_with_truncation(
    manifest: ClipManifest, annotation: Annotation
) -> None:
    player = Player(
        track_id="p1",
        bbox_xyxy=[-10, 10, 80, 300],
        bbox_source="inferred",
        occluded=True,
        truncated=True,
    )
    annotation.frames[0].players = [player]
    assert not validate_annotation(annotation, manifest).errors
    player.truncated = False
    assert validate_annotation(annotation, manifest).errors
    player.bbox_source = "observed"
    with pytest.raises(ValidationError, match="must be inferred"):
        Player.model_validate(player.model_dump())


def test_non_player_and_court_contracts_are_rejected(annotation: Annotation) -> None:
    payload = annotation.model_dump()
    payload["court_mode"] = "unavailable"
    with pytest.raises(ValidationError, match="Extra inputs"):
        Annotation.model_validate(payload)
    with pytest.raises(ValidationError, match="Extra inputs"):
        Player.model_validate(
            {
                "track_id": "p1",
                "kind": "non_player",
                "non_player_role": "chair_umpire",
                "bbox_xyxy": [0, 0, 10, 20],
                "bbox_source": "observed",
                "occluded": False,
                "truncated": False,
            }
        )


def test_ball_missing_and_observed_positions() -> None:
    Ball(track_id="b", center_px=None, status="out_of_frame", interpolation_frames=None)
    Ball(track_id="b", center_px=None, status="occluded", interpolation_frames=None)
    with pytest.raises(ValidationError):
        Ball(track_id="b", center_px=None, status="visible", interpolation_frames=None)
    with pytest.raises(ValidationError):
        Ball(
            track_id="b",
            center_px=[float("nan"), 2],
            status="visible",
            interpolation_frames=None,
        )
    with pytest.raises(ValidationError):
        Ball(
            track_id="b",
            center_px=[1, 2],
            status="visible",
            interpolation_frames=[0, 2],
        )


def test_unresolved_position_is_partial_even_when_frame_was_reviewed(
    manifest: ClipManifest, annotation: Annotation
) -> None:
    annotation.frames[0].balls = [
        Ball(
            track_id="b", center_px=None, status="unresolved", interpolation_frames=None
        )
    ]
    annotation.status = "partial"
    report = validate_annotation(annotation, manifest)
    assert report.status == "partial" and report.reviewed_frames == len(manifest.frames)
    assert report.issues and not report.errors
    annotation.status = "completed"
    assert validate_annotation(annotation, manifest).errors


def test_duplicate_tracks_and_outside_ball_are_rejected(
    manifest: ClipManifest, annotation: Annotation
) -> None:
    ball = Ball(
        track_id="b", center_px=[1, 2], status="visible", interpolation_frames=None
    )
    annotation.frames[0].balls = [ball, ball]
    assert validate_annotation(annotation, manifest).errors
    annotation.frames[0].balls = [ball]
    ball.center_px = [float(manifest.width), 0]
    assert validate_annotation(annotation, manifest).errors
