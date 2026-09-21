from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.tennis_scene.chat_annotation.runtime.contracts import (
    Annotation,
    Ball,
    ClipManifest,
    Person,
    ValidationReport,
    make_template,
    read_json,
)
from src.tennis_scene.chat_annotation.runtime.review import final_response
from src.tennis_scene.chat_annotation.runtime.validation import validate_annotation


@pytest.mark.parametrize(
    "text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}']
)
def test_json_rejects_silent_ambiguity(tmp_path: Path, text: str) -> None:
    file = tmp_path / "bad.json"
    file.write_text(text)
    with pytest.raises(ValueError):
        read_json(file)


def test_empty_confirmed_scene_differs_from_unreviewed(
    manifest: ClipManifest, annotation: Annotation, definition: dict[str, Any]
) -> None:
    assert (
        validate_annotation(annotation, manifest, "c" * 64, definition).status
        == "completed"
    )
    pending = make_template(manifest, "c" * 64)
    report = validate_annotation(pending, manifest, "c" * 64, definition)
    assert report.status == "partial" and report.reviewed_frames == 0


def test_missing_duplicate_or_foreign_frame_is_rejected(
    manifest: ClipManifest, annotation: Annotation, definition: dict[str, Any]
) -> None:
    annotation.frames.pop(2)
    assert (
        validate_annotation(annotation, manifest, "c" * 64, definition).status
        == "failed"
    )
    annotation.frames.insert(2, annotation.frames[1])
    assert validate_annotation(annotation, manifest, "c" * 64, definition).errors
    annotation.manifest_sha256 = "d" * 64
    assert any(
        "identity" in error
        for error in validate_annotation(
            annotation, manifest, "c" * 64, definition
        ).errors
    )


def test_amodal_bbox_can_extend_outside_with_truncation(
    manifest: ClipManifest, annotation: Annotation, definition: dict[str, Any]
) -> None:
    person = Person(
        track_id="person_1",
        kind="non_player",
        non_player_role="chair_umpire",
        court_relation="target",
        bbox_xyxy=[-10, 10, 80, 300],
        bbox_source="inferred",
        occluded=True,
        truncated=True,
        source_frames=[0, 1],
    )
    annotation.frames[0].people = [person]
    assert not validate_annotation(annotation, manifest, "c" * 64, definition).errors
    person.truncated = False
    assert validate_annotation(annotation, manifest, "c" * 64, definition).errors


def test_ball_null_and_three_status_contract() -> None:
    Ball(
        track_id="b",
        center_px=None,
        status=None,
        missing_reason="out_of_frame",
        source_frames=[],
    )
    Ball(
        track_id="b",
        center_px=None,
        status="occluded",
        missing_reason="unresolved",
        source_frames=[0, 2],
    )
    with pytest.raises(ValidationError):
        Ball(
            track_id="b",
            center_px=None,
            status="visible",
            missing_reason="unresolved",
            source_frames=[],
        )
    with pytest.raises(ValidationError):
        Ball(
            track_id="b",
            center_px=[float("nan"), 2],
            status="visible",
            missing_reason=None,
            source_frames=[0],
        )


def test_unknown_court_membership_is_not_completed(
    manifest: ClipManifest, annotation: Annotation, definition: dict[str, Any]
) -> None:
    annotation.frames[0].people = [
        Person(
            track_id="person_1",
            kind="player",
            non_player_role=None,
            court_relation="unknown",
            bbox_xyxy=[20, 30, 80, 180],
            bbox_source="observed",
            occluded=False,
            truncated=False,
            source_frames=[0],
        )
    ]
    report = validate_annotation(annotation, manifest, "c" * 64, definition)
    assert report.status == "partial"
    assert report.issues and not report.errors


def test_failure_response_flattens_multiline_reasons(manifest: ClipManifest) -> None:
    report = ValidationReport(
        status="failed",
        reviewed_frames=0,
        target_frames=12,
        errors=["failure"],
        issues=[],
    )
    result = final_response(manifest, report, "未生成（first\nsecond\r\nthird）")
    assert len(result.splitlines()) == 5
    assert result.splitlines()[-1] == "成果物: 未生成（first second third）"
