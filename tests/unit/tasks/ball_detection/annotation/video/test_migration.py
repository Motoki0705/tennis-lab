"""Evidence-gated conversion, independent expected values and rejection classes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from src.tasks.ball_detection.annotation.video import (
    BallTarget,
    FrameIdentity,
    VideoAnnotationInput,
    VideoIdentity,
    read_annotation,
    summarize_annotation,
    write_annotation,
)
from src.tasks.ball_detection.annotation.video.migration import (
    MigrationEvidence,
    convert_v2_annotation,
)
from src.tasks.base.annotation.contracts import FrameRange
from src.tasks.base.annotation.paths import AnnotationLayout
from tests.support.tasks.ball_detection.v2_annotation import legacy_data


@pytest.fixture
def pinned() -> VideoAnnotationInput:
    return VideoAnnotationInput(
        source=VideoIdentity(recording_id="original-match-7", camera_id="cam-east", file_name="native.mp4",
                             media_sha256="12" * 32, size_bytes=791, width=137, height=83,
                             frame_count=5, time_base="1/30000", nominal_fps="30000/1001"),
        target=BallTarget(target_semantics="active_ball", track_id="1"),
        frame_range=FrameRange(start=0, stop=5),
        frames=tuple(FrameIdentity(frame_index=i, pts=p) for i, p in enumerate((301, 1802, 3305, 6308, 8309))),
    )


def evidence(path: Path) -> MigrationEvidence:
    return MigrationEvidence(original_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             target_semantics="active_ball", target_evidence="original target.scope",
                             processing_complete=True, processing_evidence="original summary.all_frames_reviewed=true")


def test_known_v2_information_survives_conversion_without_new_states(tmp_path: Path, pinned: VideoAnnotationInput) -> None:
    raw = legacy_data(pinned)
    path = tmp_path / "old.json"
    path.write_text(json.dumps(raw))
    before = path.read_bytes()
    annotation, record = convert_v2_annotation(path, expected_input=pinned, evidence=evidence(path))
    assert [f.state for f in annotation.frames] == ["in_frame", "in_frame", "in_frame", "unknown", "in_frame"]
    assert [f.occlusion for f in annotation.frames] == ["not_occluded", "unknown", "occluded", "unknown", "not_occluded"]
    assert [(f.position.x, f.position.y) if f.position else None for f in annotation.frames] == [
        (9., 13.), (22.5, 7.25), (31.75, 8.), None, (49.25, 16.5),
    ]
    assert annotation.frames[1].position is not None
    assert annotation.frames[1].position.provenance.support_frames == (0, 2)
    assert annotation.frames[2].position is not None
    assert annotation.frames[2].position.provenance.kind == "occlusion_estimated"
    assert annotation.frames[2].notes == "legacy note"
    assert [f.break_before for f in annotation.frames] == [True, False, False, False, True]
    summary = summarize_annotation(annotation)
    assert (summary.direct_observation, summary.estimated, summary.final_unknown, summary.active_ball_negative_candidate) == (2, 2, 1, 0)
    assert annotation.input_identity == pinned
    assert record.evidence.original_sha256 == hashlib.sha256(before).hexdigest()
    assert record.annotation_sha256 == annotation.sha256
    assert record.counts == dict(observed=2, interpolated=1, occlusion_estimated=1, unresolved=1)
    layout = AnnotationLayout.testing(tmp_path / "new")
    destination = layout.path("annotations", "v3.json")
    ref = write_annotation(layout, destination, annotation, expected_input=pinned, require_complete=True)
    assert write_annotation(layout, destination, annotation, expected_input=pinned) == ref
    assert read_annotation(destination, expected_input=pinned) == annotation
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", [
    lambda d: d["source"].update(sha256="34" * 32),
    lambda d: d["source"].update(width=138),
    lambda d: d["source"].update(fps_numerator=25),
    lambda d: d["target"].update(track_id="another"),
    lambda d: d["frames"][1].update(pts=1803),
    lambda d: d["frames"][2].update(frame_index=3),
    lambda d: d["frames"][1].update(support_frames=[]),
    lambda d: d["frames"][1].update(support_frames=[1, 2]),
    lambda d: d["frames"][0].update(support_frames=[1]),
    lambda d: d["frames"][0].update(center_normalized={"x": 9 / 136, "y": 13 / 82}),
    lambda d: d["frames"][0].update(center_px={"x": 136.5, "y": 13.}, center_normalized={"x": 136.5 / 137, "y": 13 / 83}),
    lambda d: d["frames"][3].update(center_normalized={"x": 0., "y": 0.}),
    lambda d: d["frames"][3].update(status="no_active_ball"),
    lambda d: d["frames"][2].update(visibility="visible"),
    lambda d: d["frames"][0].update(center_px={"x": float("nan"), "y": 13.}),
    lambda d: d["summary"]["counts"].update(observed=3),
    lambda d: d["coordinate_system"].update(x_axis="left"),
])
def test_inconsistent_v2_is_rejected(tmp_path: Path, pinned: VideoAnnotationInput, change: Callable[[dict[str, Any]], None]) -> None:
    raw = legacy_data(pinned)
    change(raw)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        convert_v2_annotation(path, expected_input=pinned, evidence=evidence(path))


@pytest.mark.parametrize("change", [
    {"target_semantics": "all_balls"}, {"target_evidence": " "},
    {"processing_complete": False}, {"processing_evidence": " "},
    {"original_sha256": "ab" * 32},
])
def test_conversion_requires_pinned_target_and_processing_evidence(tmp_path: Path, pinned: VideoAnnotationInput, change: dict[str, Any]) -> None:
    path = tmp_path / "old.json"
    path.write_text(json.dumps(legacy_data(pinned)))
    data = evidence(path).model_dump()
    data.update(change)
    with pytest.raises(ValueError):
        convert_v2_annotation(path, expected_input=pinned, evidence=MigrationEvidence.model_validate(data))


def test_duplicate_json_keys_are_rejected(tmp_path: Path, pinned: VideoAnnotationInput) -> None:
    path = tmp_path / "old.json"
    path.write_text(json.dumps(legacy_data(pinned)).replace('"pts": 301', '"pts": 301, "pts": 301'))
    with pytest.raises(ValueError, match="duplicate JSON key"):
        convert_v2_annotation(path, expected_input=pinned, evidence=evidence(path))
