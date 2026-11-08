"""Actual immutable JSON files, input binding and the explicit v2 boundary."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.tasks.ball_detection.annotation.video import (
    VideoAnnotationInput,
    VideoBallAnnotation,
    read_annotation,
    summarize_annotation,
    write_annotation,
)
from src.tasks.ball_detection.generate_dataset.frame_store.sources.meiji import (
    read_camera,
)
from src.tasks.base.annotation.artifacts import verify_reference
from src.tasks.base.annotation.paths import AnnotationLayout


def test_native_pixels_and_input_mapping_survive_immutable_save(
    tmp_path: Path,
    annotation_data: dict[str, Any],
    expected_input: VideoAnnotationInput,
) -> None:
    annotation_data["frames"][0]["position"].update(x=136.0, y=0.0)
    annotation = VideoBallAnnotation.model_validate_json(json.dumps(annotation_data))
    layout = AnnotationLayout.testing(tmp_path / "store")
    path = layout.path("annotations", "cam7-east", "revision-0.json")
    reference = write_annotation(
        layout, path, annotation, expected_input=expected_input
    )
    assert verify_reference(layout, reference) == path
    saved = json.loads(path.read_text())
    assert saved == annotation_data
    assert saved["frames"][1]["position"] == {
        "x": 91.75,
        "y": 15.5,
        "provenance": {"kind": "interpolated", "support_frames": [41, 43]},
    }
    reloaded = read_annotation(path, expected_input=expected_input)
    assert reloaded.frames[0].position is not None
    assert (reloaded.frames[0].position.x, reloaded.frames[0].position.y) == (
        136.0,
        0.0,
    )
    assert (
        write_annotation(layout, path, reloaded, expected_input=expected_input)
        == reference
    )
    with pytest.raises(ValueError, match="unprocessed"):
        read_annotation(path, require_complete=True)
    revised = VideoBallAnnotation.model_validate_json(
        path.read_bytes().replace(b'"state":"unprocessed"', b'"state":"unknown"')
    )
    with pytest.raises(ValueError, match="immutable document"):
        write_annotation(layout, path, revised)
    next_path = layout.path("annotations", "cam7-east", "revision-1.json")
    write_annotation(
        layout, next_path, revised, expected_input=expected_input, require_complete=True
    )
    assert summarize_annotation(
        read_annotation(next_path, require_complete=True)
    ).complete
    assert read_annotation(path).frames[-1].state == "unprocessed"


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["source"].update(camera_id="cam2-west"),
        lambda d: d["source"].update(recording_id="match/recording-blue"),
        lambda d: d["source"].update(media_sha256="1a" * 32),
        lambda d: d["source"].update(width=139),
        lambda d: d["source"].update(time_base="1/48000"),
        lambda d: d["target"].update(track_id="rally-blue"),
        lambda d: d["frames"][2].update(pts=12015),
    ],
)
def test_reader_and_writer_reject_same_length_input_substitution(
    tmp_path: Path,
    annotation_data: dict[str, Any],
    expected_input: VideoAnnotationInput,
    change: Callable[[dict[str, Any]], None],
) -> None:
    change(annotation_data)
    path = tmp_path / "foreign.json"
    path.write_text(json.dumps(annotation_data))
    with pytest.raises(ValueError, match="differs from the expected input"):
        read_annotation(path, expected_input=expected_input)
    foreign = VideoBallAnnotation.model_validate_json(path.read_bytes())
    layout = AnnotationLayout.testing(tmp_path / "store")
    destination = layout.path("annotations", "bad.json")
    with pytest.raises(ValueError, match="differs from the expected input"):
        write_annotation(layout, destination, foreign, expected_input=expected_input)
    assert not layout.root.exists()


def test_writer_revalidates_unchecked_copies_before_writing(
    tmp_path: Path, annotation: VideoBallAnnotation
) -> None:
    invalid = annotation.model_copy(update={"frames": annotation.frames[:-1]})
    layout = AnnotationLayout.testing(tmp_path / "store")
    path = layout.path("annotations", "invalid.json")
    with pytest.raises(ValueError, match="every source frame"):
        write_annotation(layout, path, invalid)
    assert not layout.root.exists()


def test_v2_correspondence_preserves_known_information_without_inventing_states(
    tmp_path: Path,
    annotation_data: dict[str, Any],
) -> None:
    # Metadata/label parsing only: these opaque bytes are not decoded as a video.
    media = tmp_path / "legacy.mp4"
    media.write_bytes(b"small v2 metadata fixture")
    statuses = ["observed", "interpolated", "occlusion_estimated", "unresolved"]
    visibility = ["visible", "not_independently_visible", "occluded", "unknown"]
    pixels = [
        {"x": 96.5, "y": 12.25},
        {"x": 91.75, "y": 15.5},
        {"x": 81.375, "y": 22.0},
        None,
    ]
    pts = [301, 1802, 3305, 6308]
    normalized = [
        None if point is None else {"x": point["x"] / 137, "y": point["y"] / 83}
        for point in pixels
    ]
    legacy: dict[str, Any] = {
        "schema_version": "video_ball_annotation.v2",
        "source": {
            "file_name": media.name,
            "sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
            "width": 137,
            "height": 83,
            "frame_count": 4,
            "rotation": 0,
            "time_base": "1/90000",
            "fps_numerator": 60000,
            "fps_denominator": 1001,
        },
        "coordinate_system": {
            "origin": "top_left",
            "frame_index": "zero_based",
            "normalization": {"x": "x_px / width", "y": "y_px / height"},
        },
        "target": {"track_id": 1},
        "review": {"status": "approved"},
        "summary": {"counts": dict.fromkeys(statuses, 1)},
        "frames": [
            {
                "frame_index": i,
                "pts": pts[i],
                "label": "tennis_ball",
                "track_id": 1,
                "status": status,
                "visibility": visibility[i],
                "center_px": pixels[i],
                "center_normalized": normalized[i],
                "break_before": i == 0,
                "support_frames": [0, 2] if i == 1 else None,
            }
            for i, status in enumerate(statuses)
        ],
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy))
    original = path.read_bytes()
    old = read_camera(
        path,
        media,
        video_id="legacy-recording",
        clip_name="clip-red",
        camera_id="cam7-east",
    )
    assert old.labels.instances["point_kind"].tolist() == [0, 1, 2, 3]
    assert old.labels.frames["pts"].tolist() == [301, 1802, 3305, 6308]
    np.testing.assert_array_equal(
        old.labels.xy[:3], [[96.5, 12.25], [91.75, 15.5], [81.375, 22.0]]
    )
    assert np.isnan(old.labels.xy[3]).all()

    # Explicitly supplied external target/group/camera context is required by v3.
    # This is the README's four-row correspondence, not an automatic converter.
    data = annotation_data
    data["source"].update(
        file_name=media.name,
        media_sha256=legacy["source"]["sha256"],
        size_bytes=media.stat().st_size,
        frame_count=4,
        recording_id=old.group_id,
    )
    data["target"]["track_id"] = "1"
    data["frame_range"] = {"start": 0, "stop": 4}
    data["frames"] = [data["frames"][i] for i in [0, 1, 3, 8]]
    for i, row in enumerate(data["frames"]):
        row.update(frame_index=i, pts=pts[i])
        if row["position"] is not None:
            row["position"]["provenance"]["support_frames"] = [0, 2] if i == 1 else []
    proposed = VideoBallAnnotation.model_validate_json(json.dumps(data))
    assert [f.state for f in proposed.frames] == [
        "in_frame",
        "in_frame",
        "in_frame",
        "unknown",
    ]
    assert [f.occlusion for f in proposed.frames] == [
        "not_occluded",
        "unknown",
        "occluded",
        "unknown",
    ]
    assert [
        f.position.provenance.kind if f.position else None for f in proposed.frames
    ] == ["observed", "interpolated", "occlusion_estimated", None]
    summary = summarize_annotation(proposed)
    assert (
        summary.direct_observation,
        summary.estimated,
        summary.final_unknown,
        summary.out_of_frame,
        summary.no_active_ball,
        summary.active_ball_negative_candidate,
    ) == (1, 2, 1, 0, 0, 0)
    with pytest.raises(ValueError, match="video_ball_annotation.v3"):
        read_annotation(path)
    layout = AnnotationLayout.testing(tmp_path)
    with pytest.raises(ValueError, match="immutable document"):
        write_annotation(layout, path, proposed)
    assert path.read_bytes() == original
