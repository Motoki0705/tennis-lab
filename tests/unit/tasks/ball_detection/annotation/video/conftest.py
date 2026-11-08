"""A nonzero source scope with asymmetric camera, geometry and VFR timestamps."""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.tasks.ball_detection.annotation.video import (
    VideoAnnotationInput,
    VideoBallAnnotation,
)


@pytest.fixture
def annotation_data() -> dict[str, Any]:
    # This is a specification table, not generated from production predicates.
    decisions = [
        ("in_frame", "not_occluded", "observed", (96.5, 12.25), []),
        ("in_frame", "unknown", "interpolated", (91.75, 15.5), [41, 43]),
        ("in_frame", "not_occluded", "estimated", (86.125, 18.75), [41, 44]),
        ("in_frame", "occluded", "occlusion_estimated", (81.375, 22.0), [42, 46]),
        ("in_frame", "occluded", "interpolated", (76.625, 25.25), [44, 46]),
        ("in_frame", "occluded", None, None, []),
        ("out_of_frame", "not_applicable", None, None, []),
        ("no_active_ball", "not_applicable", None, None, []),
        ("unknown", "unknown", None, None, []),
        ("unprocessed", "unknown", None, None, []),
    ]
    pts = [9009, 10511, 12014, 15017, 16518, 18023, 21029, 22531, 24037, 27041]
    frames = []
    for index, (state, occlusion, kind, xy, support) in enumerate(decisions):
        position = (
            None
            if xy is None
            else {
                "x": xy[0],
                "y": xy[1],
                "provenance": {"kind": kind, "support_frames": support},
            }
        )
        frames.append(
            {
                "frame_index": 41 + index,
                "pts": pts[index],
                "state": state,
                "occlusion": occlusion,
                "position": position,
                "break_before": index == 0,
                "notes": "context judgement" if index == 2 else "",
            }
        )
    return {
        "schema_version": "video_ball_annotation.v3",
        "source": {
            "recording_id": "match/recording-red",
            "camera_id": "cam7-east",
            "file_name": "east.mp4",
            "media_sha256": "4b" * 32,
            "size_bytes": 98765,
            "width": 137,
            "height": 83,
            "frame_count": 103,
            "time_base": "1/90000",
            "nominal_fps": "60000/1001",
            "rotation": 0,
        },
        "target": {
            "target_semantics": "active_ball",
            "label": "tennis_ball",
            "track_id": "rally-red",
        },
        "frame_range": {"start": 41, "stop": 51},
        "coordinate_system": "native_pixels_top_left_endpoint",
        "frames": frames,
    }


@pytest.fixture
def annotation(annotation_data: dict[str, Any]) -> VideoBallAnnotation:
    result: VideoBallAnnotation = VideoBallAnnotation.model_validate_json(
        json.dumps(annotation_data)
    )
    return result


@pytest.fixture
def expected_input(annotation: VideoBallAnnotation) -> VideoAnnotationInput:
    return annotation.input_identity
