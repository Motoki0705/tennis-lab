"""Independent rejection classes for state, coordinate, provenance and timeline."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from src.tasks.ball_detection.annotation.video import VideoBallAnnotation


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda d: d["frames"][6].update(position=d["frames"][0]["position"]),
            "requires no position",
        ),
        (
            lambda d: d["frames"][7].update(position=d["frames"][0]["position"]),
            "requires no position",
        ),
        (
            lambda d: d["frames"][8].update(position=d["frames"][0]["position"]),
            "requires no position",
        ),
        (
            lambda d: d["frames"][9].update(position=d["frames"][0]["position"]),
            "requires no position",
        ),
        (
            lambda d: d["frames"][5].update(occlusion="not_occluded"),
            "without a position requires known occlusion",
        ),
        (
            lambda d: d["frames"][2].update(occlusion="not_applicable"),
            "applicable occlusion",
        ),
        (
            lambda d: d["frames"][0].update(occlusion="occluded"),
            "observed coordinates require not_occluded",
        ),
        (
            lambda d: d["frames"][3].update(occlusion="unknown"),
            "occlusion_estimated coordinates require occluded",
        ),
        (
            lambda d: d["frames"][1]["position"]["provenance"].update(
                support_frames=[]
            ),
            "two support_frames",
        ),
        (
            lambda d: d["frames"][1]["position"]["provenance"].update(
                support_frames=[43, 41]
            ),
            "unique and increasing",
        ),
        (
            lambda d: d["frames"][1]["position"]["provenance"].update(
                support_frames=[42, 43]
            ),
            "must enclose",
        ),
        (
            lambda d: d["frames"][2]["position"]["provenance"].update(
                support_frames=[41, 103]
            ),
            "exceed the original",
        ),
        (lambda d: d["frames"][0]["position"].update(x=float("nan")), "finite number"),
        (lambda d: d["frames"][0]["position"].update(y=float("inf")), "finite number"),
        (
            lambda d: d["frames"][0]["position"].update(x=-0.01),
            "greater than or equal to 0",
        ),
        (
            lambda d: d["frames"][2]["position"].update(x=136.01),
            "outside native pixel endpoints",
        ),
        (
            lambda d: d["frames"][3]["position"].update(y=82.01),
            "outside native pixel endpoints",
        ),
        (
            lambda d: d["frames"][9].update(break_before=True),
            "unprocessed frames cannot assert",
        ),
    ],
)
def test_invalid_state_position_and_provenance_are_rejected(
    annotation_data: dict[str, Any],
    change: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    change(annotation_data)
    with pytest.raises(ValueError, match=message):
        VideoBallAnnotation.model_validate_json(json.dumps(annotation_data))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.pop("schema_version"), "schema_version"),
        (lambda d: d["target"].pop("target_semantics"), "target_semantics"),
        (lambda d: d["target"].update(target_semantics="all_balls"), "active_ball"),
        (
            lambda d: d.update(coordinate_system="x/width,y/height"),
            "native_pixels_top_left_endpoint",
        ),
        (lambda d: d["source"].update(time_base="0"), "positive rationals"),
        (lambda d: d["source"].update(frame_count=50), "exceeds the original"),
        (lambda d: d["frames"].pop(), "every source frame"),
        (
            lambda d: d["frames"][5].update(frame_index=45),
            "dense and in presentation order",
        ),
        (lambda d: d["frames"][3].update(pts=12014), "strictly increasing"),
        (lambda d: d["frames"][2].update(pts=12014.5), "valid integer"),
    ],
)
def test_invalid_schema_target_and_timeline_are_rejected(
    annotation_data: dict[str, Any],
    change: Callable[[dict[str, Any]], object],
    message: str,
) -> None:
    change(annotation_data)
    with pytest.raises(ValueError, match=message):
        VideoBallAnnotation.model_validate_json(json.dumps(annotation_data))


def test_nested_unvalidated_edit_is_rejected(annotation: VideoBallAnnotation) -> None:
    # Pydantic model_copy deliberately does not validate. The outer contract must.
    invalid = annotation.frames[0].model_copy(update={"occlusion": "occluded"})
    with pytest.raises(ValueError, match="observed coordinates require not_occluded"):
        VideoBallAnnotation(
            schema_version="video_ball_annotation.v3",
            coordinate_system="native_pixels_top_left_endpoint",
            source=annotation.source,
            target=annotation.target,
            frame_range=annotation.frame_range,
            frames=(invalid, *annotation.frames[1:]),
        )
