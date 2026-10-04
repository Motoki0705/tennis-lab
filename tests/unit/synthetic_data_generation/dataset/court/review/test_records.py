"""Unknown renderer evidence and empty target splits must remain inspectable."""

from __future__ import annotations

from typing import Any

import pytest

from src.synthetic_data_generation.dataset.court.review.records import (
    rejection_summary,
    sample_summary,
    split_target_counts,
    visibility_summary,
)


def projection(points: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "courts": [
            {
                "court_instance_id": "court-000",
                "coverage_mode": "partial",
                "classes": [
                    {"class_id": 0, "class_name": "far_doubles_left", "points": points}
                ],
            }
        ]
    }


def point(
    inside: bool | None, visible: bool | None, *, front: bool = True
) -> dict[str, Any]:
    return {
        "physical_index": 0,
        "uv": [12.5, 4.0],
        "in_frame": inside,
        "in_front": front,
        "renderer_visible": visible,
    }


def sample(target: str, split: str) -> dict[str, Any]:
    return {
        "sample_id": "sample",
        "trajectory_group_id": "group",
        "trajectory_frame_index": 12,
        "view_id": "view",
        "camera": {"camera_id": "camera"},
        "width": 960,
        "height": 540,
        "split": split,
        "target_court": {"binding": {"court_instance_id": target}},
        "projection": projection([point(True, True), point(False, False)]),
    }


def test_geometry_renderer_and_unknown_are_separate() -> None:
    result = visibility_summary(
        projection(
            [
                point(True, True),
                point(True, False),
                point(False, False),
                point(False, None, front=False),
                point(None, None),
            ]
        )
    )
    assert result["counts"] == {
        "total": 5,
        "in_frame": 2,
        "visible": 1,
        "in_frame_hidden": 1,
        "out_of_frame": 2,
        "behind_camera": 1,
        "unknown_geometry": 1,
        "unknown_renderer": 2,
    }
    rows = result["courts"][0]["points"]
    assert rows[1]["renderer"] == "not_visible"
    assert rows[3]["renderer"] == "unrecorded"
    assert rows[3]["geometry"] == "behind_camera"
    assert rows[4]["geometry"] == "unrecorded"
    assert visibility_summary(None) == {
        "projection_recorded": False,
        "counts": None,
        "courts": [],
    }


def test_legacy_two_point_class_is_not_counted_as_one_point() -> None:
    result = visibility_summary(projection([point(True, True), point(True, False)]))
    assert result["counts"]["total"] == 2
    assert result["courts"][0]["points"][1]["physical_index"] == 0


def test_invalid_visibility_does_not_become_truthy() -> None:
    with pytest.raises(ValueError, match="Visibility"):
        visibility_summary(
            projection([dict(point(True, True), renderer_visible="false")])
        )


def test_sample_counts_are_bound_to_target_court() -> None:
    record = sample("court-001", "test")
    record["projection"]["courts"].append(
        {"court_instance_id": "court-001", "coverage_mode": "none", "classes": []}
    )
    summary = sample_summary(record)
    assert summary["target_court"] == "court-001"
    assert summary["target_counts"]["total"] == 0
    assert summary["counts"]["total"] == 2
    assert summary["frame"] == 12
    assert summary["target_coverage"] == "none"


def test_split_matrix_keeps_zero_cells_and_uses_target_not_background() -> None:
    records = [sample("court-001", "test"), sample("court-000", "train")]
    matrix = split_target_counts(records, ["court-000", "court-001", "court-002"])
    assert matrix["test"] == {"court-000": 0, "court-001": 1, "court-002": 0}
    assert matrix["validation"] == {"court-000": 0, "court-001": 0, "court-002": 0}
    assert matrix["train"]["court-000"] == 1
    records[0].pop("target_court")
    records[0]["metadata"] = {"target_court": "court-002"}
    assert (
        split_target_counts(records, ["court-000", "court-001", "court-002"])["test"][
            "court-002"
        ]
        == 1
    )
    with pytest.raises(ValueError, match="aligned courts"):
        split_target_counts([sample("unknown", "test")], ["court-000"])


def test_reject_counts_can_exist_without_candidate_details() -> None:
    result = rejection_summary({"metrics": {"rejected_frame_count": 15}})
    assert result["count"] == 15
    assert result["records_recorded"] is False
    assert result["record_count"] is None
    assert result["reason_counts"] == {}
    assert result["samples"] == []


def test_pre_render_rejection_has_geometry_but_no_renderer_evidence() -> None:
    record = sample("court-000", "train")
    record["projection"] = projection([point(True, None)])
    record["reasons"] = ["insufficient_pre_render_semantic_coverage"]
    result = rejection_summary(
        {"metrics": {"rejected_frame_count": 1}, "rejected_samples": [record]}
    )
    assert result["record_count"] == 1
    assert result["samples"][0]["projection_recorded"] is True
    assert result["samples"][0]["renderer_recorded"] is False
    assert result["images"] == "not_retained"
    record["projection"] = None
    assert (
        rejection_summary(
            {"metrics": {"rejected_frame_count": 1}, "rejected_samples": [record]}
        )["samples"][0]["projection_recorded"]
        is False
    )
