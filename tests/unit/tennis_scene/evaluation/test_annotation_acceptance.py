from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.tasks.ball_detection.annotation.video import read_annotation
from src.tennis_scene.evaluation.annotation_acceptance import (
    clip000_acceptance,
    summarize_reference,
    summarize_video_annotations,
)
from tests.support.ball3d_evaluation import make_annotations, make_inputs
from tests.support.tennis_scene.ball_imports import make_import_fixture


@pytest.mark.parametrize(
    ("unavailable", "unknown", "unprocessed", "status"),
    [
        (81, 25, 0, "passed"),
        (82, 25, 0, "failed"),
        (81, 26, 0, "failed"),
        (81, 25, 1, "failed"),
    ],
)
def test_clip000_thresholds_apply_to_all_cameras_together(
    unavailable, unknown, unprocessed, status
):
    report = {
        "scope": {c: [0, 1010] for c in ("cam0", "cam1", "cam2")},
        "totals": {
            "total": 3030,
            "position_unavailable": unavailable,
            "final_unknown": unknown,
            "unprocessed": unprocessed,
        },
    }
    assert clip000_acceptance(report)["status"] == status


def test_report_keeps_states_estimates_unknown_and_unprocessed_separate(
    tmp_path: Path,
) -> None:
    fixture = make_import_fixture(tmp_path)
    annotations = [read_annotation(p) for p in fixture.annotations.annotations.values()]
    # Model response contains one unfinished frame, not a final unknown judgment.
    row = (
        annotations[0]
        .frames[0]
        .model_copy(
            update={"state": "unprocessed", "position": None, "occlusion": "unknown"}
        )
    )
    annotations[0] = annotations[0].model_copy(
        update={"frames": (row, *annotations[0].frames[1:])}
    )
    far = annotations[2]
    annotations[2] = far.model_copy(
        update={
            "frames": (
                far.frames[0].model_copy(
                    update={"state": "in_frame", "occlusion": "occluded"}
                ),
                far.frames[1].model_copy(
                    update={"state": "out_of_frame", "occlusion": "not_applicable"}
                ),
                *far.frames[2:],
            )
        }
    )
    report = summarize_video_annotations(
        annotations, expected_inputs=tuple(fixture.annotations.expected_inputs.values())
    )
    totals = report["totals"]
    assert (
        totals["total"],
        totals["position_unavailable"],
        totals["final_unknown"],
        totals["unprocessed"],
    ) == (12, 4, 1, 1)
    assert (
        totals["occluded_without_position"]
        == totals["out_of_frame"]
        == totals["no_active_ball"]
        == 1
    )
    assert (totals["direct_observation"], totals["estimated"]) == (3, 4)
    assert report["cameras"]["far"]["states"] == {
        "out_of_frame": 1,
        "in_frame": 2,
        "no_active_ball": 1,
    }
    assert report["cameras"]["near"]["unknown_intervals"] == [[3, 4]]
    assert report["cameras"]["right"]["unprocessed_intervals"] == [[0, 1]]
    assert clip000_acceptance(report)["status"] == "not_applicable"
    with pytest.raises(ValueError, match="camera/input set"):
        summarize_video_annotations(
            annotations[:-1],
            expected_inputs=tuple(fixture.annotations.expected_inputs.values()),
        )


def test_reference_counts_final_kind_instead_of_precompletion_source_unknown():
    import numpy as np

    inputs = make_inputs(frames=4)
    uv = np.full((3, 4, 2), 10.0)
    kinds = np.array([["observed", "estimated", "unknown", "occluded"]] * 3)
    reference = make_annotations(inputs, uv, kinds).reference
    # Accepted estimated coordinates retain the original source_status=unknown.
    positions = {
        k: replace(r, source_status="unknown") if r.reference_kind == "estimated" else r
        for k, r in reference.positions.items()
    }
    report = summarize_reference(replace(reference, positions=positions))
    assert report["totals"] == {
        "total": 12,
        "position_usable": 6,
        "position_unavailable": 6,
        "final_unknown": 3,
        "unprocessed": 0,
    }
