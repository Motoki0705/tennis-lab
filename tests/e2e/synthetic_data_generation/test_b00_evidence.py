"""Schema and acceptance gates for committed B00 production evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from src.synthetic_data_generation.dataset.court.components.labels import (
    SEMANTIC_CLASS_NAMES,
)
from src.utils.paths import PROJECT_ROOT

EVIDENCE_PATH = PROJECT_ROOT / "research/evidence/issue-695-b00-acceptance.json"
SUMMARY_PATH = PROJECT_ROOT / "research/evidence/issue-695-b00/summary.md"


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object.")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings.")
    return value


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    return float(value)


def _object(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value, name=f"Evidence at {path}")


def test_committed_b00_evidence_is_complete_and_meets_quantitative_gates() -> None:
    evidence = _object(EVIDENCE_PATH)

    assert set(evidence) == {
        "schema",
        "issue",
        "scene_id",
        "source_video",
        "pipeline",
        "reconstruction",
        "alignment",
        "court",
        "blcs",
        "plcs",
        "camera_profiles",
    }
    assert evidence["schema"] == "issue_695_b00_acceptance_v1"
    assert evidence["issue"] == 695
    assert evidence["scene_id"] == "B00"
    assert evidence["source_video"] == "data/tennis_court.mp4"
    pipeline = _mapping(evidence["pipeline"], name="pipeline")
    stage_statuses = _mapping(
        pipeline["stage_statuses"], name="pipeline.stage_statuses"
    )
    assert all(status == "completed" for status in stage_statuses.values())
    assert pipeline["report_exists"] is True
    reconstruction = _mapping(evidence["reconstruction"], name="reconstruction")
    assert _integer(reconstruction["camera_count"], name="camera_count") >= 12
    assert _integer(reconstruction["point_count"], name="point_count") > 0
    alignment = _mapping(evidence["alignment"], name="alignment")
    assert _integer(alignment["accepted_court_count"], name="accepted_court_count") >= 2
    assert alignment["fit_holdout_disjoint"] is True
    assert alignment["all_transforms_reciprocal"] is True

    court = _mapping(evidence["court"], name="court")
    assert (
        _integer(court["trajectory_group_count"], name="trajectory_group_count") >= 24
    )
    assert _integer(court["accepted_frame_count"], name="accepted_frame_count") >= 2000
    assert _integer(court["proposal_count"], name="proposal_count") <= 5000
    assert _number(court["accepted_fraction"], name="accepted_fraction") >= 0.9
    assert (
        _number(court["maximum_adjacent_step_m"], name="maximum_adjacent_step_m")
        <= 1.05
    )
    assert _integer(court["split_leakage_count"], name="split_leakage_count") == 0
    visible_points = _mapping(
        court["renderer_visible_points_by_class"],
        name="renderer_visible_points_by_class",
    )
    assert set(visible_points) == set(SEMANTIC_CLASS_NAMES)
    assert all(
        _integer(value, name=f"renderer_visible_points_by_class.{name}") > 0
        for name, value in visible_points.items()
    )

    for domain in ("blcs", "plcs"):
        domain_evidence = _mapping(evidence[domain], name=domain)
        inventory = _mapping(
            domain_evidence["frame_inventory"], name=f"{domain}.frame_inventory"
        )
        assert (
            len(
                {
                    _integer(inventory["source"], name=f"{domain}.source"),
                    _integer(inventory["planned"], name=f"{domain}.planned"),
                    _integer(inventory["rendered"], name=f"{domain}.rendered"),
                    _integer(inventory["labelled"], name=f"{domain}.labelled"),
                }
            )
            == 1
        )
        assert (
            _integer(
                domain_evidence["court_count_difference"],
                name=f"{domain}.court_count_difference",
            )
            <= 1
        )
    plcs = _mapping(evidence["plcs"], name="plcs")
    assert plcs["motion_categories"] == [
        "general",
        "running",
        "walking",
    ]
    assert plcs["all_frames_deformed"] is True
    assert plcs["running_walking_regions_move"] is True
    assert evidence["camera_profiles"] == {"default": 6, "broadcast": 2}


def test_human_summary_is_present_and_identifies_the_same_result() -> None:
    evidence = _object(EVIDENCE_PATH)
    summary = SUMMARY_PATH.read_text(encoding="utf-8")
    court = _mapping(evidence["court"], name="court")
    blcs = _mapping(evidence["blcs"], name="blcs")
    blcs_inventory = _mapping(blcs["frame_inventory"], name="blcs.frame_inventory")
    plcs = _mapping(evidence["plcs"], name="plcs")
    plcs_inventory = _mapping(plcs["frame_inventory"], name="plcs.frame_inventory")

    assert "Issue #695 B00 acceptance" in summary
    assert f"Court accepted frames: {court['accepted_frame_count']}" in summary
    assert f"BLCS source frames: {blcs_inventory['source']}" in summary
    assert f"PLCS global frames: {plcs_inventory['source']}" in summary
    assert "SMPL-H Gaussian LBS" in summary
