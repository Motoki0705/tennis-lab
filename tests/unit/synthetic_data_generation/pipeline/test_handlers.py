"""Typed report projection and report-only recovery tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from src.synthetic_data_generation.dataset.court.assembler import (
    CourtAssemblyReport,
)
from src.synthetic_data_generation.pipeline.contracts import (
    DatasetTarget,
    ScenePipelineRequest,
    StageName,
)
from src.synthetic_data_generation.pipeline.handlers import (
    _court_report_manifest,
)


def _target_court(court_id: str, *, x_metres: float = 0.0) -> dict[str, object]:
    return {
        "court_instance_id": court_id,
        "candidate_id": f"candidate-{court_id}",
        "scene_from_court": [
            1.0,
            0.0,
            0.0,
            x_metres,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "selection_seed": 695,
    }


def _court_report() -> CourtAssemblyReport:
    return cast(
        CourtAssemblyReport,
        SimpleNamespace(
            proposal_count=2400,
            accepted_frame_count=2200,
            rejected_frame_count=200,
            trajectory_group_count=24,
        ),
    )


def test_court_report_projection_deduplicates_validated_target_courts() -> None:
    payload = {
        "scene_id": "B00",
        "schema": "court_dataset_v2",
        "profile": "production",
        "trajectory_groups": [
            {"target_court": _target_court("court-000")},
            {"target_court": _target_court("court-001", x_metres=20.0)},
            {"target_court": _target_court("court-000")},
        ],
        "diagnostics": ["diagnostics/summary.json"],
    }

    manifest = _court_report_manifest(payload, report=_court_report())

    assert manifest.domain.value == "court"
    assert manifest.frame_inventory.to_dict()["source"] == 2200
    assert [binding.court_instance_id for binding in manifest.target_courts] == [
        "court-000",
        "court-001",
    ]


def test_court_report_projection_rejects_conflicting_target_binding() -> None:
    payload = {
        "scene_id": "B00",
        "schema": "court_dataset_v2",
        "profile": "production",
        "trajectory_groups": [
            {"target_court": _target_court("court-000")},
            {"target_court": _target_court("court-000", x_metres=1.0)},
        ],
        "diagnostics": ["diagnostics/summary.json"],
    }

    with pytest.raises(ValueError, match="disagree"):
        _court_report_manifest(payload, report=_court_report())


def test_report_stage_is_a_valid_failure_recovery_cursor(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")

    request = ScenePipelineRequest(
        scene_id="B00",
        source_video=source.resolve(),
        targets=frozenset(DatasetTarget),
        from_stage=StageName.REPORT,
        config_schema="canonical_scene_pipeline_v1",
    )

    assert request.from_stage is StageName.REPORT
