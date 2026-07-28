"""Tests for deterministic dataset pipeline plans."""

from __future__ import annotations

import json

import pytest

from src.synthetic_data_generation.dataset.pipeline import (
    DatasetPipelinePlan,
    PipelineCommand,
)


def test_pipeline_plan_round_trip_and_fingerprint(tmp_path) -> None:
    plan = DatasetPipelinePlan(
        dataset="court",
        selected_algorithms={
            "camera_sampling": "inward_orbit",
            "labels": "symmetric_seven_channel",
        },
        commands=(
            PipelineCommand(
                stage="render",
                runtime="nht",
                module="src.synthetic_data_generation.dataset.court.rendering.nht",
                arguments=("--width", "320"),
            ),
        ),
    )
    path = tmp_path / "plan.json"
    plan.write(path)
    loaded = DatasetPipelinePlan.from_dict(json.loads(path.read_text()))

    assert loaded == plan
    assert len(plan.plan_fingerprint) == 64


def test_pipeline_plan_rejects_duplicate_stages_and_changed_fingerprint() -> None:
    command = PipelineCommand(
        stage="render",
        runtime="nht",
        module="src.synthetic_data_generation.dataset.court.rendering.nht",
        arguments=(),
    )
    with pytest.raises(ValueError, match="must be unique"):
        DatasetPipelinePlan(
            dataset="court",
            selected_algorithms={"camera_sampling": "inward_orbit"},
            commands=(command, command),
        )
    with pytest.raises(ValueError, match="fingerprint differs"):
        DatasetPipelinePlan(
            dataset="court",
            selected_algorithms={"camera_sampling": "inward_orbit"},
            commands=(command,),
            plan_fingerprint="0" * 64,
        )
