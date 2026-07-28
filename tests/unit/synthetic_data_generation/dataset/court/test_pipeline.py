"""Tests for court camera algorithm selection and stage ownership."""

from __future__ import annotations

from src.synthetic_data_generation.dataset.court.pipeline import CourtDatasetPipeline


def test_court_pipeline_selects_sfm_and_orbit_workers() -> None:
    base = {
        "labels": "symmetric_seven_channel",
    }
    stages = {
        "sample_cameras": {
            "enabled": True,
            "arguments": ["--seed", "1"],
        }
    }
    sfm = CourtDatasetPipeline().build_plan(
        {
            "algorithms": {**base, "camera_sampling": "sfm_neighborhood"},
            "stages": stages,
        }
    )
    orbit = CourtDatasetPipeline().build_plan(
        {
            "algorithms": {**base, "camera_sampling": "inward_orbit"},
            "stages": stages,
        }
    )

    assert sfm.commands[0].module.endswith("camera_sampling.support_probe")
    assert orbit.commands[0].module.endswith("camera_sampling.orbit_plan")
