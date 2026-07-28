"""Tests for PLCS algorithm selection and stage ownership."""

from __future__ import annotations

from src.synthetic_data_generation.dataset.plcs.pipeline import PLCSDatasetPipeline


def test_plcs_pipeline_keeps_avatar_algorithm_in_plan() -> None:
    plan = PLCSDatasetPipeline().build_plan(
        {
            "algorithms": {
                "avatar_control": "hugs_topk_lbs",
                "motion": "seeded_court_motion",
            },
            "stages": {
                "render": {
                    "enabled": True,
                    "arguments": ["--width", "480"],
                }
            },
        }
    )

    assert plan.selected_algorithms["avatar_control"] == "hugs_topk_lbs"
    assert plan.commands[0].module.endswith("dataset.plcs.rendering.nht")
    assert plan.commands[0].runtime == "nht"
