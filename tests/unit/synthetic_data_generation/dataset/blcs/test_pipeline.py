"""Tests for BLCS algorithm selection and stage ownership."""

from __future__ import annotations

import pytest

from src.synthetic_data_generation.dataset.blcs.pipeline import BLCSDatasetPipeline


def test_blcs_pipeline_selects_procedural_asset_worker() -> None:
    plan = BLCSDatasetPipeline().build_plan(
        {
            "algorithms": {
                "ball_asset": "procedural_fibonacci",
                "trajectory": "rally_physics",
            },
            "stages": {
                "prepare_assets": {
                    "enabled": True,
                    "arguments": ["--output-dir", "/tmp/new"],
                }
            },
        }
    )

    assert plan.selected_algorithms["ball_asset"] == "procedural_fibonacci"
    assert plan.commands[0].module.endswith("procedural_ball_asset_builder")
    assert plan.commands[0].runtime == "nht"


def test_blcs_pipeline_rejects_unknown_asset_algorithm() -> None:
    with pytest.raises(ValueError, match="Unknown blcs.ball_asset algorithm"):
        BLCSDatasetPipeline().build_plan(
            {
                "algorithms": {
                    "ball_asset": "silent-fallback",
                    "trajectory": "rally_physics",
                },
                "stages": {},
            }
        )
