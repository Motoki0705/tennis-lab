"""Strict configuration tests for integrated Court reference inference."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.utils.configuration import SemanticConfigurationError


def _runtime(overrides: list[str]) -> PipelineRuntimeConfig:
    config_dir = Path(__file__).parents[3] / "src/tennis_scene/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(config_name="pipeline", overrides=overrides)
    return PipelineRuntimeConfig.from_config(config)


def test_pipeline_defaults_to_physical_court_reference() -> None:
    runtime = _runtime([])
    assert runtime.plcs.court_keypoint_contract.selector == "physical_v1"
    assert runtime.blcs.court_keypoint_contract == runtime.plcs.court_keypoint_contract
    assert runtime.court_reference.reference_camera is None
    assert runtime.gvhmr.court_footpoint_filter.enabled is False


def test_pipeline_composes_shared_camera_view_reference() -> None:
    runtime = _runtime(
        [
            "court_keypoints.selector=camera_view_v2",
            "court_reference.reference_camera=cam0",
            "court_reference.view_half_turns=[false,false,true]",
        ]
    )
    assert runtime.plcs.court_keypoint_contract.selector == "camera_view_v2"
    assert runtime.blcs.court_keypoint_contract == runtime.plcs.court_keypoint_contract
    assert runtime.court_reference.reference_camera == "cam0"
    assert runtime.court_reference.view_half_turns == (False, False, True)


def test_camera_view_reference_requires_explicit_orientation() -> None:
    with pytest.raises(SemanticConfigurationError, match="camera_view_v2 requires"):
        _runtime(["court_keypoints.selector=camera_view_v2"])


def test_court_footpoint_filter_requires_dino_detector() -> None:
    with pytest.raises(
        SemanticConfigurationError,
        match="court_footpoint_filter requires detector='dino'",
    ):
        _runtime(
            [
                "gvhmr.detector=yolo",
                "gvhmr.court_footpoint_filter.enabled=true",
            ]
        )
