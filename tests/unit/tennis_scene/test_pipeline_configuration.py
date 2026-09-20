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


def test_pipeline_defaults_to_camera_view_models_with_explicit_orientation() -> None:
    runtime = _runtime(["court_reference.view_half_turns=[false,false,true]"])
    assert runtime.plcs.court_keypoint_contract.selector == "camera_view_v2"
    assert runtime.blcs.court_keypoint_contract == runtime.plcs.court_keypoint_contract
    assert runtime.court_kp.output_keypoint_contract == "camera_view_v2"
    assert runtime.plcs.checkpoint.name == "real-rgb-meiji-foot-e60-v1.ckpt"
    assert runtime.blcs.checkpoint.name == "real-rgb-meiji-e60-v1.ckpt"
    assert runtime.plcs.window_size == runtime.blcs.window_size == 128
    assert len(runtime.camera_ids) == 3
    assert runtime.court_reference.reference_camera == "cam0"
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
                "court_reference.view_half_turns=[false,false,true]",
                "gvhmr.detector=yolo",
                "gvhmr.court_footpoint_filter.enabled=true",
            ]
        )


def test_pipeline_outputs_share_identity_across_distinct_roots(tmp_path: Path) -> None:
    runtime = _runtime(
        [
            "court_reference.view_half_turns=[false,false,true]",
            f"paths.output_root={tmp_path / 'runs'}",
            f"paths.artifact_root={tmp_path / 'stage-artifacts'}",
            "output_directory=tennis_scene/generate/smoke/seed42",
            "output_name=clip",
        ]
    )
    relative = Path("tennis_scene/generate/smoke/seed42")
    assert runtime.output_path == tmp_path / "runs" / relative / "clip.npz"
    for stage in (
        runtime.court_kp,
        runtime.gvhmr,
        runtime.player_association,
        runtime.ball_detection,
        runtime.plcs,
        runtime.blcs,
    ):
        assert stage.output_path.parent == tmp_path / "stage-artifacts" / relative


@pytest.mark.parametrize("fragment", ["../escape", "/tmp/escape", "outputs/repeated"])
def test_pipeline_rejects_invalid_output_directory(fragment: str) -> None:
    from src.utils.configuration.errors import PathContractError

    with pytest.raises(PathContractError):
        _runtime(
            [
                "court_reference.view_half_turns=[false,false,true]",
                f"output_directory={fragment}",
            ]
        )


@pytest.mark.parametrize(
    "overrides",
    [
        ["court_reference.view_half_turns=[false]"],
        [
            "court_reference.view_half_turns=[false,false,true]",
            "court_reference.reference_camera=missing",
        ],
        ["court_reference.view_half_turns=[true,false,true]"],
    ],
)
def test_pipeline_rejects_invalid_reference_declarations_before_loading(
    overrides: list[str],
) -> None:
    with pytest.raises(SemanticConfigurationError):
        _runtime(overrides)


@pytest.mark.parametrize("declaration", ["camera_view_v2", "physical_v1"])
def test_pipeline_parses_explicit_legacy_input_contract(declaration: str) -> None:
    overrides = [
        "court_reference.view_half_turns=[false,false,true]",
        "court_kp.source=load",
        "court_kp.load_path=manual.json",
        f"court_kp.load_keypoint_contract={declaration}",
    ]
    if declaration == "physical_v1":
        with pytest.raises(ValueError, match="must match"):
            _runtime(overrides)
    else:
        runtime = _runtime(overrides)
        assert runtime.court_kp.load_keypoint_contract == "camera_view_v2"
