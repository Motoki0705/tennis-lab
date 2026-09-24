"""Strict, headless automatic pipeline configuration and independent data inputs."""

from pathlib import Path
from typing import Any, cast

import pytest
from hydra import compose, initialize_config_dir

from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.utils.configuration.errors import PathContractError


def _runtime(
    overrides: list[str], *, bind_inputs: bool = True
) -> PipelineRuntimeConfig:
    directory = Path(__file__).parents[3] / "src/tennis_scene/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(directory)):
        cfg = compose(config_name="pipeline", overrides=overrides)
    return PipelineRuntimeConfig.from_config(cfg, bind_inputs=bind_inputs)


def test_default_requires_no_side_annotation() -> None:
    runtime = _runtime([])
    assert runtime.court_kp.output_keypoint_contract == "camera_view_v2"
    assert runtime.court_kp.mode == "model"
    assert runtime.camera_geometry.reference_camera is None
    assert runtime.plcs_reid_checkpoint.name == "player-reid-v1.ckpt"
    assert runtime.court_side_checkpoint.name == "court-side-v1.ckpt"
    assert "blcs_association" not in runtime.enabled
    assert len(runtime.camera_ids) == 3
    assert runtime.player_placement.temporal_weight == 0.05
    assert runtime.player_placement.min_joints == 5


def test_placement_override_is_typed_and_part_of_cache_settings() -> None:
    value = _runtime(["player_reconstruction.placement.temporal_weight=0.2"])
    assert value.player_placement.temporal_weight == 0.2
    assert (
        cast(dict[str, Any], value.processing_settings["player_reconstruction"])[
            "placement"
        ]["temporal_weight"]
        == 0.2
    )


@pytest.mark.parametrize(
    "override",
    [
        "temporal_weight=-1",
        "data_sigma_m=.nan",
        "min_scale=2",
        "max_nfev=0",
        "min_joints=18",
    ],
)
def test_invalid_placement_is_rejected_before_model_loading(override: str) -> None:
    with pytest.raises(ValueError):
        _runtime([f"player_reconstruction.placement.{override}"])


def test_unknown_placement_parameter_is_rejected() -> None:
    with pytest.raises(ValueError):
        _runtime(["+player_reconstruction.placement.body_pose_weight=1"])


def test_stage_only_composition_does_not_bind_dummy_video_inputs() -> None:
    runtime = _runtime(
        [
            "camera_ids=[unused,unused]",
            "video_paths=[]",
            "camera_geometry.reference_camera=actual_camera",
        ],
        bind_inputs=False,
    )
    assert runtime.camera_ids == () and runtime.video_paths == ()
    assert runtime.camera_geometry.reference_camera == "actual_camera"


def test_pipeline_outputs_and_stage_cache_use_their_declared_roots(
    tmp_path: Path,
) -> None:
    runtime = _runtime(
        [
            f"paths.output_root={tmp_path / 'runs'}",
            f"paths.artifact_root={tmp_path / 'stage-artifacts'}",
            "output_directory=tennis_scene/generate/integration/s42",
            "output_name=clip",
        ]
    )
    relative = Path("tennis_scene/generate/integration/s42")
    assert runtime.output_path == tmp_path / "runs" / relative / "clip.npz"
    assert runtime.cache_directory == tmp_path / "stage-artifacts" / relative / "stages"


@pytest.mark.parametrize("fragment", ["../escape", "/tmp/escape", "outputs/repeated"])
def test_output_paths_cannot_escape_authority(fragment: str) -> None:
    with pytest.raises(PathContractError):
        _runtime([f"output_directory={fragment}"])


@pytest.mark.parametrize(
    "overrides",
    [
        ["video_paths=[a.mp4,b.mp4]", "camera_ids=[a,b]"],
        ["camera_ids=[a,a,c]"],
        ["person_observations.enabled=false"],
        ["people_models.detector=yolo"],
        ["people_models.runtime.static_cam=false"],
        ["cache.source=load", "cache.overwrite=true"],
    ],
)
def test_invalid_inputs_or_dependency_choices_fail_before_models(
    overrides: list[str],
) -> None:
    with pytest.raises(ValueError):
        _runtime(overrides)
