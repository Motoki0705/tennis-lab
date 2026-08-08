"""Hydra composition coverage for the canonical scene-pipeline hierarchy."""

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_dir

from src.synthetic_data_generation.alignment.contracts import AlignmentAcceptancePolicy
from src.synthetic_data_generation.alignment.settings import AlignmentEvidenceSettings
from src.synthetic_data_generation.configuration import ScenePipelineConfiguration
from src.synthetic_data_generation.dataset.blcs.rendering import BLCSNHTRenderer
from src.synthetic_data_generation.dataset.blcs.source import (
    PhysicsBLCSTrajectoryProvider,
)
from src.synthetic_data_generation.dataset.plcs.handler import PLCSStageParameters
from src.synthetic_data_generation.dataset.plcs.rendering import NHTPLCSRenderer
from src.synthetic_data_generation.reconstruction import NHTReconstructionHandler
from src.synthetic_data_generation.rendering.nht import NHTRenderClient
from src.synthetic_data_generation.scene_contract import CourtInstance, RigidTransform
from src.tasks.base.generate_dataset.camera_profiles import sample_camera_rig
from src.utils.configuration import PathRole
from src.utils.paths import PROJECT_ROOT

_CONFIG_ROOT = PROJECT_ROOT / "src/synthetic_data_generation/configs"
_NHT_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "0",
}

pytestmark = pytest.mark.local_data


def _compose(*overrides: str) -> ScenePipelineConfiguration:
    with initialize_config_dir(version_base="1.3", config_dir=str(_CONFIG_ROOT)):
        config = compose(config_name="run_scene_pipeline", overrides=list(overrides))
    return ScenePipelineConfiguration.from_config(config)


def test_default_profile_composes_exactly_six_shared_camera_slots() -> None:
    runtime = _compose()

    assert runtime.camera.profile == "default"
    assert runtime.camera.expected_camera_count == 6
    assert len(runtime.camera.slots) == 6


def test_broadcast_profile_composes_exactly_two_shared_camera_slots() -> None:
    runtime = _compose("camera=broadcast")

    assert runtime.camera.profile == "broadcast"
    assert runtime.camera.expected_camera_count == 2
    assert len(runtime.camera.slots) == 2


def test_public_nht_commands_are_installed_names_without_provider_knowledge() -> None:
    runtime = _compose()

    assert runtime.nht.reconstruct_executable == "nht-reconstruct"
    assert runtime.nht.render_executable == "nht-render"
    assert runtime.nht.environment == _NHT_ENVIRONMENT
    assert runtime.nht.reconstruction_timeout_seconds == 86_400.0
    assert runtime.nht.render_timeout_seconds == 3_600.0
    assert not hasattr(runtime.nht, "sha256")
    assert not hasattr(runtime.nht, "commit")
    assert not hasattr(runtime.nht, "repository_root")
    assert not hasattr(runtime.nht, "reconstruction_config_path")
    assert not hasattr(runtime.nht, "training_runtime")


def test_configured_paths_retain_their_declared_runtime_roles() -> None:
    runtime = _compose()
    line_model = runtime.alignment.evidence.line_model

    assert runtime.resolver.validate(
        PathRole.CHECKPOINT,
        line_model.checkpoint_path,
    ) == line_model.checkpoint_path
    for path in (
        line_model.backbone_repository_path,
        line_model.backbone_checkpoint_path,
        runtime.plcs.accad_root,
        runtime.plcs.smplh_model_root,
    ):
        assert runtime.resolver.validate(PathRole.EXTERNAL_ASSET, path) == path


def test_composition_root_can_construct_each_no_default_runtime_input() -> None:
    runtime = _compose()
    client = NHTRenderClient()

    reconstruction = NHTReconstructionHandler(
        executable=runtime.nht.reconstruct_executable,
        environment=runtime.nht.environment,
        timeout_seconds=runtime.nht.reconstruction_timeout_seconds,
    )
    assert reconstruction.environment == _NHT_ENVIRONMENT

    assert isinstance(runtime.alignment.evidence, AlignmentEvidenceSettings)
    assert isinstance(runtime.alignment.acceptance, AlignmentAcceptancePolicy)

    provider = PhysicsBLCSTrajectoryProvider(
        generator_config=runtime.blcs.generator,
        settings=runtime.blcs.trajectory_source,
    )
    blcs_renderer = BLCSNHTRenderer(
        assets=runtime.blcs.assets,
        client=client,
        executable=runtime.nht.render_executable,
        environment=runtime.nht.environment,
        timeout_seconds=runtime.blcs.render_timeout_seconds,
        execution_device=runtime.blcs.performance.execution_device,
        maximum_batch_frames=runtime.blcs.performance.maximum_batch_frames,
    )
    assert provider.settings.timeline.num_frames == 1024
    assert blcs_renderer.timeout_seconds == 3_600.0

    parameters = runtime.plcs.build_stage_parameters(seed=runtime.stages.seed)
    assert isinstance(parameters, PLCSStageParameters)
    assert len(parameters.objects) == 3
    assert parameters.scene_splits == {
        "B00": "train",
        "B00-plcs-002": "train",
    }
    assert runtime.plcs.performance.maximum_background_cache_misses == 12
    assert runtime.plcs.performance.maximum_nht_invocations == 1
    plcs_renderer = NHTPLCSRenderer(
        client=client,
        compositor=runtime.plcs.foreground_compositor,
        executable=runtime.nht.render_executable,
        environment=runtime.nht.environment,
        timeout_seconds=runtime.plcs.render_timeout_seconds,
    )
    assert plcs_renderer.compositor is runtime.plcs.foreground_compositor


def test_task_local_camera_config_copies_are_removed() -> None:
    for task in ("blcs", "plcs"):
        camera_root = PROJECT_ROOT / f"src/tasks/{task}/configs/camera"
        assert not list(camera_root.glob("*.yaml"))


def test_camera_sampling_is_deterministic_and_within_composed_slot_bounds() -> None:
    runtime = _compose()
    identity = RigidTransform.identity()
    court = CourtInstance(
        court_instance_id="court-0",
        candidate_id="candidate-0",
        scene_from_court=identity,
        court_from_scene=identity,
        fit_status="accepted",
        fit_metrics={"score": 1.0},
        holdout_status="accepted",
        holdout_metrics={"score": 1.0},
    )

    first = sample_camera_rig(runtime.camera, seed=runtime.stages.seed, court=court)
    second = sample_camera_rig(runtime.camera, seed=runtime.stages.seed, court=court)

    assert first == second
    for sampled, slot in zip(first.cameras, runtime.camera.slots, strict=True):
        x, y, height = sampled.court_local_center_m
        look_x, look_y, look_height = sampled.court_local_look_at_m
        assert slot.position_x_m[0] <= x <= slot.position_x_m[1]
        assert slot.position_y_m[0] <= y <= slot.position_y_m[1]
        assert slot.height_m[0] <= height <= slot.height_m[1]
        assert slot.look_at_x_m[0] <= look_x <= slot.look_at_x_m[1]
        assert slot.look_at_y_m[0] <= look_y <= slot.look_at_y_m[1]
        assert slot.look_at_height_m[0] <= look_height <= slot.look_at_height_m[1]
        assert slot.hfov_degrees[0] <= sampled.hfov_degrees <= slot.hfov_degrees[1]
