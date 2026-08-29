"""Hydra composition coverage for the canonical scene-pipeline hierarchy."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.synthetic_data_generation.alignment.contracts import AlignmentAcceptancePolicy
from src.synthetic_data_generation.alignment.settings import AlignmentEvidenceSettings
from src.synthetic_data_generation.configuration import ScenePipelineConfiguration
from src.synthetic_data_generation.dataset.blcs.contracts import BLCSBallRendering
from src.synthetic_data_generation.dataset.blcs.rendering import BLCSNHTRenderer
from src.synthetic_data_generation.dataset.blcs.source import (
    PhysicsBLCSTrajectoryProvider,
)
from src.synthetic_data_generation.dataset.camera_profiles import sample_camera_rig
from src.synthetic_data_generation.dataset.court.contracts import (
    OrbitCenterKind,
    OrbitCoverageMode,
    OrbitCoverageObjective,
    OrbitCurveMode,
    OrbitSamplingMode,
    OrbitShape,
    OrbitStableField,
    OrbitTargetMode,
)
from src.synthetic_data_generation.dataset.court.schema import (
    CourtDatasetSchemaVersion,
)
from src.synthetic_data_generation.dataset.plcs.handler import PLCSStageParameters
from src.synthetic_data_generation.dataset.plcs.production import PLCSProductionMode
from src.synthetic_data_generation.dataset.plcs.rendering import NHTPLCSRenderer
from src.synthetic_data_generation.reconstruction import NHTReconstructionHandler
from src.synthetic_data_generation.rendering.nht import (
    NHTComposedRenderClient,
    NHTRenderClient,
)
from src.synthetic_data_generation.scene_contract import CourtInstance, RigidTransform
from src.utils.configuration import (
    PathRole,
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)
from src.utils.paths import PROJECT_ROOT

_CONFIG_ROOT = PROJECT_ROOT / "src/synthetic_data_generation/configs"
_NHT_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "0",
}

pytestmark = pytest.mark.local_data


def _resource_repository_root() -> Path:
    for candidate in (PROJECT_ROOT, *PROJECT_ROOT.parents):
        if (
            candidate / "data/synthetic_data_generation/raw/B00.mp4"
        ).is_file() and (
            candidate / "third_party/nht/configs/production.yaml"
        ).is_file():
            return Path(candidate)
    raise FileNotFoundError("Canonical synthetic-data local resources are unavailable.")


def _compose(*overrides: str) -> ScenePipelineConfiguration:
    resource_root = _resource_repository_root()
    with initialize_config_dir(version_base="1.3", config_dir=str(_CONFIG_ROOT)):
        config = compose(
            config_name="run_scene_pipeline",
            overrides=[
                *overrides,
                f"roots.data_root={(resource_root / 'data').as_posix()}",
                f"roots.checkpoint_root={(resource_root / 'ckpt').as_posix()}",
                (
                    "roots.external_asset_root="
                    f"{(resource_root / 'third_party').as_posix()}"
                ),
            ],
        )
    return ScenePipelineConfiguration.from_config(config)


def test_default_profile_composes_exactly_six_shared_camera_slots() -> None:
    runtime = _compose()

    assert runtime.camera.profile == "default"
    assert runtime.camera.expected_camera_count == 6
    assert len(runtime.camera.slots) == 6


def test_blcs_glb_ball_is_an_explicit_data_root_relative_option() -> None:
    runtime = _compose(
        "dataset.blcs.assets.rendering=mesh",
        (
            'dataset.blcs.assets.mesh.path="synthetic_data_generation/assets/blcs/'
            'tennis ball 3d model.glb"'
        ),
    )

    assert runtime.blcs.assets.rendering is BLCSBallRendering.MESH
    assert runtime.blcs.assets.mesh is not None
    assert runtime.blcs.assets.mesh.data_root_relative_path == (
        "synthetic_data_generation/assets/blcs/tennis ball 3d model.glb"
    )
    assert runtime.blcs.assets.mesh.maximum_file_bytes == 33554432
    assert runtime.blcs.assets.mesh.maximum_source_vertices == 500000
    assert runtime.blcs.assets.mesh.maximum_source_faces == 1000000
    assert runtime.blcs.assets.mesh.maximum_faces == 4096
    assert (
        runtime.blcs.assets.mesh.path
        == (
            runtime.resolver.roots.data_root
            / "synthetic_data_generation/assets/blcs/tennis ball 3d model.glb"
        ).resolve()
    )
    assert runtime.blcs.assets.settings.radius_m == 0.0335


def test_blcs_gaussian_ball_remains_the_default_without_mesh_fallback() -> None:
    assets = _compose().blcs.assets

    assert assets.rendering is BLCSBallRendering.GAUSSIAN
    assert assets.mesh is None


def test_blcs_mesh_mode_never_falls_back_when_glb_path_is_missing() -> None:
    with pytest.raises(TypeError, match="data-root-relative string"):
        _compose("dataset.blcs.assets.rendering=mesh")
    with pytest.raises(FileNotFoundError, match="ordinary existing"):
        _compose(
            "dataset.blcs.assets.rendering=mesh",
            "dataset.blcs.assets.mesh.path=synthetic_data_generation/assets/blcs/missing.glb",
        )


def test_broadcast_profile_composes_exactly_two_shared_camera_slots() -> None:
    runtime = _compose("camera=broadcast")

    assert runtime.camera.profile == "broadcast"
    assert runtime.camera.expected_camera_count == 2
    assert len(runtime.camera.slots) == 2


@pytest.mark.parametrize(
    ("selector", "version", "target_modes"),
    [
        (
            "dataset/court=v1",
            CourtDatasetSchemaVersion.V1,
            set(OrbitTargetMode),
        ),
        (
            "dataset/court=v2",
            CourtDatasetSchemaVersion.V2,
            {OrbitTargetMode.COURT_CENTER},
        ),
    ],
)
def test_court_selectors_compose_and_validate_exact_typed_versions(
    selector: str,
    version: CourtDatasetSchemaVersion,
    target_modes: set[OrbitTargetMode],
) -> None:
    court = _compose(selector).court

    assert court.schema_version is version
    assert set(court.trajectory.shapes) == set(OrbitShape)
    assert set(court.trajectory.center_kinds) == set(OrbitCenterKind)
    assert set(court.trajectory.curve_modes) == set(OrbitCurveMode)
    assert set(court.view.target_modes) == target_modes
    assert set(court.view.coverage_modes) == set(OrbitCoverageMode)
    assert court.sampling.mode is OrbitSamplingMode.UNIFORM_ARC_LENGTH
    assert set(court.sampling.stable_field_order) == set(OrbitStableField)
    assert set(court.sampling.coverage_objective) == set(OrbitCoverageObjective)


def test_default_and_compatibility_train_selectors_remain_exact_v1() -> None:
    default = _compose().court
    compatibility = _compose("dataset/court=train").court

    assert default.schema_version is CourtDatasetSchemaVersion.V1
    assert compatibility.schema_version is CourtDatasetSchemaVersion.V1
    assert default == compatibility
    assert tuple(default.view.target_modes) == (
        OrbitTargetMode.COURT_CENTER,
        OrbitTargetMode.COMPLEX_CENTER,
        OrbitTargetMode.NEAR_BASELINE,
        OrbitTargetMode.FAR_BASELINE,
    )


@pytest.mark.parametrize(
    "override",
    [
        "dataset.court.schema_version=v3",
        "dataset/court=v1",
        "dataset/court=v2",
    ],
)
def test_version_or_version_specific_target_mismatch_fails_closed(
    override: str,
) -> None:
    extra = {
        "dataset/court=v1": "dataset.court.view.target_modes=[court_center]",
        "dataset/court=v2": (
            "dataset.court.view.target_modes=[court_center,complex_center]"
        ),
    }.get(override)
    overrides = (override,) if extra is None else (override, extra)
    with pytest.raises(SemanticConfigurationError):
        _compose(*overrides)


@pytest.mark.parametrize(
    "override",
    [
        "dataset.court.trajectory.shapes=[circle,unknown_shape]",
        "dataset.court.trajectory.center_kinds=[complex,unknown_center]",
        "dataset.court.trajectory.curve_modes=[planar,unknown_curve]",
        "dataset.court.view.target_modes=[court_center,unknown_target]",
        "dataset.court.view.coverage_modes=[full,unknown_coverage]",
        "dataset.court.sampling.mode=unknown_sampling",
        "dataset.court.sampling.stable_field_order=[shape,unknown_field]",
        "dataset.court.sampling.coverage_objective=[coverage_mode,unknown_objective]",
    ],
)
def test_unknown_court_modes_fail_at_configuration_boundary(override: str) -> None:
    with pytest.raises(SemanticConfigurationError, match="unknown value"):
        _compose(override)


def test_unknown_court_key_fails_at_configuration_boundary() -> None:
    with pytest.raises(UnknownConfigurationKeyError, match="unknown_key"):
        _compose("+dataset.court.trajectory.unknown_key=true")


def test_public_nht_commands_and_trainer_runtime_are_explicit() -> None:
    runtime = _compose()

    assert runtime.nht.reconstruct_executable == "nht-reconstruct"
    assert runtime.nht.render_executable == "nht-render"
    assert (
        runtime.nht.pipeline_config.path
        == (
            runtime.resolver.roots.external_asset_root / "nht/configs/production.yaml"
        ).resolve()
    )
    assert runtime.nht.pipeline_config.schema == "nht_pipeline_config_v1"
    assert runtime.nht.training_runtime.python == (
        runtime.resolver.roots.external_asset_root / "nht/.trainer-venv/bin/python"
    )
    assert (
        runtime.nht.training_runtime.trainer
        == (
            runtime.resolver.roots.external_asset_root
            / "nht/gsplat/examples/simple_trainer_nht.py"
        ).resolve()
    )
    assert runtime.nht.environment == _NHT_ENVIRONMENT
    assert runtime.nht.reconstruction_timeout_seconds == 86_400.0
    assert runtime.nht.render_timeout_seconds == 3_600.0
    assert not hasattr(runtime.nht, "sha256")
    assert not hasattr(runtime.nht, "commit")
    assert not hasattr(runtime.nht, "repository_root")
    assert not hasattr(runtime.nht, "reconstruction_config_path")


def test_configured_paths_retain_their_declared_runtime_roles() -> None:
    runtime = _compose()
    line_model = runtime.alignment.evidence.line_model

    assert (
        runtime.resolver.validate(
            PathRole.EXTERNAL_ASSET,
            runtime.nht.pipeline_config.path,
        )
        == runtime.nht.pipeline_config.path
    )
    assert (
        runtime.resolver.validate(
            PathRole.CHECKPOINT,
            line_model.checkpoint_path,
        )
        == line_model.checkpoint_path
    )
    for path in (
        line_model.backbone_repository_path,
        line_model.backbone_checkpoint_path,
    ):
        assert runtime.resolver.validate(PathRole.EXTERNAL_ASSET, path) == path
    for path in (runtime.plcs.accad_root, runtime.plcs.smplh_model_root):
        assert runtime.resolver.validate(PathRole.DATA, path) == path


def test_composition_root_can_construct_each_no_default_runtime_input() -> None:
    runtime = _compose()
    client = NHTRenderClient()
    composed_client = NHTComposedRenderClient()

    reconstruction = NHTReconstructionHandler(
        executable=runtime.nht.reconstruct_executable,
        pipeline_config=runtime.nht.pipeline_config,
        training_runtime=runtime.nht.training_runtime,
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
        client=composed_client,
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
    assert parameters.production_mode is PLCSProductionMode.MULTI_OBJECT_GLOBAL_TIMELINE
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


def test_single_object_plcs_config_composes_the_real_production_path() -> None:
    runtime = _compose("dataset/plcs=single_object")
    parameters = runtime.plcs.build_stage_parameters(seed=runtime.stages.seed)

    assert runtime.plcs.production_mode is PLCSProductionMode.SINGLE_OBJECT
    assert runtime.plcs.timeline.frame_selection == "all_source_frames"
    assert runtime.plcs.require_articulated_motion
    assert runtime.plcs.performance.require_cuda
    assert parameters.production_mode is PLCSProductionMode.SINGLE_OBJECT
    assert len(parameters.objects) == 1
    assert parameters.objects[0].category.value == "running"
    assert parameters.objects[0].start_frame == 0


def test_unknown_plcs_production_mode_fails_at_configuration_boundary() -> None:
    with pytest.raises(SemanticConfigurationError, match="production_mode"):
        _compose("dataset.plcs.production_mode=unknown")


def test_single_object_plcs_rejects_nonzero_start_frame() -> None:
    with pytest.raises(SemanticConfigurationError, match="start_frame=0"):
        _compose(
            "dataset/plcs=single_object",
            "dataset.plcs.objects.0.start_frame=1",
        )


def test_task_local_generation_camera_profiles_remain_available() -> None:
    canonical_root = PROJECT_ROOT / "src/synthetic_data_generation/configs/camera"
    assert {path.name for path in canonical_root.glob("*.yaml")} == {
        "broadcast.yaml",
        "default.yaml",
    }
    for task in ("blcs", "plcs"):
        camera_root = PROJECT_ROOT / f"src/tasks/{task}/configs/camera"
        profiles = tuple(sorted(camera_root.glob("*.yaml")))
        assert {path.name for path in profiles} == {"broadcast.yaml", "default.yaml"}
        assert all(path.is_file() and not path.is_symlink() for path in profiles)


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
