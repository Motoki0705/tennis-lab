"""Strict canonical scene-pipeline configuration contracts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import astuple
from pathlib import Path
from typing import cast

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from src.synthetic_data_generation.alignment.contracts import AlignmentAcceptancePolicy
from src.synthetic_data_generation.alignment.settings import AlignmentEvidenceSettings
from src.synthetic_data_generation.configuration import (
    SCENE_PIPELINE_SCHEMA,
    AlignmentConfiguration,
    CourtDatasetConfiguration,
    ScenePipelineConfiguration,
    _blcs_generator_config,
    _blcs_source_settings,
)
from src.synthetic_data_generation.dataset.blcs.source import (
    BLCSTrajectorySourceSettings,
)
from src.synthetic_data_generation.dataset.court.contracts import OrbitTargetMode
from src.synthetic_data_generation.dataset.court.schema import (
    CourtDatasetSchemaVersion,
)
from src.synthetic_data_generation.pipeline.contracts import DatasetTarget, StageName
from src.synthetic_data_generation.reconstruction import NHT_PIPELINE_CONFIG_SCHEMA
from src.tasks.blcs.generate_dataset.source_api import (
    BLCSGeneratorConfiguration,
    BLCSTimelineSpec,
)
from src.utils.configuration import (
    ConfigurationError,
    PathContractError,
    PathResolver,
    RuntimePathRoots,
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)
from src.utils.paths import PROJECT_ROOT

_CONFIG_ROOT = PROJECT_ROOT / "src/synthetic_data_generation/configs"
_BLCS_GENERATOR_RUNTIME_TYPE = cast(type[object], BLCSGeneratorConfiguration)

pytestmark = pytest.mark.local_data


def _compose(*overrides: str) -> DictConfig:
    with initialize_config_dir(version_base="1.3", config_dir=str(_CONFIG_ROOT)):
        return compose(config_name="run_scene_pipeline", overrides=list(overrides))


def test_blcs_configuration_is_parsed_through_public_source_contracts() -> None:
    config = _compose()

    generator = _blcs_generator_config(config.dataset.blcs.generator)
    source = _blcs_source_settings(config.dataset.blcs.trajectory_source)

    assert isinstance(generator, _BLCS_GENERATOR_RUNTIME_TYPE)
    assert isinstance(source.timeline, BLCSTimelineSpec)
    assert source.timeline.num_frames == 1024
    assert source.maximum_physics_attempts_per_object == 64


def test_b00_configuration_is_the_canonical_scene_request() -> None:
    runtime = ScenePipelineConfiguration.from_config(_compose())

    assert runtime.profile == "b00-production"
    assert runtime.request.scene_id == "B00"
    assert runtime.request.config_schema == SCENE_PIPELINE_SCHEMA
    assert runtime.request.from_stage is StageName.INGEST
    assert runtime.request.through_stage is StageName.REPORT
    assert runtime.request.targets == frozenset(DatasetTarget)
    assert (
        runtime.workspace.root
        == (PROJECT_ROOT / "data/synthetic_data_generation/scenes/B00").resolve()
    )
    assert "B01" not in runtime.workspace.root.parts
    assert "B02" not in runtime.workspace.root.parts
    assert (
        runtime.nht.pipeline_config.path
        == (
            runtime.resolver.roots.external_asset_root / "nht/configs/production.yaml"
        ).resolve()
    )
    assert runtime.nht.pipeline_config.schema == NHT_PIPELINE_CONFIG_SCHEMA
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


@pytest.mark.parametrize(
    ("profile", "scene_id"),
    [("b01", "B01"), ("b02", "B02"), ("b03", "B03")],
)
def test_alignment_terminal_profiles_do_not_require_plcs_scene_split(
    profile: str,
    scene_id: str,
) -> None:
    runtime = ScenePipelineConfiguration.from_config(
        _compose(
            f"profile={profile}",
            "request.through_stage=alignment",
        )
    )

    assert runtime.request.scene_id == scene_id
    assert runtime.request.through_stage is StageName.ALIGNMENT
    assert runtime.request.active_targets == frozenset()
    assert scene_id not in runtime.plcs.scene_splits


def test_dataset_terminal_must_belong_to_explicit_targets() -> None:
    config = _compose(
        "request.targets=[court]",
        "request.through_stage=plcs_dataset",
    )

    with pytest.raises(ValueError, match="requires the 'plcs' dataset target"):
        ScenePipelineConfiguration.from_config(config)


def test_b00_quantitative_and_full_timeline_values_are_config_owned() -> None:
    runtime = ScenePipelineConfiguration.from_config(_compose())

    assert runtime.court.sampling.proposal_budget == 4_800
    assert runtime.court.sampling.minimum_trajectory_groups >= 24
    assert runtime.court.sampling.minimum_accepted_frames >= 2_000
    assert runtime.court.sampling.maximum_adjacent_step_m <= 1.05
    assert runtime.blcs.timeline.frame_selection == "all_source_frames"
    assert runtime.plcs.timeline.frame_selection == "all_source_frames"
    assert runtime.blcs.timeline.chunk_size_frames not in {5, 12, 64}
    assert runtime.plcs.timeline.chunk_size_frames not in {5, 12, 64}
    assert (
        runtime.court.performance.maximum_wall_seconds,
        runtime.court.performance.maximum_nht_invocations,
        runtime.court.performance.maximum_complete_array_scans_per_sample,
    ) == (1_800.0, 8, 2)
    assert (
        runtime.blcs.performance.maximum_wall_seconds,
        runtime.blcs.performance.maximum_nht_invocations,
        runtime.blcs.performance.maximum_background_cache_misses,
        runtime.blcs.performance.maximum_published_fraction_of_dense_reference,
        runtime.blcs.performance.maximum_batch_frames,
    ) == (3_600.0, 3, 18, 0.2, 1)
    assert (
        runtime.plcs.performance.maximum_wall_seconds,
        runtime.plcs.performance.maximum_nht_invocations,
        runtime.plcs.performance.maximum_background_cache_misses,
        runtime.plcs.performance.maximum_published_fraction_of_dense_reference,
        runtime.plcs.performance.maximum_batch_frames,
    ) == (5_400.0, 1, 12, 0.25, 32)
    assert {
        runtime.court.performance.execution_device,
        runtime.blcs.performance.execution_device,
        runtime.plcs.performance.execution_device,
    } == {"cuda:0"}


@pytest.mark.parametrize(
    ("selector", "version", "target_modes"),
    [
        (
            "v1",
            CourtDatasetSchemaVersion.V1,
            frozenset(OrbitTargetMode),
        ),
        (
            "v2",
            CourtDatasetSchemaVersion.V2,
            frozenset({OrbitTargetMode.COURT_CENTER}),
        ),
        (
            "v3",
            CourtDatasetSchemaVersion.V3,
            frozenset({OrbitTargetMode.COURT_CENTER}),
        ),
    ],
)
def test_explicit_court_schema_compositions_preserve_versioned_target_contract(
    selector: str,
    version: CourtDatasetSchemaVersion,
    target_modes: frozenset[OrbitTargetMode],
) -> None:
    config = _compose(f"dataset/court={selector}")

    court = CourtDatasetConfiguration.from_mapping(
        OmegaConf.to_container(config.dataset.court, resolve=True)
    )

    assert court.schema_version is version
    assert frozenset(court.view.target_modes) == target_modes
    assert court.sampling.minimum_accepted_fraction == 0.9


@pytest.mark.parametrize(
    "version",
    [
        CourtDatasetSchemaVersion.V1,
        CourtDatasetSchemaVersion.V2,
        CourtDatasetSchemaVersion.V3,
    ],
)
@pytest.mark.parametrize(
    ("v4_only_key", "value"),
    [
        ("support", {}),
        ("benchmark_decision_id", "test-decision"),
    ],
)
def test_legacy_court_schema_rejects_v4_only_root_keys_as_semantic_mismatch(
    version: CourtDatasetSchemaVersion,
    v4_only_key: str,
    value: object,
) -> None:
    raw = OmegaConf.to_container(
        _compose(f"dataset/court={version.value}").dataset.court,
        resolve=True,
    )
    assert isinstance(raw, dict)
    raw[v4_only_key] = value

    with pytest.raises(
        SemanticConfigurationError,
        match=(
            rf"schema_version={version.value} is incompatible with V4-only "
            rf"configuration key\(s\): dataset\.court\.{v4_only_key}\."
        ),
    ):
        CourtDatasetConfiguration.from_mapping(raw)


@pytest.mark.parametrize(
    "version",
    [
        CourtDatasetSchemaVersion.V1,
        CourtDatasetSchemaVersion.V2,
        CourtDatasetSchemaVersion.V3,
    ],
)
def test_legacy_court_schema_preserves_unknown_root_key_error(
    version: CourtDatasetSchemaVersion,
) -> None:
    raw = OmegaConf.to_container(
        _compose(f"dataset/court={version.value}").dataset.court,
        resolve=True,
    )
    assert isinstance(raw, dict)
    raw["genuine_unknown"] = True

    with pytest.raises(UnknownConfigurationKeyError, match="genuine_unknown"):
        CourtDatasetConfiguration.from_mapping(raw)


def test_production_alignment_evidence_and_acceptance_are_complete_typed_values(
    tmp_path: Path,
) -> None:
    roots = RuntimePathRoots(
        project_root=(tmp_path / "project").resolve(),
        data_root=(tmp_path / "data").resolve(),
        checkpoint_root=(tmp_path / "checkpoint").resolve(),
        artifact_root=(tmp_path / "artifact").resolve(),
        output_root=(tmp_path / "output").resolve(),
        cache_root=(tmp_path / "cache").resolve(),
        external_asset_root=(tmp_path / "external-assets").resolve(),
    )
    resolver = PathResolver(roots)
    alignment = AlignmentConfiguration.from_mapping(
        OmegaConf.load(_CONFIG_ROOT / "alignment/production.yaml"),
        resolver=resolver,
    )
    evidence = alignment.evidence

    assert isinstance(evidence, AlignmentEvidenceSettings)
    assert isinstance(alignment.acceptance, AlignmentAcceptancePolicy)
    assert alignment.acceptance.holdout.minimum_camera_count == 3
    assert alignment.acceptance.holdout.minimum_correspondence_count == 80
    assert (
        evidence.seed,
        evidence.fit_fraction,
        evidence.holdout_fraction,
        evidence.minimum_fit_cameras,
        evidence.minimum_holdout_cameras,
        evidence.camera_prefix_count,
    ) == (42, 2.0 / 3.0, 1.0 / 3.0, 8, 4, 48)
    assert evidence.camera_partition_unit_count() == 4
    assert (
        evidence.line_model.checkpoint_path
        == (
            resolver.roots.checkpoint_root
            / "court_detection/line/court-detection-epoch19.ckpt"
        ).resolve()
    )
    assert (
        evidence.line_model.backbone_repository_path
        == (resolver.roots.external_asset_root / "dinov3").resolve()
    )
    assert (
        evidence.line_model.backbone_checkpoint_path
        == (
            resolver.roots.external_asset_root
            / "dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        ).resolve()
    )
    assert (
        evidence.line_model.device,
        evidence.line_model.expected_short_side,
        evidence.line_model.probability_threshold,
        evidence.line_model.maximum_selected_pixels_per_camera,
    ) == ("cuda:0", 256, 0.5, 50_000)
    architecture = evidence.line_model.architecture
    assert architecture.backbone_name == "dinov3_vitb16"
    assert architecture.backbone_strict is True
    assert architecture.backbone_train_mode == "frozen"
    assert architecture.backbone_last_n_blocks == 0
    assert architecture.backbone_out_indices == (2, 5, 8, 11)
    assert architecture.backbone_layer_mode == "uniform"
    assert architecture.lora_enabled is True
    assert (
        architecture.lora_rank,
        architecture.lora_alpha,
        architecture.lora_dropout,
        architecture.lora_target_modules,
    ) == (8, 16.0, 0.0, ("qkv", "proj", "fc1", "fc2"))
    assert architecture.decoder_channels == 256
    assert architecture.decoder_reassemble_factors == (4.0, 2.0, 1.0, 0.5)
    assert (
        architecture.line_bce_weight,
        architecture.line_dice_weight,
        architecture.line_positive_weight,
    ) == (1.0, 1.0, 8.0)
    assert astuple(evidence.ground_plane) == (
        0.01,
        0.5,
        0.08,
        0.30,
        0.005,
        0.035,
        0.006,
        0.008,
        1000,
        20_000,
        3,
        500,
        1000,
        0.97,
        1.0,
        0.01,
    )
    assert astuple(evidence.projection) == (
        0.05,
        3.0,
        0.05,
        0.35,
        2.0,
        0.0025,
        20,
    )
    assert astuple(evidence.candidate_fit) == (
        8,
        128,
        0.05,
        6.0,
        0.055,
        0.085,
        -1.5707963267948966,
        1.5707963267948966,
        0.5,
        0.02,
        0.2,
        0.3,
        10.97,
        70,
        8,
        1.0e-5,
        100_000,
        0.07290400972053463,
        0.01,
        0.35,
        0.3,
        0.6,
        1.5,
        0.5,
        1.0e-9,
    )
    assert astuple(evidence.correspondences) == (0.25, 200, 3)
    assert astuple(alignment.acceptance.fit) == (
        6,
        100,
        0.3,
        0.9,
        0.3,
        0.3,
    )
    assert astuple(alignment.acceptance.holdout) == (
        3,
        80,
        0.3,
        0.9,
        0.3,
        0.3,
    )


def test_blcs_and_plcs_production_inputs_are_typed_and_have_no_frame_subset() -> None:
    runtime = ScenePipelineConfiguration.from_config(_compose())

    assert isinstance(runtime.blcs.trajectory_source, BLCSTrajectorySourceSettings)
    assert isinstance(runtime.blcs.trajectory_source.timeline, BLCSTimelineSpec)
    assert isinstance(runtime.blcs.generator, _BLCS_GENERATOR_RUNTIME_TYPE)
    assert runtime.blcs.trajectory_source.scene_count == 3
    assert runtime.blcs.trajectory_source.maximum_physics_attempts_per_object == 64
    assert runtime.blcs.trajectory_source.split_scene_counts == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    assert runtime.blcs.trajectory_source.timeline.num_frames == 1024
    assert runtime.blcs.assets.ball.role.value == "movable"
    assert runtime.blcs.assets.ball.asset_class == "ball"
    assert runtime.blcs.assets.ball.floating_dtype == "float32"
    assert runtime.blcs.assets.ball.appearance_model == "rgb"
    assert runtime.blcs.assets.ball.appearance_space == "linear_rgb"
    assert runtime.blcs.assets.settings.radius_m == 0.0335
    assert runtime.blcs.assets.settings.visibility_threshold == 0.0001
    assert runtime.blcs.render_timeout_seconds == runtime.nht.render_timeout_seconds

    assert (
        runtime.plcs.accad_root
        == (runtime.resolver.roots.data_root / "ACCAD").resolve()
    )
    assert (
        runtime.plcs.smplh_model_root
        == (runtime.resolver.roots.data_root / "smplh").resolve()
    )
    assert runtime.plcs.scene_splits == {
        "B00": "train",
        "B00-plcs-002": "train",
    }
    assert tuple(item.category.value for item in runtime.plcs.objects) == (
        "running",
        "walking",
        "general",
    )
    assert runtime.plcs.gaussian_count == 2048
    assert runtime.plcs.smplh_batch_size == 32
    assert runtime.plcs.device == "cuda:0"
    assert runtime.plcs.appearance.source == "palette"
    assert runtime.plcs.appearance.assignment == "object_index_modulo_palette"
    assert runtime.plcs.appearance.gaussian_fill == "uniform"
    assert runtime.plcs.appearance.appearance_model == "rgb"
    assert runtime.plcs.appearance.appearance_space == "linear_rgb"
    assert len(runtime.plcs.appearance.colors) == 6
    assert (
        runtime.plcs.appearance.color_for_object(6) == runtime.plcs.appearance.colors[0]
    )
    assert runtime.plcs.render_timeout_seconds == runtime.nht.render_timeout_seconds


@pytest.mark.parametrize(
    ("key", "invalid"),
    [
        ("request.targets", []),
        ("request.targets", ["court", "unknown"]),
        ("request.from_stage", "legacy_pipeline"),
        ("request.through_stage", "legacy_pipeline"),
        ("pipeline.config_schema", "legacy"),
        ("pipeline.preflight_before_invalidation", False),
        ("camera.slots.0.hfov_degrees", [0.0, 20.0]),
        ("dataset.court.sampling.proposal_budget", 5_001),
        ("dataset.court.sampling.minimum_trajectory_groups", 23),
        ("dataset.court.trajectory.axis_ratios", [1.0, 0.9]),
        ("dataset.blcs.timeline.frame_selection", "first_64"),
        ("dataset.plcs.require_articulated_motion", False),
        ("nht.reconstruction_timeout_seconds", 0.0),
        ("alignment.evidence.holdout_fraction", 0.0),
        ("alignment.evidence.camera_prefix_count", 11),
        ("dataset.blcs.trajectory_source.timeline.min_tracks", 1),
        ("dataset.blcs.generator.physics.gravity", "9.81"),
        ("dataset.plcs.appearance.appearance_space", "srgb"),
        ("dataset.plcs.foreground_rasterizer.maximum_alpha", 1.0),
        ("dataset.court.performance.maximum_complete_array_scans_per_sample", 0),
        ("dataset.blcs.performance.execution_device", ""),
        ("dataset.blcs.performance.execution_device", "cuda:1"),
        ("dataset.blcs.performance.maximum_batch_frames", 2),
        ("dataset.blcs.assets.ball.floating_dtype", "float64"),
        ("dataset.blcs.performance.maximum_published_fraction_of_dense_reference", 1.1),
        ("dataset.plcs.performance.maximum_nht_invocations", 2),
    ],
)
def test_invalid_values_fail_closed(key: str, invalid: object) -> None:
    config = _compose()
    OmegaConf.update(config, key, invalid, merge=False)

    with pytest.raises((ConfigurationError, TypeError, ValueError)):
        ScenePipelineConfiguration.from_config(config)


def test_unknown_keys_fail_before_runtime_path_creation() -> None:
    config = deepcopy(_compose())
    with open_dict(config):
        config.pipeline.commit = "forbidden"

    with pytest.raises(ConfigurationError, match="Unknown configuration key"):
        ScenePipelineConfiguration.from_config(config)


def test_source_video_cannot_escape_external_asset_root() -> None:
    config = _compose()
    OmegaConf.update(config, "request.source_video", "../outside.mp4", merge=False)

    with pytest.raises(PathContractError):
        ScenePipelineConfiguration.from_config(config)


@pytest.mark.parametrize(
    "private_key",
    ["repository_root", "reconstruction_config_path", "training_runtime"],
)
def test_private_nht_configuration_keys_fail_closed(private_key: str) -> None:
    config = _compose()
    with open_dict(config.nht):
        config.nht[private_key] = "forbidden"

    with pytest.raises(ConfigurationError, match="Unknown configuration key"):
        ScenePipelineConfiguration.from_config(config)


def test_private_nht_python_environment_fails_closed() -> None:
    config = _compose()
    with open_dict(config.nht.environment):
        config.nht.environment.PYTHONPATH = "/private/provider/modules"

    with pytest.raises(ConfigurationError, match="environment.PYTHONPATH"):
        ScenePipelineConfiguration.from_config(config)


def _nht_config_at(root: Path, contents: str) -> Path:
    path = root / "pipeline.yaml"
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return path


def _compose_with_nht_config_root(root: Path) -> DictConfig:
    config = _compose()
    data_root = root.parent / "data"
    data_root.mkdir(exist_ok=True)
    source_video = data_root / "synthetic_data_generation/raw/B00.mp4"
    source_video.parent.mkdir(parents=True)
    source_video.write_bytes(b"configuration fixture")
    backbone = root / "dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    backbone.parent.mkdir(parents=True, exist_ok=True)
    backbone.write_bytes(b"configuration fixture")
    OmegaConf.update(
        config,
        "roots.data_root",
        str(data_root.resolve()),
        merge=False,
    )
    OmegaConf.update(
        config,
        "roots.external_asset_root",
        str(root.resolve()),
        merge=False,
    )
    OmegaConf.update(config, "nht.pipeline_config_path", "pipeline.yaml", merge=False)
    return config


@pytest.mark.parametrize("kind", ["missing", "directory", "symlink"])
def test_nht_pipeline_config_path_failures_are_rejected(
    tmp_path: Path,
    kind: str,
) -> None:
    root = tmp_path / "external"
    root.mkdir()
    path = root / "pipeline.yaml"
    if kind == "directory":
        path.mkdir()
    elif kind == "symlink":
        target = _nht_config_at(
            tmp_path / "target",
            f"schema: {NHT_PIPELINE_CONFIG_SCHEMA}\n",
        )
        path.symlink_to(target)
    config = _compose_with_nht_config_root(root)

    with pytest.raises(PathContractError, match="pipeline_config_path"):
        ScenePipelineConfiguration.from_config(config)


@pytest.mark.parametrize(
    "contents",
    [
        "schema: [\n",
        "schema: legacy_nht_pipeline_config\n",
        f"schema: {NHT_PIPELINE_CONFIG_SCHEMA}\nprivate_runtime: true\n",
    ],
)
def test_invalid_nht_pipeline_config_fails_at_configuration_boundary(
    tmp_path: Path,
    contents: str,
) -> None:
    root = tmp_path / "external"
    _nht_config_at(root, contents)

    with pytest.raises((TypeError, ValueError)):
        ScenePipelineConfiguration.from_config(_compose_with_nht_config_root(root))


@pytest.mark.parametrize(
    ("key", "basename"),
    [
        ("reconstruct_executable", "nht-reconstruct"),
        ("render_executable", "nht-render"),
    ],
)
def test_nht_commands_accept_installed_absolute_public_executables(
    tmp_path: Path,
    key: str,
    basename: str,
) -> None:
    executable = tmp_path / "bin" / basename
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    config = _compose()
    OmegaConf.update(config, f"nht.{key}", str(executable.resolve()), merge=False)

    runtime = ScenePipelineConfiguration.from_config(config)

    assert getattr(runtime.nht, key) == executable.resolve()


@pytest.mark.parametrize(
    ("key", "invalid"),
    [
        ("reconstruct_executable", "python"),
        ("reconstruct_executable", "simple_trainer_nht.py"),
        ("render_executable", "provider/bin/nht-render"),
    ],
)
def test_nht_commands_reject_private_or_relative_executables(
    key: str,
    invalid: str,
) -> None:
    config = _compose()
    OmegaConf.update(config, f"nht.{key}", invalid, merge=False)

    with pytest.raises(ConfigurationError, match="absolute path"):
        ScenePipelineConfiguration.from_config(config)


def test_nested_unknown_keys_fail_closed() -> None:
    config = OmegaConf.to_container(_compose(), resolve=True)
    assert isinstance(config, dict)
    alignment = config["alignment"]
    assert isinstance(alignment, dict)
    evidence = alignment["evidence"]
    assert isinstance(evidence, dict)
    candidate = evidence["candidate_fit"]
    assert isinstance(candidate, dict)
    candidate["fingerprint"] = "forbidden"

    with pytest.raises(ConfigurationError, match="Unknown configuration key"):
        ScenePipelineConfiguration.from_config(config)
