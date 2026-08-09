"""Strict canonical scene-pipeline configuration contracts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import astuple
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from src.synthetic_data_generation.alignment.contracts import AlignmentAcceptancePolicy
from src.synthetic_data_generation.alignment.settings import AlignmentEvidenceSettings
from src.synthetic_data_generation.configuration import (
    SCENE_PIPELINE_SCHEMA,
    ScenePipelineConfiguration,
)
from src.synthetic_data_generation.dataset.blcs.source import (
    BLCSTrajectorySourceSettings,
)
from src.synthetic_data_generation.pipeline.contracts import DatasetTarget, StageName
from src.tasks.base.generate_dataset.timeline_composer import TimelineConfig
from src.tasks.blcs.generate_dataset.scene_generator import GeneratorConfig
from src.utils.configuration import ConfigurationError, PathContractError
from src.utils.paths import PROJECT_ROOT

_CONFIG_ROOT = PROJECT_ROOT / "src/synthetic_data_generation/configs"

pytestmark = pytest.mark.local_data


def _compose(*overrides: str) -> DictConfig:
    with initialize_config_dir(version_base="1.3", config_dir=str(_CONFIG_ROOT)):
        return compose(config_name="run_scene_pipeline", overrides=list(overrides))


def test_b00_configuration_is_the_canonical_scene_request() -> None:
    runtime = ScenePipelineConfiguration.from_config(_compose())

    assert runtime.profile == "b00-production"
    assert runtime.request.scene_id == "B00"
    assert runtime.request.config_schema == SCENE_PIPELINE_SCHEMA
    assert runtime.request.from_stage is StageName.INGEST
    assert runtime.request.targets == frozenset(DatasetTarget)
    assert runtime.workspace.root == (
        PROJECT_ROOT / "data/synthetic_data_generation/scenes/B00"
    ).resolve()
    assert "B01" not in runtime.workspace.root.parts
    assert "B02" not in runtime.workspace.root.parts


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
    ) == (3_600.0, 3, 18, 0.2, 64)
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


def test_b00_alignment_evidence_and_acceptance_are_complete_typed_values() -> None:
    runtime = ScenePipelineConfiguration.from_config(_compose())
    alignment = runtime.alignment
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
        evidence.maximum_cameras,
    ) == (42, 2.0 / 3.0, 1.0 / 3.0, 8, 4, 24)
    assert evidence.line_model.checkpoint_path == (
        runtime.resolver.roots.checkpoint_root
        / "court_detection/line/court-detection-epoch19.ckpt"
    ).resolve()
    assert evidence.line_model.backbone_repository_path == (
        runtime.resolver.roots.external_asset_root / "dinov3"
    ).resolve()
    assert evidence.line_model.backbone_checkpoint_path == (
        runtime.resolver.roots.external_asset_root
        / "dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    ).resolve()
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
    assert astuple(evidence.projection) == (0.05, 3.0, 0.05, 20)
    assert astuple(evidence.candidate_fit) == (
        2,
        6.0,
        0.055,
        0.085,
        -1.5708,
        1.5708,
        0.5,
        0.02,
        0.2,
        0.3,
        12.0,
        100.0,
        30,
        8,
        1.0e-5,
        100_000,
        0.3,
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
    assert isinstance(runtime.blcs.trajectory_source.timeline, TimelineConfig)
    assert isinstance(runtime.blcs.generator, GeneratorConfig)
    assert runtime.blcs.trajectory_source.scene_count == 3
    assert runtime.blcs.trajectory_source.maximum_physics_attempts_per_object == 64
    assert runtime.blcs.trajectory_source.split_scene_counts == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    assert runtime.blcs.trajectory_source.timeline.num_frames == 1024
    assert runtime.blcs.assets.background.role.value == "background"
    assert runtime.blcs.assets.ball.role.value == "movable"
    assert runtime.blcs.assets.ball.asset_class == "ball"
    assert (
        runtime.blcs.assets.ball.appearance_model
        == runtime.blcs.assets.background.appearance_model
    )
    assert (
        runtime.blcs.assets.ball.appearance_space
        == runtime.blcs.assets.background.appearance_space
    )
    assert runtime.blcs.assets.ball_radius_m == 0.0335
    assert runtime.blcs.render_timeout_seconds == runtime.nht.render_timeout_seconds

    assert runtime.plcs.accad_root == (
        runtime.resolver.roots.data_root / "ACCAD"
    ).resolve()
    assert runtime.plcs.smplh_model_root == (
        runtime.resolver.roots.data_root / "smplh"
    ).resolve()
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
    assert runtime.plcs.appearance.color_for_object(6) == runtime.plcs.appearance.colors[0]
    assert runtime.plcs.render_timeout_seconds == runtime.nht.render_timeout_seconds


@pytest.mark.parametrize(
    ("key", "invalid"),
    [
        ("request.targets", []),
        ("request.targets", ["court", "unknown"]),
        ("request.from_stage", "legacy_pipeline"),
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
        ("alignment.evidence.maximum_cameras", 11),
        ("dataset.blcs.trajectory_source.timeline.min_tracks", 1),
        ("dataset.blcs.generator.physics.gravity", "9.81"),
        ("dataset.plcs.appearance.appearance_space", "srgb"),
        ("dataset.plcs.foreground_rasterizer.maximum_alpha", 1.0),
        ("dataset.court.performance.maximum_complete_array_scans_per_sample", 0),
        ("dataset.blcs.performance.execution_device", ""),
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
