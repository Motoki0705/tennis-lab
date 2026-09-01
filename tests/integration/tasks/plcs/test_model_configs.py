from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from src.tasks.plcs.configuration import PLCSModelConfig, PLCSTrainingConfig
from src.tasks.plcs.models.components.heads import (
    TemporalDecomposedCanonicalPoseHead,
)
from src.tasks.plcs.models.plcs_multiview_axial_model import PLCSMultiViewAxialModel
from src.tasks.plcs.training.composition import build_plcs_lightning_module
from src.tasks.plcs.training.lightning_module import PLCSLightningModule
from src.tasks.plcs.training.metrics import CANONICAL_POSE_HEADLINE_KEYS
from src.utils.configuration import (
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)

_CONFIG_DIR = Path("src/tasks/plcs/configs").resolve()


def test_axial_all_outputs_beta01_config_composes_and_binds_model() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train",
            overrides=[
                "model=multiview_axial_base",
                "loss=all_outputs_beta01",
                "model.hidden_dim=16",
                "model.num_heads=4",
                "model.ffn_dim=32",
                "model.rope_dim=4",
                "model.num_layers=1",
            ],
        )

    runtime = PLCSTrainingConfig.from_config(config)
    module = build_plcs_lightning_module(config)

    assert isinstance(module, PLCSLightningModule)
    assert runtime.model.name == "plcs_multiview_axial"
    assert runtime.model.boolean("predict_canonical_pose")
    assert isinstance(module.model, PLCSMultiViewAxialModel)
    assert module.model.canonical_pose_head is not None
    assert module.loss_fn.config.position_weight == 1.0
    assert module.loss_fn.config.position_smooth_l1_beta == 0.1
    assert module.loss_fn.config.rotation_weight == 1.0
    assert module.loss_fn.config.angle_weight == 1.0
    assert module.loss_fn.config.canonical_pose_weight == 1.0
    assert module.loss_fn.config.reprojection_weight == 0.0
    assert module.train_metrics.predict_canonical_pose
    assert set(CANONICAL_POSE_HEADLINE_KEYS) <= set(
        module.metric_logging_contract.for_stage("train").headline_keys
    )


def test_noncanonical_axial_model_keeps_trajectory_only_metric_contract() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train",
            overrides=[
                "model=multiview_axial_base",
                "loss=no_canonical",
                "model.predict_canonical_pose=false",
                "model.hidden_dim=16",
                "model.num_heads=4",
                "model.ffn_dim=32",
                "model.rope_dim=4",
                "model.num_layers=1",
            ],
        )

    module = build_plcs_lightning_module(config)

    assert isinstance(module, PLCSLightningModule)
    assert not module.io_adapter.predict_canonical_pose
    assert not module.train_metrics.predict_canonical_pose
    assert not set(CANONICAL_POSE_HEADLINE_KEYS).intersection(
        module.metric_logging_contract.for_stage("train").headline_keys
    )


def test_axial_reprojection_config_composes_and_binds_loss() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train",
            overrides=[
                "model=multiview_axial_base",
                "loss=all_outputs_beta01_reprojection",
                "model.hidden_dim=16",
                "model.num_heads=4",
                "model.ffn_dim=32",
                "model.rope_dim=4",
                "model.num_layers=1",
            ],
        )

    module = build_plcs_lightning_module(config)

    assert isinstance(module, PLCSLightningModule)
    assert module.loss_fn.config.position_weight == 1.0
    assert module.loss_fn.config.position_smooth_l1_beta == 0.1
    assert module.loss_fn.config.rotation_weight == 1.0
    assert module.loss_fn.config.angle_weight == 1.0
    assert module.loss_fn.config.canonical_pose_weight == 1.0
    assert module.loss_fn.config.reprojection_weight == 1.0
    assert module.loss_fn.config.reprojection_smooth_l1_beta == 0.01


def test_temporal_canonical_pose_model_config_composes_and_binds_head() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train",
            overrides=[
                "model=multiview_axial_base_temporal_pose",
                "loss=canonical_only",
                "model.hidden_dim=16",
                "model.num_heads=4",
                "model.ffn_dim=32",
                "model.rope_dim=4",
                "model.num_layers=1",
            ],
        )

    runtime = PLCSTrainingConfig.from_config(config)
    module = build_plcs_lightning_module(config)

    assert runtime.model.string("canonical_pose_readout") == "temporal_decomposition"
    assert isinstance(module.model, PLCSMultiViewAxialModel)
    assert isinstance(
        module.model.canonical_pose_head, TemporalDecomposedCanonicalPoseHead
    )


def test_tracking_query_profile_composes_and_validates() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train_tracking_chunked",
            overrides=["model=tracking_query"],
        )

    assert config.model.name == "plcs_track_query"
    assert config.model.hidden_dim == 64
    assert config.model.num_heads == 4
    assert config.model.num_stages == 4
    assert config.model.ffn_dim == 128
    assert config.model.rope_dim == 16
    assert config.model.dropout == 0.0
    assert config.model.mhc.coefficient_dim == 64
    assert config.model.cswa.compression_ratio == 4

    parsed = PLCSModelConfig.from_mapping(config.model)
    assert parsed.name == "plcs_track_query"
    assert parsed.integer("hidden_dim") == 64
    assert parsed.integer("num_stages") == 4


def test_reference_profile_composes_from_canonical_architecture() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train_tracking",
            overrides=[
                "model=tracking_query_reference",
                "court_keypoints=camera_view_v2",
            ],
        )

    runtime = PLCSTrainingConfig.from_config(config)
    assert runtime.model.name == "plcs_track_query_reference"
    assert runtime.model.string("target_frame_contract") == (
        "reference_camera_court_rzpi_v1"
    )
    assert runtime.model.string("track_query_rope_contract") == (
        "time_camera_reference_selector_v1"
    )
    assert runtime.model.string("reference_selector_mode") == "reference"
    assert runtime.model.integer("hidden_dim") == 64
    assert runtime.model.integer("num_stages") == 4
    assert "role_rope_enabled" not in runtime.model.values


def test_reference_rejects_rope_dim_four_and_accepts_dim_six() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train_tracking",
            overrides=[
                "model=tracking_query_reference",
                "court_keypoints=camera_view_v2",
                "model.hidden_dim=24",
                "model.num_heads=4",
                "model.rope_dim=6",
            ],
        )
    assert PLCSModelConfig.from_mapping(config.model).integer("rope_dim") == 6

    with open_dict(config.model):
        config.model.rope_dim = 4
    with pytest.raises(SemanticConfigurationError, match="at least 6"):
        PLCSModelConfig.from_mapping(config.model)


@pytest.mark.parametrize(
    ("model_profile", "court_profile"),
    [
        ("tracking_query_reference", "physical_v1"),
        ("tracking_query", "camera_view_v2"),
    ],
)
def test_track_query_runtime_rejects_mixed_v1_v2_contracts(
    model_profile: str,
    court_profile: str,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train_tracking",
            overrides=[
                f"model={model_profile}",
                f"court_keypoints={court_profile}",
            ],
        )
    with pytest.raises(SemanticConfigurationError, match="track-query models require"):
        PLCSTrainingConfig.from_config(config)


def test_reference_v2_does_not_reinterpret_role_rope_enabled() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train_tracking",
            overrides=[
                "model=tracking_query_reference",
                "court_keypoints=camera_view_v2",
                "+model.role_rope_enabled=true",
            ],
        )
    with pytest.raises(UnknownConfigurationKeyError, match="role_rope_enabled"):
        PLCSModelConfig.from_mapping(config.model)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("target_frame_contract", "physical_court_v1", "target_frame_contract"),
        ("track_query_rope_contract", "time_camera_role_v1", "rope_contract"),
        ("reference_selector_mode", "legacy_role", "selector_mode"),
    ],
)
def test_reference_v2_rejects_unknown_or_mixed_semantic_markers(
    field: str,
    value: str,
    message: str,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train_tracking",
            overrides=[
                "model=tracking_query_reference",
                "court_keypoints=camera_view_v2",
            ],
        )
    with open_dict(config.model):
        config.model[field] = value
    with pytest.raises(SemanticConfigurationError, match=message):
        PLCSModelConfig.from_mapping(config.model)
