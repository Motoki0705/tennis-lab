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
from src.tasks.plcs.models.plcs_multiview_model import PLCSMultiViewModel
from src.tasks.plcs.training.composition import build_plcs_lightning_module
from src.tasks.plcs.training.lightning_module import PLCSLightningModule
from src.tasks.plcs.training.metrics import CANONICAL_POSE_HEADLINE_KEYS
from src.utils.configuration import (
    MissingConfigurationKeyError,
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)

_CONFIG_DIR = Path("src/tasks/plcs/configs").resolve()


def test_multiview_all_outputs_beta01_config_composes_and_binds_model() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train",
            overrides=[
                "model=multiview_canonical",
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
    assert runtime.model.name == "plcs_multiview"
    assert runtime.model.boolean("predict_canonical_pose")
    assert isinstance(module.model, PLCSMultiViewModel)
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


def test_noncanonical_model_keeps_trajectory_only_metric_contract() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train",
            overrides=[
                "model=multiview",
                "loss=no_canonical",
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


def test_multiview_reprojection_config_composes_and_binds_loss() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train",
            overrides=[
                "model=multiview_canonical",
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


def test_kp14_pose_reprojection_recipe_composes_with_training_contract() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="train_kp14_pose_reprojection")

    runtime = PLCSTrainingConfig.from_config(config)
    module = build_plcs_lightning_module(config)

    assert isinstance(module, PLCSLightningModule)
    assert config.court_keypoints.selector == "physical_v1"
    assert runtime.model.name == "plcs_multiview_axial"
    assert isinstance(module.model, PLCSMultiViewAxialModel)
    assert runtime.model.boolean("predict_canonical_pose")
    assert runtime.model.string("canonical_pose_readout") == "direct"
    assert module.model.canonical_pose_head is not None

    assert config.data.mode == "multiview_sequence"
    assert config.data.num_court_kp == 14
    assert config.data.num_views_range == [3, 3]
    assert config.data.seq_len_range == [64, 64]
    assert config.data.batch_size == 1

    assert module.loss_fn.config.position_weight == 1.0
    assert module.loss_fn.config.rotation_weight == 0.05
    assert module.loss_fn.config.angle_weight == 0.05
    assert module.loss_fn.config.canonical_pose_weight == 1.0
    assert module.loss_fn.config.reprojection_weight == 1.0
    assert module.loss_fn.config.reprojection_smooth_l1_beta == 0.01

    assert config.training.trainer.precision == "bf16-mixed"
    assert config.training.trainer.max_epochs == 100
    assert config.training.trainer.accumulate_grad_batches == 4
    assert not config.training.early_stopping.enabled
    assert config.run.gpus == 1
    assert config.run.test_after_fit
    assert config.run.output_dir == (
        "plcs/"
        "plcs_multiview_axial_kp14_pose_reprojection_w1_rw005_aw005_e100_gpu0_half"
    )


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


@pytest.mark.parametrize(
    (
        "model_name",
        "hidden_dim",
        "num_heads",
        "num_stages",
        "ffn_dim",
        "rope_dim",
        "dropout",
    ),
    [
        ("track_query", 64, 4, 4, 128, 16, 0.0),
        ("track_query_base", 512, 8, 8, 1408, 64, 0.1),
    ],
)
def test_track_query_size_configs_compose_and_validate(
    model_name: str,
    hidden_dim: int,
    num_heads: int,
    num_stages: int,
    ffn_dim: int,
    rope_dim: int,
    dropout: float,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train_tracking_chunked",
            overrides=[f"model={model_name}"],
        )

    assert config.model.name == "plcs_track_query"
    assert config.model.hidden_dim == hidden_dim
    assert config.model.num_heads == num_heads
    assert config.model.num_stages == num_stages
    assert config.model.ffn_dim == ffn_dim
    assert config.model.rope_dim == rope_dim
    assert config.model.dropout == dropout
    assert config.model.mhc.coefficient_dim == 64
    assert config.model.cswa.compression_ratio == 4

    parsed = PLCSModelConfig.from_mapping(config.model)
    assert parsed.name == "plcs_track_query"
    assert parsed.integer("hidden_dim") == hidden_dim
    assert parsed.integer("num_stages") == num_stages


@pytest.mark.parametrize(
    ("condition", "ffn_mode", "mhc_writeback"),
    [
        ("a", "per_attention", "after_object_temporal"),
        ("b", "shared", "after_object_temporal"),
        ("c", "per_attention", "layer_end"),
        ("d", "shared", "layer_end"),
    ],
)
def test_all_four_track_query_ablation_configs_compose_and_validate(
    condition: str,
    ffn_mode: str,
    mhc_writeback: str,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train_tracking",
            overrides=[f"model=track_query_ablation_{condition}"],
        )

    runtime = PLCSTrainingConfig.from_config(config)

    assert runtime.model.name == "plcs_track_query_ablation"
    assert runtime.model.string("ffn_mode") == ffn_mode
    assert runtime.model.string("mhc_writeback") == mhc_writeback
    assert runtime.model.integer("num_queries") == 4


@pytest.mark.parametrize(
    ("violation", "error"),
    [
        ("missing_ffn", MissingConfigurationKeyError),
        ("missing_writeback", MissingConfigurationKeyError),
        ("unknown", UnknownConfigurationKeyError),
        ("invalid_ffn", SemanticConfigurationError),
        ("invalid_writeback", SemanticConfigurationError),
        ("unknown_ffn_type", SemanticConfigurationError),
    ],
)
def test_ablation_axes_reject_missing_unknown_invalid_and_inconsistent_values(
    violation: str,
    error: type[Exception],
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train_tracking",
            overrides=["model=track_query_ablation_a"],
        )

    with open_dict(config.model):
        if violation == "missing_ffn":
            del config.model["ffn_mode"]
        elif violation == "missing_writeback":
            del config.model["mhc_writeback"]
        elif violation == "unknown":
            config.model["legacy_ablation"] = True
        elif violation == "invalid_ffn":
            config.model.ffn_mode = "legacy"
        elif violation == "invalid_writeback":
            config.model.mhc_writeback = "before_spatial"
        else:
            config.model.ffn_type = "unknown"

    with pytest.raises(error):
        PLCSModelConfig.from_mapping(config.model)


@pytest.mark.parametrize(
    ("profile", "model_name", "selector_mode"),
    [
        (
            "track_query_reference",
            "plcs_track_query_reference",
            "reference",
        ),
        (
            "track_query_ablation_d_v2_selector",
            "plcs_track_query_reference_ablation",
            "reference",
        ),
        (
            "track_query_ablation_d_v2_selector_zero",
            "plcs_track_query_reference_ablation",
            "selector_zero",
        ),
    ],
)
def test_reference_v2_profiles_compose_with_explicit_independent_contracts(
    profile: str,
    model_name: str,
    selector_mode: str,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train_tracking",
            overrides=[
                f"model={profile}",
                "court_keypoints=camera_view_v2",
            ],
        )

    runtime = PLCSTrainingConfig.from_config(config)
    assert runtime.model.name == model_name
    assert runtime.model.string("target_frame_contract") == (
        "reference_camera_court_rzpi_v1"
    )
    assert runtime.model.string("track_query_rope_contract") == (
        "time_camera_reference_selector_v1"
    )
    assert runtime.model.string("reference_selector_mode") == selector_mode
    assert "role_rope_enabled" not in runtime.model.values


@pytest.mark.parametrize(
    "profile",
    ["track_query_reference", "track_query_ablation_d_v2_selector"],
)
def test_reference_v2_rejects_rope_dim_four_and_accepts_dim_six(
    profile: str,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train_tracking",
            overrides=[
                f"model={profile}",
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
        ("track_query_reference", "physical_v1"),
        ("track_query", "camera_view_v2"),
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
                "model=track_query_reference",
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
                "model=track_query_ablation_d_v2_selector",
                "court_keypoints=camera_view_v2",
            ],
        )
    with open_dict(config.model):
        config.model[field] = value
    with pytest.raises(SemanticConfigurationError, match=message):
        PLCSModelConfig.from_mapping(config.model)
