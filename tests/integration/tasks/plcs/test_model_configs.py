from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from src.tasks.plcs.configuration import PLCSModelConfig, PLCSTrainingConfig
from src.tasks.plcs.models.plcs_multiview_model import PLCSMultiViewModel
from src.tasks.plcs.training.composition import build_plcs_lightning_module
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

    assert module.loss_fn.config.position_weight == 1.0
    assert module.loss_fn.config.position_smooth_l1_beta == 0.1
    assert module.loss_fn.config.rotation_weight == 1.0
    assert module.loss_fn.config.angle_weight == 1.0
    assert module.loss_fn.config.canonical_pose_weight == 1.0
    assert module.loss_fn.config.reprojection_weight == 1.0
    assert module.loss_fn.config.reprojection_smooth_l1_beta == 0.01


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
        ("non_swiglu", SemanticConfigurationError),
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
            config.model.ffn_type = "mlp"

    with pytest.raises(error):
        PLCSModelConfig.from_mapping(config.model)
