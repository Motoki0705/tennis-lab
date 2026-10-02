from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.models.components.heads import (
    TemporalDecomposedCanonicalPoseHead,
)
from src.tasks.plcs.models.plcs_multiview_axial_model import PLCSMultiViewAxialModel
from src.tasks.plcs.training.composition import build_plcs_lightning_module
from src.tasks.plcs.training.lightning_module import PLCSLightningModule
from src.tasks.plcs.training.metrics import CANONICAL_POSE_HEADLINE_KEYS

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
