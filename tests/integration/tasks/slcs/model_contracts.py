"""SLCS registration: real pseudo-label data, six heads and optional contexts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from src.tasks.slcs.data.splits import save_split_file
from src.tasks.slcs.training.lightning_module import SLCSLightningModule
from src.tasks.slcs.training.losses import build_slcs_loss_inputs
from src.tasks.slcs.training.runner import SLCSTrainingRunner
from tests.support.model_contracts.parameters import (
    OptimizerContract,
    ParameterContract,
    names_under,
)
from tests.support.model_contracts.registry import ModelFamily, save_checkpoint
from tests.support.tasks.slcs.dataset import (
    SLCSFixtureDatasetConfig,
    build_slcs_dataset_fixture,
)

CONFIG_DIR = Path("src/tasks/slcs/configs")
SELECTORS = {"base": "slcs_fusion", "small": "slcs_fusion"}


def _compose(root: Path, gan: bool, *, contexts: bool = False) -> DictConfig:
    assert not gan, "SLCS has no GAN selector"
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()), version_base="1.3"
    ):
        return compose(
            config_name="train",
            overrides=[
                "model=small",
                f"paths.data_root={root}",
                "data.dataset_root=fixture",
                "data.split_file=fixture/splits.json",
                "data.batch_size=2",
                "data.num_workers=0",
                "data.pin_memory=false",
                "data.window_size=8",
                "data.train_stride=8",
                "data.eval_stride=8",
                "data.augmentation.enabled=false",
                "data.dino.image_height=48",
                "data.dino.image_width=64",
                "data.dino.embed_dim=8",
                "model.hidden_dim=16",
                "model.num_heads=2",
                "model.ffn_dim=32",
                "model.rope_dim=8",
                "model.num_shared_layers=1",
                "model.dropout=0.0",
                f"model.num_position_layers={int(contexts)}",
                f"model.num_rotation_layers={int(contexts)}",
                *[
                    f"model.{key}={str(contexts).lower()}"
                    for key in (
                        "missing_ball_court_context",
                        "missing_ball_temporal_context",
                        "missing_ball_one_sided_context",
                    )
                ],
                "training.compile.enabled=false",
                "training.learning_rate=0.001",
                "training.weight_decay=0.0",
                "training.warmup_steps=0",
                "training.trainer.max_epochs=2",
                "training.qualitative_logging.enabled=false",
                "loss.ball_velocity_weight=0.1",
                "loss.ball_velocity_scale_mps=30.0",
            ],
        )


def _build(config: DictConfig) -> tuple[Any, Any]:
    index = build_slcs_dataset_fixture(
        Path(config.paths.data_root) / "fixture",
        SLCSFixtureDatasetConfig(
            videos=("video_000",), num_frames=8, ball_visibility=0.65
        ),
    )
    save_split_file(
        index.root / "splits.json",
        {"video_000": "train"},
        seed=0,
        val_ratio=0.0,
        test_ratio=0.0,
    )
    # Reuse train only through the explicit overfit data mode; no implicit split fallback.
    config.data.overfit = True
    runner = SLCSTrainingRunner()
    data = runner.build_datamodule(config)
    data.setup("fit")
    return runner.build_lightning_module(config, data, steps_per_epoch=1), data


def _batch(config: DictConfig) -> Any:
    data = SLCSTrainingRunner().build_datamodule(config)
    data.setup("fit")
    return next(iter(data.train_dataloader()))


def _result(module: Any, batch: Any) -> Any:
    outputs = module.forward_batch(batch)
    targets = module.model_adapter.build_training_targets(batch)
    losses = module.loss_fn(build_slcs_loss_inputs(outputs, targets))
    assert losses["ball_velocity"] > 0
    return {"loss": losses["total"], "outputs": outputs}


def _parameters(module: Any) -> tuple[ParameterContract, tuple[OptimizerContract, ...]]:
    active = names_under(module, "model")
    groups = {
        key: names_under(module, f"model.{key}")
        for key in (
            "player_embed",
            "ball_embed",
            "dino_encoder",
            "entity_layers",
            "time_layers",
            "dino_cross_layers",
            "player_position_head",
            "player_rotation_head",
            "ball_position_head",
            "player_position_scale_head",
            "player_rotation_scale_head",
            "ball_position_scale_head",
        )
    }
    if module.model.missing_ball_context is not None:
        for key in (
            "missing_ball_context",
            "missing_ball_temporal_context",
            "missing_ball_one_sided_context",
            "position_entity_layers",
            "position_time_layers",
            "position_dino_cross_layers",
            "rotation_entity_layers",
            "rotation_time_layers",
            "rotation_dino_cross_layers",
        ):
            groups[key] = names_under(module, f"model.{key}")
    return ParameterContract(active, frozenset(), frozenset()), (
        OptimizerContract("generator", active, groups),
    )


def _assert_batch(batch: Any) -> None:
    assert batch["padding_mask"].shape == (2, 8)
    assert batch["padding_mask"].dtype == torch.bool and not batch["padding_mask"].any()
    assert (~batch["ball_vis"]).any() and batch["ball_vis"].any()
    assert batch["dino_tokens"].dtype == torch.float32


def _assert_prediction(output: Any) -> None:
    for key, shape in (
        ("player_position", (2, 2, 8, 3)),
        ("player_rotation", (2, 2, 8, 2)),
        ("ball_position", (2, 8, 3)),
        ("player_position_log_b", (2, 2, 8)),
        ("player_rotation_log_b", (2, 2, 8)),
        ("ball_position_log_b", (2, 8)),
    ):
        value = getattr(output, key)
        assert (
            value.shape == shape
            and value.dtype == torch.float32
            and torch.isfinite(value).all()
        ), key


SHARED = ModelFamily(
    name="slcs_fusion",
    compose=_compose,
    build=_build,
    batch=_batch,
    supervised_result=_result,
    predict=lambda module, batch: module.forward_batch(batch),
    save=save_checkpoint,
    restore=lambda config, path: SLCSLightningModule.load_from_checkpoint(
        path, config=config, map_location="cpu", weights_only=False
    ),
    parameters=_parameters,
    assert_batch=_assert_batch,
    assert_prediction=_assert_prediction,
)
CONTEXTS = replace(
    SHARED,
    name="slcs_fusion/split_contexts",
    compose=lambda root, gan: _compose(root, gan, contexts=True),
)
