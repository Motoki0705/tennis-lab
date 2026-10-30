"""Refiner regression, Flow and physics profiles over real generated rallies."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from src.tasks.ball_refiner_3d.configuration.training import training_config
from src.tasks.ball_refiner_3d.data.datamodule import RefinerDataModule
from src.tasks.ball_refiner_3d.generate_dataset.generator import generate_dataset
from src.tasks.ball_refiner_3d.inference.clip import predict_normalized
from src.tasks.ball_refiner_3d.model_io.checkpoint import checkpoint_flight_clock
from src.tasks.ball_refiner_3d.training.lightning_module import RefinerLightningModule
from tests.support.model_contracts.lightning import fit_contract
from tests.support.model_contracts.parameters import (
    OptimizerContract,
    ParameterContract,
    names_under,
)
from tests.support.model_contracts.registry import ModelFamily, save_checkpoint

CONFIG_DIR = Path("src/tasks/ball_refiner_3d/configs")
SELECTORS = {"regression": "regression", "flow": "flow"}


def _compose(
    root: Path, gan: bool, *, architecture: str = "regression", physics: bool = False
) -> DictConfig:
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()), version_base="1.3"
    ):
        return compose(
            config_name="train",
            overrides=[
                f"model={architecture}",
                f"paths.data_root={root}",
                f"paths.output_root={root}",
                "run.output_dir=run",
                "run.gpus=0",
                "training.steps=2",
                "training.evaluate_every=1",
                "training.batch_size=2",
                "training.cpu_threads=1",
                "training.learning_rate=0.001",
                "training.weight_decay=0.0",
                "training.log_every=1",
                "model.width=16",
                "model.layers=1",
                "model.heads=2",
                "model.ffn_dim=32",
                "model.rope_dim=8",
                "model.dropout=0.0",
                "model.flow_steps=2",
                "data.window.length=8",
                "data.window.max_length=12",
                "data.window.long_probability=1.0",
                "training.gan.discriminator.num_layers=1",
                "training.gan.discriminator.dropout=0.0",
                f"training.gan.enabled={str(gan).lower()}",
                "training.gan.schedule_enabled=true",
                "training.gan.transition.start_step=1",
                "training.gan.warmup_steps=1",
                f"model.physics_heads={str(physics).lower()}",
                *[
                    f"loss.physics.{key}_weight={float(physics)}"
                    for key in ("field", "segment", "reconstruction", "consistency")
                ],
            ],
        )


def _build(config: DictConfig) -> tuple[Any, Any]:
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()), version_base="1.3"
    ):
        generation = compose(
            config_name="generate_dataset",
            overrides=[
                f"paths.data_root={config.paths.data_root}",
                "generation.train_rallies=2",
                "generation.val_rallies=1",
                "generation.test_rallies=1",
                "generation.workers=1",
                "generation.seed=992",
                "generation.min_frames=32",
                "rally.max_total_frames=256",
            ],
        )
    generate_dataset(generation)
    settings = training_config(config)
    settings.runtime.run.output_dir.mkdir(parents=True)
    data = RefinerDataModule(settings)
    module = RefinerLightningModule(settings)
    module._contract_clock = data.clock if settings.model.physics_heads else None
    # Initialize the real training lifecycle for the pre-fit objective probe.
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.datamodule = data
    module.trainer = trainer
    module.on_train_start()
    return module, data


def _batch(config: DictConfig) -> Any:
    return next(iter(RefinerDataModule(training_config(config)).train_dataloader()))


def _result(module: Any, batch: Any) -> Any:
    outputs = []
    handle = module.model.register_forward_hook(
        lambda _module, _args, output: outputs.append(output)
    )
    try:
        result = module._compute_supervised_result(batch, "train")
    finally:
        handle.remove()
    assert len(outputs) == 1
    return {**result, "outputs": outputs[0]}


def _parameters(module: Any) -> tuple[ParameterContract, tuple[OptimizerContract, ...]]:
    generator = names_under(module, "model")
    groups = {
        key: names_under(module, f"model.{key}")
        for key in ("input", "blocks", "output", "event_head")
    }
    if module.settings.model.architecture == "flow":
        groups["flow_time"] = names_under(module, "model.flow_time")
    if module.settings.model.physics_heads:
        groups.update(
            {
                key: names_under(module, f"model.{key}")
                for key in ("field_head", "segment_head")
            }
        )
    optimizers = [OptimizerContract("generator", generator, groups)]
    if module.discriminator is not None:
        optimizers.append(
            OptimizerContract(
                "discriminator",
                names_under(module, "discriminator"),
                {
                    "trunk": names_under(module, "discriminator.network.blocks"),
                    "head": names_under(module, "discriminator.network.head"),
                },
            )
        )
    return ParameterContract(
        frozenset(dict(module.named_parameters())), frozenset(), frozenset()
    ), tuple(optimizers)


def _assert_batch(batch: Any) -> None:
    assert batch["coordinates"].shape[:2] == batch["padding"].shape
    assert batch["coordinates"].shape[-1] == 3
    assert (
        batch["coordinates"].dtype == torch.float32
        and batch["missing"].dtype == torch.bool
    )
    assert (
        batch["padding"].dtype == torch.bool
        and batch["missing"][batch["padding"]].all()
    )
    assert batch["event_target"].shape == batch["padding"].shape


def _assert_prediction(output: Any) -> None:
    assert output.coordinates.shape[-1] == 3 and output.event_logits.shape[-1] == 2
    for value in (output.coordinates, output.event_logits):
        assert value.dtype == torch.float32 and torch.isfinite(value).all()
    if output.physics is not None:
        assert output.physics.segment_states is not None
        for value in output.physics:
            assert value.dtype == torch.float32 and torch.isfinite(value).all()


def _predict(module: Any, batch: Any) -> Any:
    return predict_normalized(
        module.model,
        batch["coordinates"],
        batch["missing"],
        batch_size=2,
        seed=42,
        clock=module._contract_clock,
        segment=batch["segment"] if module.settings.model.physics_heads else None,
    )


def _restore(config: DictConfig, path: Path) -> Any:
    module = RefinerLightningModule.load_from_checkpoint(
        path, settings=training_config(config), map_location="cpu", weights_only=False
    )
    module._contract_clock = (
        checkpoint_flight_clock(
            torch.load(path, map_location="cpu", weights_only=False)
        )
        if module.settings.model.physics_heads
        else None
    )
    return module


REGRESSION = ModelFamily(
    name="refiner/regression",
    compose=_compose,
    build=_build,
    batch=_batch,
    supervised_result=_result,
    predict=_predict,
    save=save_checkpoint,
    restore=_restore,
    parameters=_parameters,
    assert_batch=_assert_batch,
    assert_prediction=_assert_prediction,
    fit=lambda *args, **kwargs: fit_contract(
        *args, **kwargs, scheduled_gan=True, validation_batches=1
    ),
)
FLOW = replace(
    REGRESSION,
    name="refiner/flow",
    compose=lambda root, gan: _compose(root, gan, architecture="flow"),
)
PHYSICS = replace(
    REGRESSION,
    name="refiner/physics",
    compose=lambda root, gan: _compose(root, gan, physics=True),
)
