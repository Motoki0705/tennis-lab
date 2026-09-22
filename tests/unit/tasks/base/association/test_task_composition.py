"""Task recipes must configure the model that the shared training runner executes."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.base.training.association_losses import association_loss
from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.blcs.training.runner import BLCSTrainingRunner
from src.tasks.plcs.training.runner import PLCSTrainingRunner
from src.utils.paths import PROJECT_ROOT


def recipe(task: str) -> Any:
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / f"src/tasks/{task}/configs"), version_base="1.3"
    ):
        return compose(config_name="train_association")


@pytest.mark.parametrize("task", ["plcs", "blcs"])
def test_task_runner_uses_single_model_configuration_and_shared_training(task):
    cfg = recipe(task)
    assert (cfg.model.hidden_dim, cfg.model.num_stages, cfg.model.num_heads) == (
        512,
        12,
        8,
    )
    assert cfg.training.trainer.max_epochs == 60
    assert cfg.training.compile.enabled
    assert cfg.run.resume is None and cfg.run.init_weights is None
    cfg.model.hidden_dim = 48
    cfg.model.num_heads = 4
    cfg.model.num_stages = 2
    cfg.model.rope_dim = 12
    cfg.model.ffn_dim = 96
    runner = PLCSTrainingRunner() if task == "plcs" else BLCSTrainingRunner()
    runner.prepare_config(cfg)
    dm = runner.build_datamodule(cfg)
    module = runner.build_lightning_module(cfg, dm)
    assert isinstance(module, BaseLightningModule)
    assert module.compilation_targets() == {"model": module.model}
    assert module.model.config.hidden_dim == 48
    assert len(module.model.stages) == 2
    assert module.model_io.model is module.model


@pytest.mark.parametrize("task", ["plcs", "blcs"])
@pytest.mark.parametrize("field", ["association.model", "model.cswa"])
def test_legacy_or_unused_model_configuration_fails_explicitly(task, field):
    cfg = recipe(task)
    OmegaConf.update(cfg, field, {}, force_add=True)
    runner = PLCSTrainingRunner() if task == "plcs" else BLCSTrainingRunner()
    with pytest.raises(ValueError):
        runner.prepare_config(
            cfg
        ) if task == "plcs" else runner.validate_runtime_config(cfg)


def test_matching_is_float32_inside_mixed_precision_training():
    logits = torch.randn(1, 2, 3, 4, 11, dtype=torch.bfloat16, requires_grad=True)
    side_logits = torch.randn(1, 2, dtype=torch.bfloat16, requires_grad=True)
    target = torch.arange(4).expand(1, 2, 3, 4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        values = association_loss(
            {"object_id_logits": logits, "side_logits": side_logits},
            target,
            torch.ones_like(target, dtype=torch.bool),
            torch.tensor([[False, True]]),
            torch.ones(1, 2, dtype=torch.bool),
            torch.tensor([0]),
        )
    values["loss"].backward()
    assert values["loss"].dtype == torch.float32
    assert torch.isfinite(values["loss"])
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert side_logits.grad is not None and torch.isfinite(side_logits.grad).all()
