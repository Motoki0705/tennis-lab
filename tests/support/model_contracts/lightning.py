"""Audit actual Lightning optimizer updates, including manual GAN isolation."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle

from tests.support.model_contracts.parameters import (
    OptimizerContract,
    ParameterContract,
    assert_gradients,
    assert_optimizer_coverage,
    assert_update,
    snapshot,
)


class TrainingAudit(pl.Callback):
    """Observe real backward/step boundaries without replacing training behavior."""

    def __init__(
        self,
        parameters: ParameterContract,
        optimizers: tuple[OptimizerContract, ...],
        *,
        gan: bool,
        scheduled_gan: bool = False,
    ) -> None:
        self.parameters = parameters
        self.contracts = optimizers
        self.gan = gan
        self.scheduled_gan = scheduled_gan
        self.updates: Counter[str] = Counter()
        self.backwards: list[str] = []
        self.losses: list[float] = []
        self._handles: list[RemovableHandle] = []
        self._outside_grads: dict[str, Tensor | None] = {}
        self._current: OptimizerContract | None = None
        self._before_step: dict[str, Tensor] = {}

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        assert_optimizer_coverage(
            pl_module, self.parameters, trainer.optimizers, self.contracts
        )
        for optimizer, contract in zip(trainer.optimizers, self.contracts, strict=True):
            self._handles.extend(
                [
                    optimizer.register_step_pre_hook(
                        self._pre_step(pl_module, contract)
                    ),
                    optimizer.register_step_post_hook(
                        self._post_step(pl_module, contract)
                    ),
                ]
            )

    def on_train_batch_start(
        self, trainer: pl.Trainer, pl_module: Any, batch: Any, batch_idx: int
    ) -> None:
        # Exercise supervised-only G first, then real D/G training on batch two.
        if self.gan and not self.scheduled_gan and self.updates["generator"] == 1:
            pl_module.activate_gan_phase(int(trainer.current_epoch))
            pl_module.set_gan_weight(0.1)

    def on_before_backward(
        self, trainer: pl.Trainer, pl_module: Any, loss: Tensor
    ) -> None:
        assert loss.ndim == 0 and torch.isfinite(loss), "nonfinite/scalar training loss"
        self.losses.append(float(loss.detach()))
        enabled = frozenset(
            name for name, p in pl_module.named_parameters() if p.requires_grad
        )
        matches = [
            contract
            for contract in self.contracts
            if contract.parameters - self.parameters.optimizer_frozen == enabled
        ]
        assert len(matches) == 1, (
            f"backward trainable names do not identify one optimizer: {sorted(enabled)}"
        )
        self._current = matches[0]
        self._outside_grads = {
            name: None if p.grad is None else p.grad.detach().clone()
            for name, p in pl_module.named_parameters()
            if name not in self._current.parameters
        }

    def on_after_backward(self, trainer: pl.Trainer, pl_module: Any) -> None:
        assert self._current is not None
        assert_gradients(
            pl_module,
            active=self.parameters.active & self._current.parameters,
            absent=self.parameters.frozen | self.parameters.unused,
        )
        named = dict(pl_module.named_parameters())
        for name, previous in self._outside_grads.items():
            current = named[name].grad
            assert (
                (current is None)
                if previous is None
                else (current is not None and torch.equal(current, previous))
            ), f"{self._current.name}: backward changed unowned gradient: {name}"
        self.backwards.append(self._current.name)

    def _pre_step(self, module: Any, contract: OptimizerContract) -> Any:
        def check(optimizer: Optimizer, args: Any, kwargs: Any) -> None:
            # Automatic optimization runs backward inside the step closure.
            self._before_step = snapshot(module)

        return check

    def _post_step(self, module: Any, contract: OptimizerContract) -> Any:
        def check(optimizer: Optimizer, args: Any, kwargs: Any) -> None:
            assert self._current is not None and self._current.name == contract.name
            assert_update(
                module,
                self._before_step,
                replace(
                    contract, parameters=contract.parameters & self.parameters.active
                ),
            )
            self.updates[contract.name] += 1

        return check

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


def fit_contract(
    module: pl.LightningModule,
    datamodule: pl.LightningDataModule | DataLoader[Any],
    parameters: ParameterContract,
    optimizers: tuple[OptimizerContract, ...],
    root: Path,
    *,
    gan: bool,
    scheduled_gan: bool = False,
    validation_batches: int = 0,
) -> TrainingAudit:
    audit = TrainingAudit(parameters, optimizers, gan=gan, scheduled_gan=scheduled_gan)
    trainer = pl.Trainer(
        default_root_dir=root,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_steps=3 if gan else 1,
        max_epochs=2,
        limit_train_batches=1,
        limit_val_batches=validation_batches,
        reload_dataloaders_every_n_epochs=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[audit],
    )
    try:
        if isinstance(datamodule, DataLoader):
            trainer.fit(module, train_dataloaders=datamodule)
        else:
            trainer.fit(module, datamodule=datamodule)
    finally:
        audit.close()
    assert trainer.global_step == (3 if gan else 1)
    assert audit.backwards == (
        ["generator", "discriminator", "generator"] if gan else ["generator"]
    )
    assert audit.updates == (
        {"generator": 2, "discriminator": 1} if gan else {"generator": 1}
    )
    assert audit.losses
    parameters.validate(module)  # toggling must restore even unused/frozen flags
    if gan:
        strategy = module.gan_training
        assert strategy.supervised_only_step_count == 1
        assert strategy.hybrid_gan_step_count == 1
    return audit
