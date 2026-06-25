"""1-step training smoke test for BaseLightningModule.

Runs a real ``Trainer.fit`` for a single optimization step on a tiny in-memory
dataset to verify that ``configure_optimizers`` (optimizer + scheduler with the
step/epoch interval logic) wires into the Lightning loop without error. CPU-only,
no checkpoints, minimal data.
"""

from __future__ import annotations

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.tasks.base.training.lightning_module import BaseLightningModule

pytestmark = [pytest.mark.integration, pytest.mark.slow]


class _SmokeModule(BaseLightningModule):
    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.net = nn.Linear(4, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.net(x).squeeze(-1)
        loss = nn.functional.mse_loss(pred, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.net(x).squeeze(-1)
        loss = nn.functional.mse_loss(pred, y)
        self.log("val/loss", loss)
        return loss


def _loader(n: int = 8, batch: int = 4) -> DataLoader:
    x = torch.randn(n, 4)
    y = torch.randn(n)
    return DataLoader(TensorDataset(x, y), batch_size=batch)


def _trainer(tmp_path) -> pl.Trainer:
    return pl.Trainer(
        max_steps=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
    )


def test_one_step_with_step_scheduler(tmp_path) -> None:
    module = _SmokeModule({"training": {"max_epochs": 1, "steps_per_epoch": 2}})
    trainer = _trainer(tmp_path)
    trainer.fit(module, _loader(), _loader())
    assert trainer.global_step == 1


def test_one_step_with_epoch_warmup_scheduler(tmp_path) -> None:
    module = _SmokeModule({"training": {"max_epochs": 4, "warmup_epochs": 2}})
    trainer = _trainer(tmp_path)
    trainer.fit(module, _loader(), _loader())
    assert trainer.global_step == 1
    # optimizer actually got constructed and is an AdamW
    opt = trainer.optimizers[0]
    assert isinstance(opt, torch.optim.AdamW)
