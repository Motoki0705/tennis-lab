"""1-step training smoke test for BaseLightningModule.

Runs a real ``Trainer.fit`` for a single optimization step on a tiny in-memory
dataset to verify that ``configure_optimizers`` (optimizer + scheduler with the
step/epoch interval logic) wires into the Lightning loop without error. CPU-only,
no checkpoints, minimal data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.tasks.base.configuration import CompileConfig
from src.tasks.base.training.compilation import compile_modules
from src.tasks.base.training.lightning_module import BaseLightningModule

pytestmark = [pytest.mark.integration, pytest.mark.slow]


class _SmokeModule(BaseLightningModule):
    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.model = nn.Linear(4, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x).squeeze(-1)
        loss = nn.functional.mse_loss(pred, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x).squeeze(-1)
        loss = nn.functional.mse_loss(pred, y)
        self.log("val/loss", loss)
        return loss


def _loader(n: int = 8, batch: int = 4, *, drop_last: bool = False) -> DataLoader:
    x = torch.randn(n, 4)
    y = torch.randn(n)
    return DataLoader(TensorDataset(x, y), batch_size=batch, drop_last=drop_last)


def _trainer(tmp_path) -> pl.Trainer:
    return pl.Trainer(
        max_epochs=1,
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


def _config(
    tmp_path: Path,
    *,
    max_epochs: int,
    steps_per_epoch: int | None,
    warmup_steps: int | None,
    warmup_epochs: int | None,
) -> dict[str, object]:
    return {
        "paths": {
            "project_root": str(tmp_path),
            "data_root": "data",
            "checkpoint_root": "checkpoints",
            "artifact_root": "artifacts",
            "output_root": "outputs",
            "cache_root": ".cache",
            "external_asset_root": "external",
        },
        "training": {
            "trainer": {
                "max_epochs": max_epochs,
                "gradient_clip_val": None,
                "deterministic": True,
                "precision": "32-true",
                "log_every_n_steps": 1,
                "check_val_every_n_epoch": 1,
                "accumulate_grad_batches": 1,
                "reload_dataloaders_every_n_epochs": 0,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "benchmark": False,
            },
            "learning_rate": 1.0e-3,
            "weight_decay": 0.0,
            "warmup_steps": warmup_steps,
            "warmup_epochs": warmup_epochs,
            "min_lr": 0.0,
            "steps_per_epoch": steps_per_epoch,
            "optimizer": {"betas": [0.9, 0.999]},
            "checkpoint": {
                "enabled": False,
                "filename": "model-{epoch}",
                "monitor": "val/loss",
                "mode": "min",
                "save_top_k": 1,
                "save_last": False,
            },
            "early_stopping": {
                "enabled": False,
                "monitor": "val/loss",
                "mode": "min",
                "patience": 1,
                "min_delta": 0.0,
                "check_on_train_epoch_end": False,
            },
            "lr_monitor": {"enabled": False, "interval": "step"},
            "qualitative_logging": {
                "enabled": False,
                "every_n_epochs": 1,
                "num_samples": 1,
                "selection_mode": "random",
                "selected_indices": None,
            },
            "gan": {
                "enabled": False,
                "target_weight": 0.0,
                "warmup_epochs": 1,
                "generator_gradient_clip_val": None,
                "discriminator_gradient_clip_val": None,
                "transition": {"start_epoch": 0},
            },
            "compile": {
                "enabled": True,
                "backend": "inductor",
                "mode": "reduce-overhead",
                "fullgraph": False,
                "dynamic": False,
            },
            "matmul_precision": "high",
            "allow_tf32": False,
        },
    }


def test_one_step_with_step_scheduler(tmp_path) -> None:
    module = _SmokeModule(
        _config(
            tmp_path,
            max_epochs=1,
            steps_per_epoch=1,
            warmup_steps=0,
            warmup_epochs=None,
        )
    )
    trainer = _trainer(tmp_path)
    trainer.fit(module, _loader(), _loader())
    assert trainer.global_step == 1


@pytest.mark.parametrize(
    "accumulation,limit,max_steps,drop_last,expected_steps",
    [
        (1, 1.0, -1, True, 12),
        (2, 1.0, -1, True, 6),
        (1, 2, -1, True, 6),
        (1, 1.0, 5, True, 5),
        (1, 1.0, -1, False, 15),
    ],
    ids=["matching", "accumulation", "limit", "max_steps", "partial_batch"],
)
def test_explicit_step_budget_matches_actual_updates_before_training(
    tmp_path, accumulation, limit, max_steps, drop_last, expected_steps
) -> None:
    # 9 samples / batch 2: four batches with drop_last, five without it.
    module = _SmokeModule(
        _config(
            tmp_path,
            max_epochs=3,
            steps_per_epoch=4,
            warmup_steps=0,
            warmup_epochs=None,
        )
    )
    before = {key: value.clone() for key, value in module.state_dict().items()}
    trainer = pl.Trainer(
        max_epochs=3,
        max_steps=max_steps,
        accumulate_grad_batches=accumulation,
        limit_train_batches=limit,
        limit_val_batches=0,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    loader = _loader(n=9, batch=2, drop_last=drop_last)
    if expected_steps == 12:
        trainer.fit(module, loader)
        assert trainer.global_step == 12
        assert trainer.lr_scheduler_configs[0].scheduler.T_max == 12
        return

    with pytest.raises(ValueError, match="Explicit step schedule budget") as error:
        trainer.fit(module, loader)
    for detail in (
        "steps_per_epoch=4",
        "configured_total_steps=12",
        f"expected_total_steps={expected_steps}",
        f"num_training_batches={len(loader) if limit == 1.0 else limit}",
        f"accumulate_grad_batches={accumulation}",
        f"limit_train_batches={limit}",
        f"max_steps={max_steps}",
        "max_epochs=3",
    ):
        assert detail in str(error.value)
    assert trainer.global_step == 0
    assert not trainer.optimizers[0].state
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_one_step_with_epoch_warmup_scheduler(tmp_path) -> None:
    module = _SmokeModule(
        _config(
            tmp_path,
            max_epochs=4,
            steps_per_epoch=None,
            warmup_steps=None,
            warmup_epochs=2,
        )
    )
    trainer = _trainer(tmp_path)
    trainer.fit(module, _loader(), _loader())
    assert trainer.global_step == 1
    # optimizer actually got constructed and is an AdamW
    opt = trainer.optimizers[0]
    assert isinstance(opt, torch.optim.AdamW)


def test_one_step_with_compiled_primary_model(tmp_path) -> None:
    module = _SmokeModule(
        _config(
            tmp_path,
            max_epochs=1,
            steps_per_epoch=1,
            warmup_steps=0,
            warmup_epochs=None,
        )
    )
    compile_config = CompileConfig(
        enabled=True,
        backend=cast(Any, "aot_eager"),
        mode="default",
        fullgraph=False,
        dynamic=False,
    )

    assert compile_modules(module.compilation_targets(), compile_config) == ("model",)
    trainer = _trainer(tmp_path)
    trainer.fit(module, _loader(), _loader())

    assert trainer.global_step == 1
    assert module.model._compiled_call_impl is not None
