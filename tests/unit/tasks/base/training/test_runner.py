"""Unit tests for config-driven BaseTrainingRunner construction."""

from __future__ import annotations

from typing import Any, cast

from omegaconf import OmegaConf

from src.tasks.base.training.runner import BaseTrainingRunner


def test_build_trainer_forwards_dataloader_reload_interval() -> None:
    config = OmegaConf.create(
        {
            "run": {"gpus": 0, "fast_dev_run": False},
            "training": {
                "trainer": {
                    "max_epochs": 2,
                    "gradient_clip_val": None,
                    "deterministic": True,
                    "precision": None,
                    "log_every_n_steps": 1,
                    "check_val_every_n_epoch": 1,
                    "reload_dataloaders_every_n_epochs": 1,
                    "enable_progress_bar": False,
                    "enable_model_summary": False,
                }
            },
        }
    )

    trainer = BaseTrainingRunner().build_trainer(
        config,
        callbacks=[],
        logger=cast(Any, False),
    )

    assert trainer.reload_dataloaders_every_n_epochs == 1
