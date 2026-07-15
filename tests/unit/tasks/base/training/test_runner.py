"""Unit tests for shared Trainer configuration forwarding."""

from typing import Any, cast

import pytest
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger

from src.tasks.base.training.runner import BaseTrainingRunner


class _CpuRunner(BaseTrainingRunner):
    def select_devices(self, config: Any) -> tuple[str, int]:
        return "cpu", 1


def test_build_trainer_forwards_overfit_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_trainer(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(pl, "Trainer", fake_trainer)
    config = OmegaConf.create(
        {
            "run": {"fast_dev_run": False, "gpus": 0},
            "training": {
                "trainer": {
                    "max_epochs": 10,
                    "deterministic": True,
                    "log_every_n_steps": 1,
                    "check_val_every_n_epoch": 1,
                    "gradient_clip_val": None,
                    "precision": None,
                    "overfit_batches": 1,
                }
            },
        }
    )

    _CpuRunner().build_trainer(
        config,
        callbacks=[],
        logger=cast(TensorBoardLogger, object()),
    )

    assert captured["overfit_batches"] == 1
