"""Lightweight tests for BLCS Lightning metric visibility."""

from __future__ import annotations

from typing import Any, cast

import torch

from src.tasks.blcs.training.lightning_module import (
    BLCS_TRAJECTORY_METRIC_CONTRACT,
    BLCSLightningModule,
)


def test_stage_logging_omits_batch_aliases_and_eval_loss_components() -> None:
    class _Recorder:
        metric_logging_contract = BLCS_TRAJECTORY_METRIC_CONTRACT

        def __init__(self) -> None:
            self.names: list[str] = []
            self.gan_enabled = True

        def log(self, name: str, value: Any, **kwargs: Any) -> None:
            del value, kwargs
            self.names.append(name)

        def _log_gan_metrics(self, stage: str, metrics: dict[str, Any]) -> None:
            raise AssertionError(
                f"BLCS stage logging bypassed its contract: {stage=}, {metrics=}"
            )

    recorder = _Recorder()
    BLCSLightningModule._log_stage_metrics(
        cast("BLCSLightningModule", recorder),
        "val",
        torch.tensor(1.0),
        {"position_error_m": 0.1, "loss_position": 0.5},
    )

    assert recorder.names == ["val/loss"]
    assert "val/pos_error_m" not in recorder.names
    assert "val/loss_position" not in recorder.names

    recorder.names.clear()
    BLCSLightningModule._log_stage_metrics(
        cast("BLCSLightningModule", recorder),
        "train",
        torch.tensor(1.0),
        {
            "position_error_m": 0.1,
            "loss_position": 0.5,
            "loss_gan_generator": 0.2,
            "loss_gan_discriminator": 0.3,
            "gan_weight": 0.1,
            "gan_phase_active": 1.0,
        },
    )
    assert recorder.names == ["train/loss"]

    recorder.names.clear()
    BLCSLightningModule._log_stage_metrics(
        cast("BLCSLightningModule", recorder),
        "test",
        torch.tensor(1.0),
        {"position_error_m": 0.1, "loss_position": 0.5},
    )
    assert recorder.names == ["test/loss"]
