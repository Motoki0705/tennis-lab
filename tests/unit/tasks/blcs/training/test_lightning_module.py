"""Lightweight tests for BLCS Lightning metric visibility."""

from __future__ import annotations

from typing import Any, cast

import torch

from src.tasks.blcs.training.lightning_module import (
    BLCS_TRAJECTORY_METRIC_CONTRACT,
    BLCSLightningModule,
)


def test_stage_logging_records_loss_components_without_batch_metric_aliases() -> None:
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
        {"trajectory_rmse_m": 0.1, "loss_position": 0.5},
    )

    assert recorder.names == ["val/loss", "val/loss_position"]
    assert "val/pos_error_m" not in recorder.names
    assert "val/loss_position" in recorder.names

    recorder.names.clear()
    BLCSLightningModule._log_stage_metrics(
        cast("BLCSLightningModule", recorder),
        "train",
        torch.tensor(1.0),
        {
            "trajectory_rmse_m": 0.1,
            "loss_position": 0.5,
            "loss_gan_generator": 0.2,
            "loss_gan_discriminator": 0.3,
            "gan_weight": 0.1,
            "gan_phase_active": 1.0,
        },
    )
    assert recorder.names == [
        "train/loss",
        "train/loss_position",
        "train/loss_gan_generator",
        "train/loss_gan_discriminator",
    ]

    recorder.names.clear()
    BLCSLightningModule._log_stage_metrics(
        cast("BLCSLightningModule", recorder),
        "test",
        torch.tensor(1.0),
        {"trajectory_rmse_m": 0.1, "loss_position": 0.5},
    )
    assert recorder.names == ["test/loss", "test/loss_position"]


def test_train_epoch_omits_uncomputed_rollout_but_validation_logs_it() -> None:
    from src.tasks.blcs.model_io import BLCSTrajectoryPrediction
    from src.tasks.blcs.training.metrics import BLCSMetrics
    from tests.support.tasks.blcs_physics import physics_sample, prepared_samples

    batch, target = prepared_samples([physics_sample(8)])
    prediction = BLCSTrajectoryPrediction(target.position, target.velocity)

    class Recorder:
        metric_logging_contract = BLCS_TRAJECTORY_METRIC_CONTRACT

        def __init__(self, horizons):
            self.tracker = BLCSMetrics(rollout_horizons=horizons)
            self.tracker.update(prediction, target, observed=batch["ball_vis"])
            self.names = []

        def _metric_tracker_for_stage(self, stage):
            return self.tracker

        def _epoch_metric_log_name(self, stage, name):
            return f"{stage}/{name}"

        def log(self, name, value, **kwargs):
            self.names.append(name)

    train = Recorder(())
    BLCSLightningModule._flush_stage_metrics(train, "train")
    assert "train/trajectory_rmse_m" in train.names
    assert not any("rollout" in name for name in train.names)
    validation = Recorder((1, 3))
    BLCSLightningModule._flush_stage_metrics(validation, "val")
    assert "val/rollout_position_rmse_m" in validation.names
    assert "val/rollout_velocity_rmse_mps" in validation.names
