"""One-step CPU training smoke tests for both track-query tasks."""

from __future__ import annotations

from pathlib import Path

import pytest
import pytorch_lightning as pl
from omegaconf import OmegaConf

from src.tasks.ball_tracking.data import BallTrackingDataModule
from src.tasks.ball_tracking.training import BallTrackingLightningModule
from src.tasks.player_tracking.data import PlayerTrackingDataModule
from src.tasks.player_tracking.training import PlayerTrackingLightningModule

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.mark.parametrize(
    ("config_path", "datamodule_class", "module_class"),
    [
        (
            "src/tasks/ball_tracking/configs/train.yaml",
            BallTrackingDataModule,
            BallTrackingLightningModule,
        ),
        (
            "src/tasks/player_tracking/configs/train.yaml",
            PlayerTrackingDataModule,
            PlayerTrackingLightningModule,
        ),
    ],
)
def test_tracking_task_runs_one_training_and_validation_step(
    tmp_path: Path, config_path, datamodule_class, module_class
) -> None:
    config = OmegaConf.load(Path(config_path))
    config.data.split_sizes.train = 4
    config.data.split_sizes.val = 4
    config.data.batch_size = 2
    datamodule = datamodule_class(config)
    module = module_class(config)
    trainer = pl.Trainer(
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
    trainer.fit(module, datamodule=datamodule)
    assert trainer.global_step == 1
