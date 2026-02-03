"""Event detection training runner using BaseTrainingRunner."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger

from src.base.training.runner import BaseTrainingRunner
from src.evnet_detection.data.datamodule import EventDetectionDataModule
from src.evnet_detection.training.lightning_module import EventDetectionLightningModule

if TYPE_CHECKING:
    from omegaconf import DictConfig


class EventDetectionTrainingRunner(BaseTrainingRunner):
    """Training runner for event detection models.

    Supports both UV and 3D trajectory input types (determined by config).
    """

    def build_datamodule(self, config: DictConfig) -> pl.LightningDataModule:
        """Build the EventDetectionDataModule."""
        return EventDetectionDataModule(config)

    def build_lightning_module(
        self,
        config: DictConfig,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        """Build the EventDetectionLightningModule."""
        _ = datamodule, steps_per_epoch
        return EventDetectionLightningModule(config)

    def checkpoint_prefix(self, config: Any) -> str:
        """Use 'event-detection' prefix for checkpoints."""
        return "event-detection"

    def early_stopping_patience(self, config: Any) -> int:
        """Event detection uses patience=5."""
        return 5

    def early_stopping_min_delta(self, config: Any) -> float | None:
        """Event detection uses min_delta=1e-3."""
        return 1.0e-3

    def trainer_kwargs(
        self, config: Any, accelerator: str, devices: int
    ) -> dict[str, Any]:
        """Add precision and log_every_n_steps settings."""
        kwargs: dict[str, Any] = {"log_every_n_steps": 50}
        if accelerator == "gpu":
            kwargs["precision"] = "16-mixed"
        return kwargs

    def callbacks_extra(
        self, config: Any, datamodule: pl.LightningDataModule, logger: TensorBoardLogger
    ) -> list[Any]:
        """No extra callbacks for event detection."""
        return []

    def run_dry_run(self, config: Any, output_dir: Path) -> None:
        """Run a 1-batch fit to validate wiring with dry_run config overrides."""
        print("Running dry run (no training)...")
        self._force_cpu_for_dry_run()

        overrides = OmegaConf.create({
            "run": {"dry_run": True},
            "data": {
                "batch_size": 2,
                "num_workers": 0,
            },
        })
        OmegaConf.set_struct(config, False)
        dry_cfg = OmegaConf.merge(config, overrides)
        OmegaConf.set_struct(dry_cfg, False)

        datamodule = self.build_datamodule(dry_cfg)
        datamodule.setup(stage="fit")
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        self._print_batch_shapes(batch)
        self.dry_run_postprocess(batch, output_dir)

        lightning_module = self.build_lightning_module(dry_cfg, datamodule)

        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(lightning_module, datamodule=datamodule)
        print(f"Dry run complete. Outputs saved to {output_dir}")
