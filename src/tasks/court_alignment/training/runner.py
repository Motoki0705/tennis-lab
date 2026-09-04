"""Hydra composition boundary for court-alignment training and evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn

from src.tasks.base.configuration import TrainingRuntimeConfig
from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.base.training.repro import resolve_queue_repro_dir
from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.court_alignment.configuration import (
    CNN_MODEL_TARGET,
    DINO_MODEL_TARGET,
    CourtAlignmentRuntimeConfig,
)
from src.tasks.court_alignment.models.checkpoint import (
    load_court_alignment_model_checkpoint,
)
from src.tasks.court_alignment.training.detr_lightning_module import (
    DinoCourtAlignmentLightningModule,
)
from src.tasks.court_alignment.training.lightning_module import (
    CourtAlignmentLightningModule,
)
from src.utils.configuration import PathRole


class CourtAlignmentTrainingRunner(BaseTrainingRunner):
    """Build the shared procedural data pipeline and configured alignment model."""

    def validate_runtime_config(self, config: Any) -> TrainingRuntimeConfig:
        evaluation = (
            isinstance(config, (DictConfig, Mapping)) and "evaluation" in config
        )
        return CourtAlignmentRuntimeConfig.from_config(
            config,
            evaluation=evaluation,
        ).runtime

    def prepare_config(self, config: Any) -> None:
        evaluation = (
            isinstance(config, (DictConfig, Mapping)) and "evaluation" in config
        )
        _ = CourtAlignmentRuntimeConfig.from_config(
            config,
            evaluation=evaluation,
        )

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        datamodule = instantiate(config.data)
        if not isinstance(datamodule, pl.LightningDataModule):
            raise TypeError(
                "data._target_ must construct a pytorch_lightning.LightningDataModule."
            )
        return datamodule

    def maybe_load_init_weights(
        self,
        config: TrainingRuntimeConfig,
        lightning_module: pl.LightningModule,
    ) -> None:
        """Strictly warm-start the task CNN from a historical Lightning payload."""

        init_path = config.run.init_weights
        if init_path is None:
            return
        checkpoint_path = config.resolver.validate(PathRole.CHECKPOINT, init_path)
        model = getattr(lightning_module, "model", None)
        if not isinstance(model, nn.Module):
            raise TypeError(
                "Court-alignment init_weights requires lightning_module.model "
                "to be a torch module."
            )
        metadata = load_court_alignment_model_checkpoint(model, checkpoint_path)
        print(
            "[init_weights] strict-loaded "
            f"{metadata['state_dict_key_count']} model tensors from {checkpoint_path}; "
            "optimizer, scheduler, loop, and epoch state were not loaded."
        )

    def test_checkpoint_path(self, config: Any, trainer: pl.Trainer) -> str:
        """Require and select the same validation-best checkpoint for testing."""
        runtime = self.validate_runtime_config(config)
        if not runtime.training.checkpoint.enabled:
            raise RuntimeError(
                "Court alignment cannot test after fit without checkpointing."
            )
        checkpoint_callback = trainer.checkpoint_callback
        if not isinstance(checkpoint_callback, ModelCheckpoint):
            raise RuntimeError(
                "Court alignment trainer has no ModelCheckpoint callback."
            )
        best_model_path = checkpoint_callback.best_model_path
        if not isinstance(best_model_path, str) or not best_model_path.strip():
            raise RuntimeError(
                "Court alignment training produced no validation-best checkpoint."
            )
        return "best"

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        del datamodule
        model_target = str(config.model._target_)
        if model_target == DINO_MODEL_TARGET:
            module: BaseLightningModule = DinoCourtAlignmentLightningModule(config)
        elif model_target == CNN_MODEL_TARGET:
            module = CourtAlignmentLightningModule(config)
        else:
            raise ValueError(f"Unsupported court-alignment model target: {model_target!r}.")
        module.steps_per_epoch = steps_per_epoch
        return module

    def evaluate(self, config: DictConfig) -> None:
        """Evaluate one explicit checkpoint without fitting."""
        court_runtime = CourtAlignmentRuntimeConfig.from_config(
            config,
            evaluation=True,
        )
        runtime = court_runtime.runtime
        self.seed_everything(runtime)
        self.apply_runtime_settings(runtime)

        checkpoint_path = court_runtime.evaluation_checkpoint
        if checkpoint_path is None:
            raise ValueError(
                "evaluation.checkpoint_path must be set to an existing checkpoint."
            )
        checkpoint_path = runtime.resolver.validate(
            PathRole.CHECKPOINT,
            checkpoint_path,
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Court-alignment checkpoint does not exist: {checkpoint_path}"
            )

        output_dir = self.prepare_output_dir(runtime)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.save_config(config, output_dir)
        datamodule = self.build_datamodule(config)
        module = self.build_lightning_module(config, datamodule)
        accelerator, devices = self.select_devices(config)
        trainer_cfg = runtime.training.trainer
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            precision=trainer_cfg.precision,
            deterministic=trainer_cfg.deterministic,
            benchmark=trainer_cfg.benchmark,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=trainer_cfg.enable_progress_bar,
            enable_model_summary=trainer_cfg.enable_model_summary,
        )
        trainer.test(
            module,
            datamodule=datamodule,
            ckpt_path=str(checkpoint_path),
            weights_only=False,
        )

        repro_dir = resolve_queue_repro_dir()
        if repro_dir is not None:
            (repro_dir / "output_dir.txt").write_text(
                f"{checkpoint_path.parent}\n", encoding="utf-8"
            )
        print(f"Evaluation complete. Checkpoint: {checkpoint_path}")


__all__ = ["CourtAlignmentTrainingRunner"]
