"""Shared Lightning lifecycle; always evaluate the validation-selected checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor

from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.base.triangulation_residual.configuration import (
    ResidualConfig,
    validate_config,
)
from src.tasks.base.triangulation_residual.contracts import feature_dimension
from src.tasks.base.triangulation_residual.data import ResidualDataModule
from src.tasks.base.triangulation_residual.losses import (
    masked_mean,
    metric_arrays,
    residual_loss,
)
from src.tasks.base.triangulation_residual.model import GeometricResidualModel
from src.utils.schema.court_normalization import (
    add_court_coordinate_normalization,
    validate_court_coordinate_normalization,
)


def checkpoint_contract(config: ResidualConfig) -> dict[str, Any]:
    return {
        "family": config.model.name,
        "schema_version": 1,
        "task": config.task,
        "joints": config.joints,
        "feature_dim": feature_dimension(config.joints),
        "coordinate_frame": "physical_court",
        "output_unit": "metre",
        "root_definition": "coco_hip_midpoint"
        if config.task == "plcs"
        else "ball_center",
        "relative_axes": "court",
        "root_indices": list(config.root_indices),
        "missing_seed": "linear_interpolation_edge_hold_else_root_with_original_valid_mask",
        "camera_estimate_is_input_only": True,
    }


class ResidualLightningModule(BaseLightningModule):
    def __init__(self, config: DictConfig, steps_per_epoch: int | None = None) -> None:
        self.residual_config = validate_config(config)
        super().__init__(config)
        self.steps_per_epoch = steps_per_epoch
        self.model = GeometricResidualModel(
            self.residual_config.task, self.residual_config.model
        )
        self.phase_statistics: dict[str, dict[str, tuple[Tensor, Tensor]]] = {}
        self.last_test_metrics: dict[str, float] = {}

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        add_court_coordinate_normalization(
            checkpoint, artifact="geometry residual checkpoint"
        )
        checkpoint["geometric_residual_contract"] = checkpoint_contract(
            self.residual_config
        )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        validate_court_coordinate_normalization(
            checkpoint, artifact="geometry residual checkpoint"
        )
        if checkpoint.get("geometric_residual_contract") != checkpoint_contract(
            self.residual_config
        ):
            raise ValueError("Incompatible geometric residual checkpoint semantics")

    def forward(
        self, features: Tensor, view_valid: Tensor, time_positions: Tensor
    ) -> dict[str, Tensor]:
        return cast(dict[str, Tensor], self.model(features, view_valid, time_positions))

    def on_train_epoch_start(self) -> None:
        data = cast(Any, self.trainer).datamodule
        if not isinstance(data, ResidualDataModule):
            raise TypeError("ResidualLightningModule requires its paired DataModule")
        data.train_dataset.set_epoch(self.current_epoch)

    def _step(
        self, batch: dict[str, Any], phase: str
    ) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        output = self.model(
            batch["features"], batch["view_valid"], batch["time_positions"]
        )
        loss, parts, world = residual_loss(
            output, batch, self.residual_config.loss, self.residual_config.task
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite residual loss")
        if phase == "train":
            self.log(
                "train/loss", loss, on_step=True, on_epoch=True, batch_size=len(world)
            )
            for key, val in parts.items():
                self.log(
                    f"train/{key}_loss",
                    val,
                    on_step=False,
                    on_epoch=True,
                    batch_size=len(world),
                )
            error = (world.detach() - batch["target_world"]).norm(dim=-1)
            self.log(
                "train/world_mpjpe_m",
                masked_mean(error, batch["frame_valid"][..., None]),
                on_step=False,
                on_epoch=True,
                batch_size=len(world),
            )
        else:
            metrics = metric_arrays(world.detach(), batch, self.residual_config.task)
            statistics = self.phase_statistics[phase]
            for key, values in metrics.items():
                mask = batch["frame_valid"]
                if values.ndim == 3:
                    mask = mask[..., None].expand_as(values)
                total, count = torch.where(mask, values, 0).sum(), mask.sum()
                if key in statistics:
                    old_total, old_count = statistics[key]
                    total, count = total + old_total, count + old_count
                statistics[key] = (total.detach(), count.detach())
            self.log(
                f"{phase}/loss",
                loss,
                on_step=False,
                on_epoch=True,
                batch_size=len(world),
            )
        return loss, output, world

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        return self._step(batch, "train")[0]

    def on_validation_epoch_start(self) -> None:
        self.phase_statistics["val"] = {}

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        self._step(batch, "val")

    def _finish_metrics(self, phase: str) -> dict[str, float]:
        metrics = {}
        for name, (total, count) in self.phase_statistics[phase].items():
            value = total / count.clamp_min(1)
            self.log(f"{phase}/{name}", value, prog_bar=name == "world_mpjpe_m")
            metrics[name] = float(value.cpu())
        return metrics

    def on_validation_epoch_end(self) -> None:
        metrics = self._finish_metrics("val")
        print(
            f"RESIDUAL_VALIDATION task={self.residual_config.task} epoch={self.current_epoch} "
            f"world_mpjpe_m={metrics['world_mpjpe_m']:.6f} "
            f"initial_world_mpjpe_m={metrics['initial_world_mpjpe_m']:.6f}",
            flush=True,
        )

    def on_test_epoch_start(self) -> None:
        self.phase_statistics["test"] = {}
        self._reset_test_prediction_buffer()

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        _, output, world = self._step(batch, "test")
        self.collect_test_predictions(batch, {"output": output, "world": world})

    def test_prediction_payload(
        self, batch: dict[str, Any], result: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        payload = {
            "pred_world": result["world"],
            "target_world": batch["target_world"],
            "initial_world": batch["init_world"],
            "frame_valid": batch["frame_valid"],
            "init_valid": batch["init_valid"],
            "severity": batch["severity"],
            "geometry_attempts": batch["geometry_attempts"],
        }
        payload.update(
            {f"pred_{key}": value for key, value in result["output"].items()}
        )
        return {key: self._to_numpy(value) for key, value in payload.items()}

    def on_test_epoch_end(self) -> None:
        self.last_test_metrics = self._finish_metrics("test")
        self.save_test_predictions(metrics=self.last_test_metrics)


class ResidualTrainingRunner(BaseTrainingRunner):
    def prepare_config(self, config: Any) -> None:
        validate_config(config)

    def build_datamodule(self, config: Any) -> ResidualDataModule:
        data = ResidualDataModule(validate_config(config))
        data.setup()
        output_dir = data.config.runtime.run.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "split_audit.json").write_text(
            json.dumps(data.split_audit, indent=2, default=str)
        )
        return data

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> ResidualLightningModule:
        return ResidualLightningModule(config, steps_per_epoch)

    def test_after_fit(
        self,
        trainer: pl.Trainer,
        lightning_module: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        callbacks: list[Any],
    ) -> None:
        monitored = [
            c
            for c in callbacks
            if isinstance(c, ModelCheckpoint) and c.monitor == "val/world_mpjpe_m"
        ]
        if (
            len(monitored) != 1
            or not monitored[0].best_model_path
            or monitored[0].best_model_score is None
        ):
            raise RuntimeError(
                "A validation-selected best checkpoint is required for residual testing"
            )
        best = Path(monitored[0].best_model_path)
        if not best.is_file():
            raise FileNotFoundError(best)
        if not isinstance(lightning_module, ResidualLightningModule):
            raise TypeError("Residual runner/model mismatch")
        results = trainer.test(
            lightning_module,
            datamodule=datamodule,
            ckpt_path=str(best),
            weights_only=False,
        )
        out = lightning_module.residual_config.runtime.run.output_dir
        (out / "evaluation.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(best),
                    "selection": "minimum val/world_mpjpe_m",
                    "best_epoch_score": float(monitored[0].best_model_score.cpu()),
                    "test": results,
                },
                indent=2,
            )
        )
        print(f"BEST_CHECKPOINT={best}", flush=True)
