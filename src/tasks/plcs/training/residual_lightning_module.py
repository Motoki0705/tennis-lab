"""PLCS Lightning lifecycle; always evaluate the validation-selected checkpoint."""

from __future__ import annotations

import json
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor

from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.plcs.configuration import validate_residual_config
from src.tasks.plcs.data.residual_datamodule import ResidualDataModule
from src.tasks.plcs.model_io.residual_checkpoint import (
    checkpoint_contract,
    validate_residual_checkpoint,
)
from src.tasks.plcs.model_io.residual_contracts import (
    feature_dimension,
    validate_model_inputs,
)
from src.tasks.plcs.models.triangulation_residual import GeometricResidualModel
from src.tasks.plcs.training.residual_losses import (
    masked_mean,
    residual_loss,
)
from src.tasks.plcs.training.residual_metrics import (
    metric_arrays,
    prediction_diagnostics,
)
from src.utils.schema.court_normalization import (
    add_court_coordinate_normalization,
)


class ResidualLightningModule(BaseLightningModule):
    def __init__(self, config: DictConfig, steps_per_epoch: int | None = None) -> None:
        self.residual_config = validate_residual_config(config)
        super().__init__(config)
        self.steps_per_epoch = steps_per_epoch
        self.model = GeometricResidualModel(self.residual_config.model)
        self.phase_statistics: dict[str, dict[str, tuple[Tensor, Tensor]]] = {}
        self.validation_predictions: dict[str, list[np.ndarray]] = {}
        self.last_test_metrics: dict[str, float] = {}

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        add_court_coordinate_normalization(
            checkpoint, artifact="geometry residual checkpoint"
        )
        checkpoint["geometric_residual_contract"] = checkpoint_contract(
            self.residual_config
        )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        validated = validate_residual_checkpoint(checkpoint)
        if validated["geometric_residual_contract"] != checkpoint_contract(
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
        validate_model_inputs(
            batch["features"],
            batch["view_valid"],
            batch["time_positions"],
            input_dim=feature_dimension(self.residual_config.joints),
        )
        output = self.model(
            batch["features"], batch["view_valid"], batch["time_positions"]
        )
        loss, parts, world = residual_loss(
            output,
            batch,
            self.residual_config.loss,
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
            metrics = metric_arrays(world.detach(), batch)
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
        self.validation_predictions = {}

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        _, output, world = self._step(batch, "val")
        payload = self.test_prediction_payload(
            batch, {"output": output, "world": world}
        )
        for key, value in payload.items():
            self.validation_predictions.setdefault(key, []).append(value)

    def _finish_metrics(self, phase: str) -> dict[str, float]:
        metrics = {}
        for name, (total, count) in self.phase_statistics[phase].items():
            value = total / count.clamp_min(1)
            self.log(f"{phase}/{name}", value, prog_bar=name == "world_mpjpe_m")
            metrics[name] = float(value.cpu())
        return metrics

    def on_validation_epoch_end(self) -> None:
        metrics = self._finish_metrics("val")
        diagnostics = prediction_diagnostics(
            {
                key: np.concatenate(values)
                for key, values in self.validation_predictions.items()
            },
        )
        for key, value in (
            ("world_median_m", diagnostics["world"]["predicted_m"]["median"]),
            ("initial_world_median_m", diagnostics["world"]["initial_m"]["median"]),
            ("improved_fraction", diagnostics["world"]["improved_fraction"]),
            (
                "sample_improved_fraction",
                diagnostics["world"]["sample_improved_fraction"],
            ),
        ):
            self.log(f"val/{key}", float(value))
        if not self.trainer.sanity_checking:
            out = self.residual_config.runtime.run.output_dir / "validation_diagnostics"
            out.mkdir(parents=True, exist_ok=True)
            (out / f"epoch_{self.current_epoch:03d}.json").write_text(
                json.dumps(diagnostics, indent=2, allow_nan=False)
            )
        print(
            f"RESIDUAL_VALIDATION task=plcs epoch={self.current_epoch} "
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
        for key in (
            "corruption_family",
            "num_views",
            "persistent_fraction",
            "persistent_mask",
            "calibration_attempts",
            "calibration_failed_candidates",
        ):
            if key in batch:
                payload[key] = batch[key]
        return {key: self._to_numpy(value) for key, value in payload.items()}

    def on_test_epoch_end(self) -> None:
        self.last_test_metrics = self._finish_metrics("test")
        diagnostics = prediction_diagnostics(
            {
                key: np.concatenate(values)
                for key, values in self._test_pred_arrays.items()
            },
        )
        self.save_test_predictions(
            metrics=self.last_test_metrics, diagnostic_metrics=diagnostics
        )
