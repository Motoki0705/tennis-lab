"""Shared Lightning lifecycle; always evaluate the validation-selected checkpoint."""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor

from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.base.triangulation_residual.configuration import (
    FeatureConfig,
    ResidualConfig,
    validate_config,
)
from src.tasks.base.triangulation_residual.contracts import (
    feature_dimension,
    validate_model_inputs,
)
from src.tasks.base.triangulation_residual.data import ResidualDataModule
from src.tasks.base.triangulation_residual.diagnostics import prediction_diagnostics
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
        "schema_version": 2,
        "ffn_type": config.model.ffn_type,
        "features": asdict(config.features),
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


def migrate_legacy_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Validate embedded semantics and explicitly upgrade known schema-1 copies.

    Only historical v1/v2 SwiGLU/raw checkpoints may omit the new config fields.
    The supplied mapping and its tensors are never modified. Migration provenance
    is retained for inference metadata; ordinary training configs remain strict.
    """
    validate_court_coordinate_normalization(
        checkpoint, artifact="geometry residual checkpoint"
    )
    marker = checkpoint.get("geometric_residual_contract")
    if (
        not isinstance(marker, dict)
        or type(marker.get("schema_version")) is not int
        or marker["schema_version"] not in (1, 2)
    ):
        raise ValueError("Incompatible geometric residual checkpoint semantics")
    hyper_parameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyper_parameters, Mapping) or "config" not in hyper_parameters:
        raise ValueError("Geometric residual checkpoint requires its embedded config")
    embedded = hyper_parameters["config"]
    if isinstance(embedded, DictConfig):
        embedded = OmegaConf.to_container(embedded, resolve=True)
    config = OmegaConf.create(embedded)
    if not isinstance(config, DictConfig):
        raise ValueError("Geometric residual checkpoint config must be a mapping")
    legacy = marker["schema_version"] == 1
    if legacy:
        if not isinstance(config.get("model"), DictConfig):
            raise ValueError("Geometric residual checkpoint requires model config")
        if "ffn_type" not in config.model:
            config.model.ffn_type = "swiglu"
        if "features" not in config:
            config.features = asdict(FeatureConfig("raw", 1.0))
    parsed = validate_config(config)
    expected = checkpoint_contract(parsed)
    source_expected = dict(expected)
    if legacy:
        if parsed.model.ffn_type != "swiglu" or parsed.features != FeatureConfig(
            "raw", 1.0
        ):
            raise ValueError("Legacy residual checkpoints require SwiGLU/raw semantics")
        source_expected["schema_version"] = 1
        del source_expected["ffn_type"], source_expected["features"]
    if marker != source_expected:
        raise ValueError("Incompatible geometric residual checkpoint semantics")
    migrated = dict(checkpoint)
    if legacy:
        migrated["hyper_parameters"] = {
            **hyper_parameters,
            "config": OmegaConf.to_container(config, resolve=True),
        }
        migrated["geometric_residual_contract"] = expected
        migrated["geometric_residual_migration"] = {
            "source_contract": deepcopy(marker),
            "target_schema_version": 2,
            "explicit_settings": {
                "model.ffn_type": "swiglu",
                "features": asdict(parsed.features),
            },
        }
        warnings.warn(
            "Migrating legacy geometric residual checkpoint schema 1 to schema 2 "
            "with model.ffn_type=swiglu, features.residual_encoding=raw, "
            "features.residual_scale=1.0; source checkpoint is unchanged",
            UserWarning,
            stacklevel=2,
        )
    return migrated


class ResidualLightningModule(BaseLightningModule):
    def __init__(self, config: DictConfig, steps_per_epoch: int | None = None) -> None:
        self.residual_config = validate_config(config)
        super().__init__(config)
        self.steps_per_epoch = steps_per_epoch
        self.model = GeometricResidualModel(
            self.residual_config.task, self.residual_config.model
        )
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
        validated = migrate_legacy_checkpoint(checkpoint)
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
            self.residual_config.task,
            v2=self.residual_config.v2,
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
            self.residual_config.task,
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
            self.residual_config.task,
        )
        self.save_test_predictions(
            metrics=self.last_test_metrics, diagnostic_metrics=diagnostics
        )


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
