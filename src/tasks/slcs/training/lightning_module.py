"""Lightning module for SLCS training."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
from omegaconf import DictConfig
from torch import Tensor

from src.tasks.base.model_io import BoundModelIO
from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.slcs.configuration import SLCSTrainingRuntimeConfig
from src.tasks.slcs.model_io import (
    SLCSDecodedOutput,
    SLCSModelIOAdapter,
    SLCSRawOutput,
    SLCSTrainingTargets,
)
from src.tasks.slcs.model_io.factory import create_slcs_model_io
from src.tasks.slcs.models.slcs_model import SLCSFusionModel
from src.tasks.slcs.training.losses import SLCSLoss, build_slcs_loss_inputs
from src.tasks.slcs.training.metrics import SLCSMetrics


class SLCSLightningModule(BaseLightningModule):
    """Supervised training of :class:`SLCSFusionModel` on pseudo-labels."""

    def __init__(self, config: SLCSTrainingRuntimeConfig | DictConfig) -> None:
        runtime = (
            SLCSTrainingRuntimeConfig.from_config(config)
            if isinstance(config, DictConfig)
            else config
        )
        super().__init__(runtime.raw)
        self.max_epochs = runtime.training.trainer.max_epochs
        self.model: SLCSFusionModel
        self.model_adapter: SLCSModelIOAdapter
        self.model_io: BoundModelIO[
            Mapping[str, object], SLCSRawOutput, SLCSDecodedOutput
        ]
        self.model, self.model_adapter, self.model_io = create_slcs_model_io(
            runtime.model, runtime.data
        )
        self.loss_fn = SLCSLoss(runtime.loss)
        self._metrics = {
            "train": SLCSMetrics(),
            "val": SLCSMetrics(),
            "test": SLCSMetrics(),
        }

    # ------------------------------------------------------------------

    def forward_batch(self, batch: dict[str, Tensor]) -> SLCSDecodedOutput:
        """Validate, execute, and decode one collated SLCS batch."""
        return self.model_io.run(batch)

    def _step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        call = self.model_io.build_call(batch)
        targets = self.model_adapter.build_training_targets(batch)
        outputs = self.model_io.decode_output(self.model_io.execute_call(call))
        losses = self.loss_fn(build_slcs_loss_inputs(outputs, targets))
        batch_metrics = self._metrics[stage].update(outputs, targets)

        batch_size = int(batch["frame_mask"].shape[0])
        on_step = stage == "train"
        self.log(
            f"{stage}/loss",
            losses["total"],
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        for name, value in losses.items():
            if name == "total":
                continue
            self.log(
                f"{stage}/loss_{name}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        for name, value in batch_metrics.items():
            self.log(
                f"{stage}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=name.endswith("position_error_m"),
                batch_size=batch_size,
            )
        if stage == "test":
            self.collect_test_predictions(
                batch, {"outputs": outputs, "targets": targets}
            )
        return cast(Tensor, losses["total"])

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "test")

    # ------------------------------------------------------------------
    # Epoch-end aggregation
    # ------------------------------------------------------------------

    def _log_epoch_metrics(self, stage: str) -> None:
        metrics = self._metrics[stage].compute()
        for name, value in metrics.items():
            self.log(f"{stage}/{name}_epoch", value, on_epoch=True)
        self._metrics[stage].reset()

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics("val")

    def on_test_epoch_end(self) -> None:
        metrics = self._metrics["test"].compute()
        for name, value in metrics.items():
            self.log(f"test/{name}_epoch", value, on_epoch=True)
        self._metrics["test"].reset()
        self.save_test_predictions(metrics)

    # ------------------------------------------------------------------
    # Test prediction persistence (issue #533 repro bundles)
    # ------------------------------------------------------------------

    def test_prediction_payload(
        self, batch: Any, result: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        outputs = cast(SLCSDecodedOutput, result["outputs"])
        targets = cast(SLCSTrainingTargets, result["targets"])
        payload_keys = {
            "pred_player_position": outputs.player_position,
            "pred_player_rotation": outputs.player_rotation,
            "pred_player_position_log_b": outputs.player_position_log_b,
            "pred_player_rotation_log_b": outputs.player_rotation_log_b,
            "pred_ball_position": outputs.ball_position,
            "pred_ball_position_log_b": outputs.ball_position_log_b,
            "target_player_position": targets.target_player_position,
            "target_player_rotation": targets.target_player_rotation,
            "target_player_valid": targets.player_mask,
            "target_ball_position": targets.target_ball_position,
            "target_ball_valid": targets.ball_mask,
            "frame_mask": targets.frame_mask,
        }
        return {k: self._to_numpy(v) for k, v in payload_keys.items()}


__all__ = ["SLCSLightningModule"]
