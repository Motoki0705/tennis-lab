"""Lightning training module for multi-person track queries."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from torch import Tensor

from src.tasks.base.training.tracking_lightning_module import (
    TrackingLightningModule,
    TrackingStepResult,
)
from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.model_io import (
    PLCSTrackingBoundModelIO,
    PLCSTrackQueryIOAdapter,
    build_plcs_model_io,
    validate_plcs_checkpoint_normalization,
    write_plcs_checkpoint_normalization,
)
from src.tasks.plcs.training.tracking_losses import PLCSTrackingLoss
from src.tasks.plcs.training.tracking_metrics import plcs_tracking_metrics


class PLCSTrackingLightningModule(TrackingLightningModule[dict[str, Tensor]]):
    """Train and evaluate clip-local player slots."""

    def __init__(self, config: Any) -> None:
        runtime = PLCSTrainingConfig.from_config(config)
        super().__init__(config)
        model_io = build_plcs_model_io(runtime)
        adapter = model_io.adapter
        if not isinstance(adapter, PLCSTrackQueryIOAdapter):
            raise ValueError(
                "PLCSTrackingLightningModule requires a track-query model-I/O pair."
            )
        self.io_adapter = adapter
        self.model_io = cast(PLCSTrackingBoundModelIO, model_io)
        self.model = self.model_io.model
        self.plcs_runtime = runtime
        self.criterion = PLCSTrackingLoss(
            config.loss,
            normalization=runtime.court_coordinate_normalization.contract,
        )
        if runtime.tracking_metrics is None:
            raise ValueError("PLCS tracking requires tracking_metrics configuration.")
        self.tracking_metric_config = runtime.tracking_metrics

    def compute_tracking_step(
        self,
        batch: dict[str, Tensor],
        *,
        compute_metrics: bool,
    ) -> TrackingStepResult[dict[str, Tensor]]:
        """Run PLCS model-I/O, matching, loss, and optional metrics."""
        prepared = self.io_adapter.prepare_training_batch(batch)
        raw_prediction = self.model_io.execute_call(prepared.call)
        decoded = self.model_io.decode_output(raw_prediction)
        prediction = {
            "position": decoded.position,
            "rotation": decoded.rotation,
            "presence_logits": decoded.presence_logits,
        }
        loss_inputs, assignments = self.criterion.prepare_inputs(prediction, batch)
        losses = self.criterion(loss_inputs)
        metrics = (
            plcs_tracking_metrics(
                prediction,
                batch,
                assignments,
                config=self.tracking_metric_config,
                normalization=self.plcs_runtime.court_coordinate_normalization.contract,
            )
            if compute_metrics
            else {}
        )
        return TrackingStepResult(
            losses=losses,
            metrics=metrics,
            prediction=prediction,
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Persist the selected normalization beside tracking model state."""
        write_plcs_checkpoint_normalization(
            checkpoint,
            self.plcs_runtime.court_coordinate_normalization.contract,
        )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Reject normalization mismatch before tracking state restoration."""
        validate_plcs_checkpoint_normalization(
            checkpoint,
            self.plcs_runtime.court_coordinate_normalization.contract,
        )

    def tracking_prediction_result(
        self, prediction: dict[str, Tensor]
    ) -> dict[str, Any]:
        """Return the canonical PLCS tensor mapping unchanged."""
        return prediction

    def test_prediction_payload(
        self, batch: Any, result: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        return {
            "pred_position": self._to_numpy(result["position"]),
            "pred_rotation": self._to_numpy(result["rotation"]),
            "pred_presence_logits": self._to_numpy(result["presence_logits"]),
            "target_position": self._to_numpy(batch["target_position"]),
            "target_rotation": self._to_numpy(batch["target_rotation"]),
            "target_presence": self._to_numpy(batch["target_presence"]),
            "target_instance_id": self._to_numpy(batch["target_instance_id"]),
            "padding_mask": self._to_numpy(batch["padding_mask"]),
        }
