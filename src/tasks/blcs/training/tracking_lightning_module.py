"""Lightning module for the multi-ball track-query baseline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
from torch import Tensor

from src.tasks.base.configuration import as_config_mapping, require_config_mapping
from src.tasks.base.training.tracking_lightning_module import (
    TrackingLightningModule,
    TrackingStepResult,
)
from src.tasks.base.training.tracking_metrics import TrackingMetricConfig
from src.tasks.blcs.model_io import (
    BLCSTrackQueryPrediction,
    TrackQueryBoundModelIO,
    TrackQueryModelIOAdapter,
)
from src.tasks.blcs.training.tracking_losses import BLCSTrackingLoss
from src.tasks.blcs.training.tracking_metrics import blcs_tracking_metrics


class BLCSTrackingLightningModule(TrackingLightningModule[BLCSTrackQueryPrediction]):
    """Train and evaluate multi-ball clip-local slots."""

    def __init__(
        self,
        config: Any,
        *,
        model_io: TrackQueryBoundModelIO,
    ) -> None:
        super().__init__(config)
        self.model_io = model_io
        self.model = model_io.model
        self.io_adapter = cast("TrackQueryModelIOAdapter", model_io.adapter)
        self.criterion = BLCSTrackingLoss(config.loss)
        root = as_config_mapping(config, path="configuration")
        self.tracking_metrics = TrackingMetricConfig.from_mapping(
            require_config_mapping(root, "tracking_metrics", path="configuration")
        )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Reject the deleted checkpoint-key contract without migrating it."""
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise TypeError("Tracking checkpoint must contain a state_dict mapping.")
        legacy_prefix = "model.group_encoder."
        legacy_keys = [key for key in state_dict if key.startswith(legacy_prefix)]
        if legacy_keys:
            raise RuntimeError(
                "Checkpoint uses the deleted model.group_encoder key contract; "
                "retrain or explicitly convert the artifact outside runtime loading. "
                f"First incompatible key: {legacy_keys[0]}."
            )

    def compute_tracking_step(
        self,
        batch: dict[str, Tensor],
        *,
        compute_metrics: bool,
    ) -> TrackingStepResult[BLCSTrackQueryPrediction]:
        """Run BLCS model-I/O, matching, loss, and optional metrics."""
        prepared = self.io_adapter.build_training_batch(batch)
        prediction = self.model_io.decode_output(
            self.model_io.execute_call(prepared.call)
        )
        loss_inputs, assignments = self.criterion.prepare_inputs(prediction, prepared)
        losses = self.criterion(loss_inputs)
        metrics = (
            blcs_tracking_metrics(
                prediction,
                prepared,
                assignments,
                config=self.tracking_metrics,
            )
            if compute_metrics
            else {}
        )
        return TrackingStepResult(
            losses=losses,
            metrics=metrics,
            prediction=prediction,
        )

    def tracking_prediction_result(
        self, prediction: BLCSTrackQueryPrediction
    ) -> dict[str, Any]:
        """Keep the BLCS typed prediction under its canonical result key."""
        return {"prediction": prediction}

    def test_prediction_payload(
        self, batch: Any, result: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        prediction = cast("BLCSTrackQueryPrediction", result["prediction"])
        prepared = self.io_adapter.build_training_batch(batch)
        return {
            "pred_position": self._to_numpy(prediction.position),
            "pred_presence_logits": self._to_numpy(prediction.presence_logits),
            "target_position": self._to_numpy(prepared.target_position),
            "target_presence": self._to_numpy(prepared.target_presence),
            "target_instance_id": self._to_numpy(prepared.target_instance_id),
            "frame_mask": self._to_numpy(prepared.frame_mask),
        }
