"""Lightning module for the multi-ball track-query baseline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
from torch import Tensor

from src.tasks.base.configuration import as_config_mapping, require_config_mapping
from src.tasks.base.model_io import (
    validate_checkpoint_court_coordinate_contract,
    write_checkpoint_court_coordinate_contract,
)
from src.tasks.base.training.tracking_lightning_module import (
    TrackingLightningModule,
    TrackingStepResult,
)
from src.tasks.base.training.tracking_metrics import TrackingMetricConfig
from src.tasks.blcs.configuration import parse_court_coordinate_normalization
from src.tasks.blcs.model_io import (
    BLCSTrackQueryPrediction,
    TrackQueryBoundModelIO,
    TrackQueryModelIOAdapter,
)
from src.tasks.blcs.training.tracking_losses import BLCSTrackingLoss
from src.tasks.blcs.training.tracking_metrics import blcs_tracking_metrics
from src.utils.schema.court_normalization import (
    resolve_court_coordinate_normalization,
)


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
        self.court_coordinate_normalization = (
            parse_court_coordinate_normalization(config)
        )
        self.criterion = BLCSTrackingLoss(
            config.loss,
            normalization=self.court_coordinate_normalization,
            gravity=float(config.physics.gravity),
            frame_dt=1.0 / float(config.rally.output_fps),
        )
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
        # Hooks exercised without a fully initialized runtime retain the legacy
        # v1 checkpoint contract. Fully constructed v2 runtimes always carry
        # their selected contract and therefore still reject missing/mismatched
        # normalization metadata.
        normalization = getattr(
            self,
            "court_coordinate_normalization",
            resolve_court_coordinate_normalization("v1"),
        )
        validate_checkpoint_court_coordinate_contract(
            checkpoint,
            normalization,
            location="BLCS tracking checkpoint",
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Persist the exact tracking normalization contract."""
        write_checkpoint_court_coordinate_contract(
            checkpoint,
            self.court_coordinate_normalization,
            location="BLCS tracking checkpoint",
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
                normalization=self.court_coordinate_normalization,
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
            "frame_valid": self._to_numpy(prepared.frame_valid),
        }
