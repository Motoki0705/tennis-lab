"""Lightning training module for multi-person track queries."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from torch import Tensor

from src.tasks.base.training.metric_logging import (
    compute_scalar_metric_statistics,
    evaluation_only_metric_logging_contract,
)
from src.tasks.base.training.tracking_lightning_module import (
    TrackingLightningModule,
    TrackingStepResult,
)
from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.model_io import (
    PLCSTrackingBoundModelIO,
    PLCSTrackQueryIOAdapter,
    build_plcs_model_io,
    plcs_reference_metadata_from_batch,
    resolve_plcs_track_query_reference_contract,
    validate_plcs_checkpoint_court_keypoints,
    validate_plcs_checkpoint_track_query_reference,
    write_plcs_checkpoint_court_keypoints,
    write_plcs_checkpoint_track_query_reference,
)
from src.tasks.plcs.training.tracking_losses import PLCSTrackingLoss
from src.tasks.plcs.training.tracking_metrics import plcs_tracking_statistics
from src.utils.schema.court_normalization import (
    add_court_coordinate_normalization,
    validate_court_coordinate_normalization,
)

PLCS_TRACKING_METRIC_CONTRACT = evaluation_only_metric_logging_contract(
    "PLCS tracking",
    headline_keys=(
        "position_error_m",
        "angular_error_deg",
        "presence_f1",
        "id_switches",
    ),
    progress_bar_keys=("position_error_m", "angular_error_deg"),
)


class PLCSTrackingLightningModule(TrackingLightningModule[dict[str, Any]]):
    """Train and evaluate clip-local player slots."""

    metric_logging_contract = PLCS_TRACKING_METRIC_CONTRACT

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
        self.track_query_reference_contract = (
            resolve_plcs_track_query_reference_contract(
                runtime.model,
                runtime.court_keypoint_contract,
            )
        )
        self.criterion = PLCSTrackingLoss(config.loss)
        if runtime.tracking_metrics is None:
            raise ValueError("PLCS tracking requires tracking_metrics configuration.")
        self.tracking_metric_config = runtime.tracking_metrics

    def compute_tracking_step(
        self,
        batch: dict[str, Tensor],
        *,
        compute_metrics: bool,
    ) -> TrackingStepResult[dict[str, Any]]:
        """Run PLCS model-I/O, matching, loss, and optional metrics."""
        reference_metadata = plcs_reference_metadata_from_batch(batch)
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
        statistics = (
            plcs_tracking_statistics(
                prediction,
                batch,
                assignments,
                config=self.tracking_metric_config,
                court_reference_provenance=prepared.court_reference_provenance,
                reference_view_index=(
                    reference_metadata.reference_view_index
                    if reference_metadata is not None
                    else None
                ),
            )
            if compute_metrics
            else None
        )
        metrics = (
            compute_scalar_metric_statistics(
                statistics,
                zero_denominator_value=0.0,
            )
            if statistics is not None
            else {}
        )
        prediction_result: dict[str, Any] = dict(prediction)
        if reference_metadata is not None:
            prediction_result["reference_metadata"] = reference_metadata
        return TrackingStepResult(
            losses=losses,
            metrics=metrics,
            prediction=prediction_result,
            statistics=statistics,
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        add_court_coordinate_normalization(
            checkpoint, artifact="PLCS tracking checkpoint"
        )
        write_plcs_checkpoint_court_keypoints(
            checkpoint,
            self.plcs_runtime.court_keypoint_contract,
        )
        write_plcs_checkpoint_track_query_reference(
            checkpoint,
            self.track_query_reference_contract,
        )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        validate_court_coordinate_normalization(
            checkpoint, artifact="PLCS tracking checkpoint"
        )
        validate_plcs_checkpoint_court_keypoints(
            checkpoint,
            self.plcs_runtime.court_keypoint_contract,
        )
        validate_plcs_checkpoint_track_query_reference(
            checkpoint,
            self.track_query_reference_contract,
        )

    def tracking_prediction_result(
        self, prediction: dict[str, Any]
    ) -> dict[str, Any]:
        """Return the canonical PLCS tensor mapping unchanged."""
        return prediction

    def test_prediction_payload(
        self, batch: Any, result: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        payload: dict[str, np.ndarray] = {
            "pred_position": self._to_numpy(result["position"]),
            "pred_rotation": self._to_numpy(result["rotation"]),
            "pred_presence_logits": self._to_numpy(result["presence_logits"]),
            "target_position": self._to_numpy(batch["target_position"]),
            "target_rotation": self._to_numpy(batch["target_rotation"]),
            "target_presence": self._to_numpy(batch["target_presence"]),
            "target_instance_id": self._to_numpy(batch["target_instance_id"]),
            "padding_mask": self._to_numpy(batch["padding_mask"]),
        }
        reference_metadata = plcs_reference_metadata_from_batch(batch)
        if reference_metadata is not None:
            num_views_range = cast(
                "list[int] | tuple[int, int]",
                self.plcs_runtime.data.values["num_views_range"],
            )
            payload.update(
                {
                    key: self._to_numpy(value)
                    for key, value in reference_metadata.prediction_payload(
                        max_views=int(num_views_range[1]),
                    ).items()
                }
            )
        return payload
