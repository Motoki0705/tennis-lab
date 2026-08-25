"""Lightning module for the multi-ball track-query baseline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
from torch import Tensor

from src.tasks.base.configuration import as_config_mapping, require_config_mapping
from src.tasks.base.model_io import (
    validate_model_artifact_court_keypoint_contract,
    write_model_artifact_court_keypoint_contract,
)
from src.tasks.base.training.tracking_lightning_module import (
    TrackingLightningModule,
    TrackingStepResult,
)
from src.tasks.base.training.tracking_metrics import TrackingMetricConfig
from src.tasks.blcs.configuration import parse_court_keypoint_contract
from src.tasks.blcs.model_io import (
    BLCSTrackQueryPrediction,
    TrackQueryBoundModelIO,
    TrackQueryModelIOAdapter,
    blcs_reference_metadata_from_batch,
)
from src.tasks.blcs.model_io.checkpoints import (
    resolve_blcs_track_query_reference_contract,
    validate_blcs_checkpoint_track_query_reference,
    write_blcs_checkpoint_track_query_reference,
)
from src.tasks.blcs.training.tracking_losses import BLCSTrackingLoss
from src.tasks.blcs.training.tracking_metrics import blcs_tracking_metrics
from src.utils.schema.court_normalization import (
    add_court_coordinate_normalization,
    validate_court_coordinate_normalization,
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
        self.court_keypoint_contract = parse_court_keypoint_contract(config)
        self.track_query_reference_contract = (
            resolve_blcs_track_query_reference_contract(config)
        )
        self.criterion = BLCSTrackingLoss(config.loss)
        root = as_config_mapping(config, path="configuration")
        self.tracking_metrics = TrackingMetricConfig.from_mapping(
            require_config_mapping(root, "tracking_metrics", path="configuration")
        )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Reject the deleted checkpoint-key contract without migrating it."""
        validate_court_coordinate_normalization(
            checkpoint, artifact="BLCS tracking checkpoint"
        )
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
        validate_model_artifact_court_keypoint_contract(
            checkpoint,
            self.court_keypoint_contract,
            location="BLCS tracking checkpoint",
        )
        track_query_reference = getattr(
            self,
            "track_query_reference_contract",
            None,
        )
        if track_query_reference is not None:
            validate_blcs_checkpoint_track_query_reference(
                checkpoint,
                track_query_reference,
            )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        add_court_coordinate_normalization(
            checkpoint, artifact="BLCS tracking checkpoint"
        )
        write_model_artifact_court_keypoint_contract(
            checkpoint,
            self.court_keypoint_contract,
            location="BLCS tracking checkpoint",
        )
        track_query_reference = getattr(
            self,
            "track_query_reference_contract",
            None,
        )
        if track_query_reference is not None:
            write_blcs_checkpoint_track_query_reference(
                checkpoint,
                track_query_reference,
            )

    def compute_tracking_step(
        self,
        batch: dict[str, Tensor],
        *,
        compute_metrics: bool,
    ) -> TrackingStepResult[BLCSTrackQueryPrediction]:
        """Run BLCS model-I/O, matching, loss, and optional metrics."""
        reference_metadata = blcs_reference_metadata_from_batch(batch)
        prepared = self.io_adapter.build_training_batch(batch)
        prepared = replace(prepared, reference_metadata=reference_metadata)
        prediction = self.model_io.decode_output(
            self.model_io.execute_call(prepared.call)
        )
        prediction = replace(prediction, reference_metadata=reference_metadata)
        loss_inputs, assignments = self.criterion.prepare_inputs(prediction, prepared)
        losses = self.criterion(loss_inputs)
        metrics = (
            blcs_tracking_metrics(
                prediction,
                prepared,
                assignments,
                config=self.tracking_metrics,
                court_keypoint_contract=self.court_keypoint_contract,
                reference_view_index=(
                    reference_metadata.reference_view_index
                    if reference_metadata is not None
                    else None
                ),
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

    def set_counterfactual_prediction_dir(self, output_dir: Path) -> None:
        """Route one explicit checkpoint-only pass to its isolated raw directory."""
        if not isinstance(output_dir, Path) or not output_dir.is_absolute():
            raise ValueError(
                "BLCS counterfactual prediction output must be an absolute Path."
            )
        self._counterfactual_prediction_dir = output_dir

    def _test_predictions_dir(self) -> Path:
        output_dir = getattr(self, "_counterfactual_prediction_dir", None)
        if output_dir is None:
            return cast("Path", super()._test_predictions_dir())
        if not isinstance(output_dir, Path):
            raise TypeError("BLCS counterfactual prediction output is invalid.")
        return output_dir

    def test_prediction_payload(
        self, batch: Any, result: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        prediction = cast("BLCSTrackQueryPrediction", result["prediction"])
        prepared = self.io_adapter.build_training_batch(batch)
        payload: dict[str, np.ndarray] = {
            "pred_position": self._to_numpy(prediction.position),
            "pred_presence_logits": self._to_numpy(prediction.presence_logits),
            "target_position": self._to_numpy(prepared.target_position),
            "target_presence": self._to_numpy(prepared.target_presence),
            "target_instance_id": self._to_numpy(prepared.target_instance_id),
            "frame_valid": self._to_numpy(prepared.frame_valid),
        }
        if getattr(self, "_counterfactual_prediction_dir", None) is not None:
            payload.update(
                {
                    "ball_uv": self._to_numpy(batch["ball_uv"]),
                    "ball_vis": self._to_numpy(batch["ball_vis"]),
                    "court_kp": self._to_numpy(batch["court_kp"]),
                    "court_vis": self._to_numpy(batch["court_vis"]),
                    "padding_mask": self._to_numpy(batch["padding_mask"]),
                    "target_velocity": self._to_numpy(prepared.target_velocity),
                    "target_slot_mask": self._to_numpy(prepared.target_slot_mask),
                    "clean_ball_uv": self._to_numpy(batch["clean_ball_uv"]),
                    "clean_ball_vis": self._to_numpy(batch["clean_ball_vis"]),
                    "candidate_gt_index": self._to_numpy(batch["candidate_gt_index"]),
                }
            )
        if prediction.reference_metadata is not None:
            metadata_payload = prediction.reference_metadata.prediction_payload(
                max_views=int(self.config.data.num_views_range[1]),
            )
            payload.update(
                {key: self._to_numpy(value) for key, value in metadata_payload.items()}
            )
        return payload
