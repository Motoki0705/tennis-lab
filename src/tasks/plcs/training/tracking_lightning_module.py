"""Lightning training module for multi-person track queries."""

from __future__ import annotations

from typing import Any, Self, cast

import numpy as np
from torch import Tensor, nn

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
from src.utils.geometry.court_pose import world_pose_to_canonical_pose
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


def _require_independent_presence_head(model: nn.Module) -> nn.Module:
    """Return a direct presence head or fail before fine-tuning can start."""
    presence_head = getattr(model, "presence_head", None)
    registered_head = dict(model.named_children()).get("presence_head")
    if not isinstance(presence_head, nn.Module) or registered_head is not presence_head:
        raise ValueError(
            "training.fine_tune_mode='presence_head' requires the configured "
            "model to expose an independent registered nn.Module named "
            f"'presence_head'; got {type(model).__name__}."
        )
    return presence_head


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
        self.fine_tune_mode = runtime.fine_tune_mode
        if self.fine_tune_mode == "presence_head":
            self._configure_presence_head_fine_tuning()

    def _configure_presence_head_fine_tuning(self) -> None:
        """Freeze all state except the independently registered presence head."""
        presence_head = _require_independent_presence_head(self.model)
        self.requires_grad_(False)
        presence_head.requires_grad_(True)
        self.model.eval()
        presence_head.train()

    def train(self, mode: bool = True) -> Self:
        """Keep the frozen trunk deterministic while training the presence head."""
        super().train(mode)
        if self.fine_tune_mode == "presence_head":
            presence_head = _require_independent_presence_head(self.model)
            self.model.eval()
            presence_head.train(mode)
        return self

    def optimizer_param_groups(self) -> list[dict[str, Any]] | None:
        """Optimize only explicitly trainable parameters during head fine-tuning."""
        if self.fine_tune_mode == "all":
            return None
        trainable_parameters = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError(
                "presence-head fine-tuning resolved no trainable parameters."
            )
        return [{"params": trainable_parameters}]

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
        if decoded.canonical_pose is not None:
            prediction["canonical_pose"] = decoded.canonical_pose
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
        canonical_pose = result.get("canonical_pose")
        if canonical_pose is not None:
            if "target_human_kp_3d" not in batch:
                raise ValueError(
                    "Canonical tracking test predictions require "
                    "batch['target_human_kp_3d']."
                )
            target_canonical_pose = world_pose_to_canonical_pose(
                batch["target_human_kp_3d"],
                batch["target_position"],
                batch["target_rotation"],
            )
            payload["pred_canonical_pose"] = self._to_numpy(canonical_pose)
            payload["target_canonical_pose"] = self._to_numpy(
                target_canonical_pose
            )
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
