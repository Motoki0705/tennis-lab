"""Lightning module for the multi-ball track-query baseline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from src.tasks.base.configuration import as_config_mapping, require_config_mapping
from src.tasks.base.data.reference_orientation import (
    reflect_court_vectors,
    select_counterfactual_reference_views,
)
from src.tasks.base.training.tracking_benchmark import (
    TRACKING_FUSION_BENCHMARK_BATCH_SIZE,
    TRACKING_FUSION_BENCHMARK_CLASSES,
    TRACKING_FUSION_BENCHMARK_DETECTIONS,
    TRACKING_FUSION_BENCHMARK_FRAMES,
    TRACKING_FUSION_BENCHMARK_PEAKS,
    TRACKING_FUSION_BENCHMARK_VIEWS,
    TrackingFusionBenchmarkResult,
    benchmark_tracking_fusion_cuda,
)
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
from src.tasks.blcs.models.components.observation_fusion import (
    KP7TrackObservationFusion,
)
from src.tasks.blcs.training.tracking_losses import BLCSTrackingLoss
from src.tasks.blcs.training.tracking_metrics import blcs_tracking_metrics


class BLCSTrackingLightningModule(
    TrackingLightningModule[BLCSTrackQueryPrediction]
):
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
        counterfactual_prediction: BLCSTrackQueryPrediction | None = None
        counterfactual_reference_view_index: Tensor | None = None
        counterfactual_orientation_sign: Tensor | None = None
        metrics: dict[str, Tensor] = {}
        if compute_metrics:
            counterfactual_reference_view_index, counterfactual_orientation_sign = (
                select_counterfactual_reference_views(
                    batch["camera_center"],
                    batch["view_mask"],
                    batch["reference_view_index"],
                    batch["orientation_sign"],
                )
            )
            counterfactual_batch = dict(batch)
            counterfactual_batch["reference_view_index"] = (
                counterfactual_reference_view_index
            )
            counterfactual_batch["orientation_sign"] = counterfactual_orientation_sign
            counterfactual_batch["target_position"] = reflect_court_vectors(
                batch["source_target_position"], counterfactual_orientation_sign
            )
            counterfactual_batch["target_velocity"] = reflect_court_vectors(
                batch["source_target_velocity"], counterfactual_orientation_sign
            )
            with torch.no_grad():
                counterfactual_prepared = self.io_adapter.build_training_batch(
                    counterfactual_batch
                )
                counterfactual_prediction = self.model_io.decode_output(
                    self.model_io.execute_call(counterfactual_prepared.call)
                )
                _, counterfactual_assignments = self.criterion.prepare_inputs(
                    counterfactual_prediction,
                    counterfactual_prepared,
                )
            metrics = blcs_tracking_metrics(
                prediction,
                prepared,
                assignments,
                counterfactual_prediction=counterfactual_prediction,
                counterfactual_assignments=counterfactual_assignments,
                counterfactual_orientation_sign=counterfactual_orientation_sign,
                config=self.tracking_metrics,
            )
        return TrackingStepResult(
            losses=losses,
            metrics=metrics,
            prediction=prediction,
            counterfactual_prediction=counterfactual_prediction,
            counterfactual_reference_view_index=counterfactual_reference_view_index,
            counterfactual_orientation_sign=counterfactual_orientation_sign,
        )

    def tracking_prediction_result(
        self, result: TrackingStepResult[BLCSTrackQueryPrediction]
    ) -> dict[str, Any]:
        """Keep the BLCS typed prediction under its canonical result key."""
        output: dict[str, Any] = {"prediction": result.prediction}
        if result.counterfactual_prediction is not None:
            output.update(
                {
                    "counterfactual_prediction": result.counterfactual_prediction,
                    "counterfactual_reference_view_index": (
                        result.counterfactual_reference_view_index
                    ),
                    "counterfactual_orientation_sign": (
                        result.counterfactual_orientation_sign
                    ),
                }
            )
        return output

    def benchmark_court_peak_fusion(
        self,
    ) -> TrackingFusionBenchmarkResult | None:
        """Benchmark the active BLCS KP7 reference observation fusion on CUDA."""
        if self.model.court_observation_profile != "kp7_reference":
            return None
        device = next(self.model.parameters()).device
        if device.type != "cuda":
            raise RuntimeError(
                "KP7 reference test evidence requires a CUDA fusion benchmark."
            )
        shape = (
            TRACKING_FUSION_BENCHMARK_BATCH_SIZE,
            TRACKING_FUSION_BENCHMARK_VIEWS,
            TRACKING_FUSION_BENCHMARK_FRAMES,
        )
        object_shape = (*shape, TRACKING_FUSION_BENCHMARK_DETECTIONS)
        peak_shape = (
            *shape,
            TRACKING_FUSION_BENCHMARK_CLASSES,
            TRACKING_FUSION_BENCHMARK_PEAKS,
        )
        dtype = next(self.model.parameters()).dtype
        inputs = {
            "court_peak_uv": torch.rand(*peak_shape, 2, device=device, dtype=dtype),
            "court_peak_score": torch.rand(*peak_shape, device=device, dtype=dtype),
            "court_peak_covariance": torch.eye(2, device=device, dtype=dtype)
            .view(1, 1, 1, 1, 1, 2, 2)
            .expand(*peak_shape, 2, 2)
            * 1.0e-4,
            "court_peak_valid": torch.ones(
                *peak_shape, device=device, dtype=torch.bool
            ),
            "ball_uv": torch.rand(*object_shape, 2, device=device, dtype=dtype),
            "ball_score": torch.rand(*object_shape, device=device, dtype=dtype),
            "ball_visible": torch.ones(
                *object_shape, device=device, dtype=torch.bool
            ),
            "state_valid": torch.ones(
                shape[0],
                shape[2],
                shape[1],
                object_shape[-1],
                device=device,
                dtype=torch.bool,
            ),
        }
        fusion = self.model.observation_encoder
        if not isinstance(fusion, KP7TrackObservationFusion):
            raise RuntimeError("KP7 reference profile has no KP7 observation fusion.")
        was_training = fusion.training
        fusion.eval()
        try:
            return benchmark_tracking_fusion_cuda(
                lambda: fusion(
                    ball_uv=inputs["ball_uv"],
                    ball_visible=inputs["ball_visible"],
                    state_valid=inputs["state_valid"],
                    ball_score=inputs["ball_score"],
                    court_kp=None,
                    court_visible=None,
                    point_attention_mask=None,
                    court_peak_uv=inputs["court_peak_uv"],
                    court_peak_score=inputs["court_peak_score"],
                    court_peak_covariance=inputs["court_peak_covariance"],
                    court_peak_valid=inputs["court_peak_valid"],
                ),
                inputs=inputs,
            )
        finally:
            fusion.train(was_training)

    def test_prediction_payload(
        self, batch: Any, result: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        prediction = cast("BLCSTrackQueryPrediction", result["prediction"])
        counterfactual = cast(
            "BLCSTrackQueryPrediction", result["counterfactual_prediction"]
        )
        prepared = self.io_adapter.build_training_batch(batch)
        return {
            "pred_position": self._to_numpy(prediction.position),
            "pred_presence_logits": self._to_numpy(prediction.presence_logits),
            "counterfactual_pred_position": self._to_numpy(
                counterfactual.position
            ),
            "counterfactual_pred_presence_logits": self._to_numpy(
                counterfactual.presence_logits
            ),
            "target_position": self._to_numpy(prepared.target_position),
            "source_target_position": self._to_numpy(
                prepared.source_target_position
            ),
            "target_presence": self._to_numpy(prepared.target_presence),
            "target_instance_id": self._to_numpy(prepared.target_instance_id),
            "frame_mask": self._to_numpy(prepared.frame_mask),
            "reference_view_index": self._to_numpy(
                prepared.reference_view_index
            ),
            "orientation_sign": self._to_numpy(prepared.orientation_sign),
            "counterfactual_reference_view_index": self._to_numpy(
                result["counterfactual_reference_view_index"]
            ),
            "counterfactual_orientation_sign": self._to_numpy(
                result["counterfactual_orientation_sign"]
            ),
        }
