"""Lightning training module for multi-person track queries."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from src.tasks.base.data.reference_orientation import (
    reflect_court_vectors,
    reflect_heading,
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
from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.model_io import (
    PLCSTrackingBoundModelIO,
    PLCSTrackQueryIOAdapter,
    build_plcs_model_io,
)
from src.tasks.plcs.models.components.observation_fusion import (
    KP7PlayerObservationFusion,
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
        self.criterion = PLCSTrackingLoss(config.loss)
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
        counterfactual_prediction: dict[str, Tensor] | None = None
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
            counterfactual_batch["target_rotation"] = reflect_heading(
                batch["source_target_rotation"], counterfactual_orientation_sign
            )
            counterfactual_batch["target_human_kp_3d"] = reflect_court_vectors(
                batch["source_target_human_kp_3d"],
                counterfactual_orientation_sign,
            )
            with torch.no_grad():
                counterfactual_prepared = self.io_adapter.prepare_training_batch(
                    counterfactual_batch
                )
                raw_counterfactual = self.model_io.execute_call(
                    counterfactual_prepared.call
                )
                decoded_counterfactual = self.model_io.decode_output(
                    raw_counterfactual
                )
                counterfactual_prediction = {
                    "position": decoded_counterfactual.position,
                    "rotation": decoded_counterfactual.rotation,
                    "presence_logits": decoded_counterfactual.presence_logits,
                }
                _, counterfactual_assignments = self.criterion.prepare_inputs(
                    counterfactual_prediction,
                    counterfactual_batch,
                )
            metrics = plcs_tracking_metrics(
                prediction,
                batch,
                assignments,
                counterfactual_prediction=counterfactual_prediction,
                counterfactual_assignments=counterfactual_assignments,
                counterfactual_orientation_sign=counterfactual_orientation_sign,
                config=self.tracking_metric_config,
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
        self, result: TrackingStepResult[dict[str, Tensor]]
    ) -> dict[str, Any]:
        """Return the canonical PLCS tensor mapping unchanged."""
        output: dict[str, Any] = dict(result.prediction)
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
        """Benchmark the active PLCS KP7 reference observation fusion on CUDA."""
        if self.model.court_observation_profile != "kp7_reference":
            return None
        device = next(self.model.parameters()).device
        if device.type != "cuda":
            raise RuntimeError(
                "KP7 reference test evidence requires a CUDA fusion benchmark."
            )
        fusion = self.model.kp7_observation_encoder
        if not isinstance(fusion, KP7PlayerObservationFusion):
            raise RuntimeError("KP7 reference profile has no observation fusion.")
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
            "player_anchor": torch.rand(
                *object_shape, 2, device=device, dtype=dtype
            ),
            "player_features": torch.rand(
                *object_shape,
                fusion.set_fusion.object_feature_dim,
                device=device,
                dtype=dtype,
            ),
            "detection_mask": torch.ones(
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
        was_training = fusion.training
        fusion.eval()
        try:
            return benchmark_tracking_fusion_cuda(
                lambda: fusion(
                    human_kp=inputs["player_anchor"],
                    detection_mask=inputs["detection_mask"],
                    camera_state_valid=inputs["state_valid"],
                    court_kp=None,
                    court_vis=None,
                    court_peak_uv=inputs["court_peak_uv"],
                    court_peak_score=inputs["court_peak_score"],
                    court_peak_covariance=inputs["court_peak_covariance"],
                    court_peak_valid=inputs["court_peak_valid"],
                    player_anchor=inputs["player_anchor"],
                    player_features=inputs["player_features"],
                ),
                inputs=inputs,
            )
        finally:
            fusion.train(was_training)

    def test_prediction_payload(
        self, batch: Any, result: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        counterfactual = cast(
            "dict[str, Tensor]", result["counterfactual_prediction"]
        )
        return {
            "pred_position": self._to_numpy(result["position"]),
            "pred_rotation": self._to_numpy(result["rotation"]),
            "pred_presence_logits": self._to_numpy(result["presence_logits"]),
            "counterfactual_pred_position": self._to_numpy(
                counterfactual["position"]
            ),
            "counterfactual_pred_rotation": self._to_numpy(
                counterfactual["rotation"]
            ),
            "counterfactual_pred_presence_logits": self._to_numpy(
                counterfactual["presence_logits"]
            ),
            "target_position": self._to_numpy(batch["target_position"]),
            "source_target_position": self._to_numpy(
                batch["source_target_position"]
            ),
            "target_rotation": self._to_numpy(batch["target_rotation"]),
            "source_target_rotation": self._to_numpy(
                batch["source_target_rotation"]
            ),
            "target_presence": self._to_numpy(batch["target_presence"]),
            "target_instance_id": self._to_numpy(batch["target_instance_id"]),
            "frame_mask": self._to_numpy(batch["frame_mask"]),
            "reference_view_index": self._to_numpy(
                batch["reference_view_index"]
            ),
            "orientation_sign": self._to_numpy(batch["orientation_sign"]),
            "counterfactual_reference_view_index": self._to_numpy(
                result["counterfactual_reference_view_index"]
            ),
            "counterfactual_orientation_sign": self._to_numpy(
                result["counterfactual_orientation_sign"]
            ),
        }
