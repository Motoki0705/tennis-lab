"""Lightning module for mixed dense and synthetic-only pose supervision."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from torch import Tensor

from src.tasks.base.training.repro import resolve_queue_repro_dir
from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.tasks.court_detection.data.contracts import CourtTargetBundleSpec
from src.tasks.court_detection.model_io.adapters import CourtPoseModelIOAdapter
from src.tasks.court_detection.model_io.contracts import (
    CourtModelOutput,
    CourtPoseTargetBatch,
    CourtPoseTrainingResult,
    CourtTrainingResult,
)
from src.tasks.court_detection.model_io.mixed_adapter import (
    MixedCourtPoseModelIOAdapter,
)
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
)
from src.utils.configuration import PathRole


class MixedCourtDetectionLightningModule(CourtDetectionLightningModule):
    """Use the standard model while restricting pose metrics/losses by mask."""

    def __init__(
        self,
        config: object,
        *,
        target_bundle: CourtTargetBundleSpec | None = None,
        target_bundle_state: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(
            config,
            target_bundle=target_bundle,
            target_bundle_state=target_bundle_state,
        )
        runtime = CourtTrainingConfig.from_config(config).shared
        self._test_prediction_output_key = runtime.run.output_dir.relative_to(
            runtime.resolver.roots.output_root
        )
        if not self.pose_variant:
            return
        raw_model_io = cast(object, self.model_io)
        if not isinstance(raw_model_io, CourtPoseModelIOAdapter):
            raise TypeError(
                "Pose-enabled mixed Court training requires a pose adapter."
            )
        mixed = MixedCourtPoseModelIOAdapter(
            raw_model_io.spec,
            loss_config=raw_model_io.pose_loss_config,
            execution_boundary=raw_model_io.execution_boundary,
        )
        mixed.validate_model_pair(self.model)
        cast(Any, self).model_io = mixed
        self.consistency_instrumented = mixed.consistency_instrumented

    def _test_predictions_dir(self) -> Path:
        queue_repro_dir = resolve_queue_repro_dir()
        if queue_repro_dir is not None:
            queue_predictions: Path = queue_repro_dir / "predictions"
            return queue_predictions
        artifact_predictions: Path = self.path_resolver.resolve(
            PathRole.ARTIFACT,
            "test_predictions",
            self._test_prediction_output_key,
        )
        return artifact_predictions

    def _shared_step(
        self,
        batch: Mapping[str, object],
        stage: str,
    ) -> CourtTrainingResult | CourtPoseTrainingResult:
        raw_model_io = cast(object, self.model_io)
        if not isinstance(raw_model_io, MixedCourtPoseModelIOAdapter):
            return super()._shared_step(batch, stage)
        model_io = raw_model_io

        pose_call = model_io.prepare_training_batch(batch)
        output = cast(
            CourtModelOutput,
            self.model(*pose_call.model_call.model_args),
        )
        progress_fraction = (
            self._progress_fraction(stage)
            if model_io.consistency_instrumented
            else None
        )
        result = model_io.training_result(
            output,
            pose_call,
            progress_fraction=progress_fraction,
        )
        self._log_training_result(stage, result)
        if stage == "train":
            self._record_matrix_loss_result(result)

        image_size = batch.get("image_size")
        if not isinstance(image_size, Tensor):
            raise ValueError("Court batch image_size must be a Tensor.")
        for kind in self.target_bundle.kinds:
            self._stage_metrics[stage][kind].update(
                result.output.dense_logits[kind],
                pose_call.targets[kind],
                image_size=image_size,
            )

        mask = model_io.pose_supervision_mask(pose_call)
        if not bool(mask.any()):
            return result
        target_pose = pose_call.targets.get("pose")
        if not isinstance(target_pose, CourtPoseTargetBatch):
            raise ValueError("Supervised mixed Court batch lacks a pose target.")
        self._pose_metrics[stage].update(result.decoded_pose, target_pose)

        geometry_tracker = self._pose_geometry_metrics.get(stage)
        if geometry_tracker is not None:
            kp_target = pose_call.targets.get("kp")
            if not isinstance(kp_target, Mapping):
                raise ValueError(
                    "Pose metrics require the canonical singleton KP target."
                )
            full_image_size = pose_call.targets.get("image_size")
            if not isinstance(full_image_size, Tensor):
                raise ValueError("Pose metrics require a typed image_size target.")
            ground_truth_points = cast(Tensor, kp_target["points_xy"])[mask].squeeze(2)
            point_visible = cast(Tensor, kp_target["point_visible"])[mask].squeeze(2)
            supervised_image_size = full_image_size[mask]
            if result.consistency is not None:
                geometry_tracker.update(
                    result.consistency,
                    ground_truth_points_normalized=ground_truth_points,
                    point_visible=point_visible,
                    image_size=supervised_image_size,
                )
            else:
                geometry_tracker.update_pose_prediction(
                    result.decoded_pose,
                    target_pose,
                    ground_truth_points_normalized=ground_truth_points,
                    point_visible=point_visible,
                    image_size=supervised_image_size,
                )
        return result


__all__ = ["MixedCourtDetectionLightningModule"]
