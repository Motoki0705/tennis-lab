"""Lightning training, metric, and reproducible prediction-bundle lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import torch
from hydra.utils import instantiate
from torch import Tensor, nn

from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.court_alignment.inference.decoder import (
    CourtInstanceBatch,
    CourtInstances,
    CourtPeakDetections,
    decode_keypoint_peaks,
    group_peak_votes,
)
from src.tasks.court_alignment.models.cnn import (
    validate_court_alignment_input,
    validate_court_alignment_output,
)
from src.tasks.court_alignment.training.losses import CourtAlignmentLoss


class CourtAlignmentLightningModule(BaseLightningModule):
    """Train the CNN against KP14 heatmaps and centre-vote targets."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.model = cast(nn.Module, instantiate(config.model))
        loss_fn = instantiate(config.loss)
        if not isinstance(loss_fn, CourtAlignmentLoss):
            raise TypeError("loss._target_ must construct CourtAlignmentLoss.")
        self.loss_fn = loss_fn
        self._metrics = {
            stage: instantiate(config.metrics)
            for stage in ("val", "test")
        }
        self._decoder_threshold = float(config.decoder.threshold)
        self._decoder_nms_kernel = int(config.decoder.nms_kernel)
        self._decoder_max_peaks = int(config.decoder.max_peaks)
        self._decoder_subpixel_refine = bool(config.decoder.subpixel_refine)
        self._cluster_distance_px = float(config.decoder.cluster_distance_px)
        self._max_instances = int(config.decoder.max_instances)
        self._test_loss_sums: dict[str, float] = {}
        self._test_loss_samples = 0

    def forward(self, image: Tensor) -> Mapping[str, Tensor]:
        """Predict full-resolution KP logits and centre votes."""
        return cast(Mapping[str, Tensor], self.model(image))

    @staticmethod
    def _tensor(batch: Mapping[str, object], key: str) -> Tensor:
        value = batch.get(key)
        if not isinstance(value, Tensor):
            raise TypeError(f"Court-alignment batch {key!r} must be a Tensor.")
        return value

    def _decode(
        self, heatmap_logits: Tensor, center_votes: Tensor
    ) -> tuple[CourtPeakDetections, CourtInstances]:
        peaks = decode_keypoint_peaks(
            heatmap_logits,
            center_votes,
            threshold=self._decoder_threshold,
            nms_kernel=self._decoder_nms_kernel,
            max_peaks=self._decoder_max_peaks,
            subpixel_refine=self._decoder_subpixel_refine,
        )
        grouped = group_peak_votes(
            peaks.keypoints_px,
            peaks.center_votes_px,
            peaks.valid,
            peaks.scores,
            cluster_distance_px=self._cluster_distance_px,
            max_instances=self._max_instances,
        )
        if isinstance(grouped, CourtInstanceBatch):
            grouped = CourtInstances((grouped,))
        return peaks, grouped

    def _shared_step(
        self,
        batch: Mapping[str, object],
        stage: str,
    ) -> Tensor:
        image = self._tensor(batch, "image")
        parameter = next(self.model.parameters(), torch.empty(0, device=image.device))
        validate_court_alignment_input(image, expected_dtype=parameter.dtype)
        output = self(image)
        typed_output = validate_court_alignment_output(output)
        heatmap_logits = typed_output.heatmap_logits
        center_votes = typed_output.center_votes
        target_heatmaps = self._tensor(batch, "target_heatmaps")
        target_center_votes = self._tensor(batch, "target_center_votes")
        target_center_vote_mask = self.loss_fn.validate_inputs(
            heatmap_logits,
            target_heatmaps,
            center_votes,
            target_center_votes,
            self._tensor(batch, "target_center_vote_mask"),
        )
        loss_output = self.loss_fn(
            heatmap_logits,
            target_heatmaps,
            center_votes,
            target_center_votes,
            target_center_vote_mask,
        )
        total = getattr(loss_output, "total", None)
        heatmap_loss = getattr(loss_output, "heatmap", None)
        vote_loss = getattr(loss_output, "center_vote", None)
        if not all(
            isinstance(value, Tensor)
            for value in (total, heatmap_loss, vote_loss)
        ):
            raise TypeError(
                "loss._target_ must return total, heatmap, and center_vote tensors."
            )
        total = cast(Tensor, total)
        heatmap_loss = cast(Tensor, heatmap_loss)
        vote_loss = cast(Tensor, vote_loss)
        batch_size = int(image.shape[0])
        for name, value in {
            "loss": total,
            "loss_heatmap": heatmap_loss,
            "loss_center_vote": vote_loss,
        }.items():
            self.log(
                f"{stage}/{name}",
                value,
                on_step=stage == "train" and name == "loss",
                on_epoch=True,
                prog_bar=name == "loss",
                batch_size=batch_size,
            )

        peaks: CourtPeakDetections | None = None
        instances: CourtInstances | None = None
        if stage != "train":
            # Multi-peak decoding and Python-side centre clustering are
            # evaluation operations.  Keeping them out of training avoids a
            # device synchronization for every candidate peak.
            with torch.no_grad():
                peaks, instances = self._decode(heatmap_logits, center_votes)
                self._metrics[stage].update(
                    instances,
                    self._tensor(batch, "keypoints"),
                    self._tensor(batch, "visibility"),
                    centers=self._tensor(batch, "centers"),
                    num_courts=self._tensor(batch, "num_courts"),
                    image_size=(int(image.shape[-2]), int(image.shape[-1])),
                    target_normalized=False,
                )

        if stage == "test":
            if peaks is None or instances is None:
                raise RuntimeError("Test decoding did not produce predictions.")
            for name, value in {
                "loss": total,
                "loss_heatmap": heatmap_loss,
                "loss_center_vote": vote_loss,
            }.items():
                self._test_loss_sums[name] = self._test_loss_sums.get(name, 0.0) + (
                    float(value.detach()) * batch_size
                )
            self._test_loss_samples += batch_size
            self.collect_test_predictions(
                batch,
                {"peaks": peaks, "instances": instances},
            )
        return total

    def training_step(
        self, batch: Mapping[str, object], batch_idx: int
    ) -> Tensor:
        del batch_idx
        return self._shared_step(batch, "train")

    def validation_step(
        self, batch: Mapping[str, object], batch_idx: int
    ) -> Tensor:
        del batch_idx
        return self._shared_step(batch, "val")

    def test_step(self, batch: Mapping[str, object], batch_idx: int) -> Tensor:
        del batch_idx
        return self._shared_step(batch, "test")

    def _reset_stage(self, stage: str) -> None:
        self._metrics[stage].reset()

    def on_validation_epoch_start(self) -> None:
        self._reset_stage("val")

    def on_test_epoch_start(self) -> None:
        self._reset_stage("test")
        self._reset_test_prediction_buffer()
        self._test_loss_sums.clear()
        self._test_loss_samples = 0

    def _flush_stage_metrics(self, stage: str) -> dict[str, float]:
        metrics = cast(dict[str, float], self._metrics[stage].compute())
        for name, value in metrics.items():
            self.log(
                f"{stage}/{name}",
                value,
                on_epoch=True,
                prog_bar=name
                in {"instance_f1", "instance_kp_mean_error_px"},
            )
        return metrics

    def on_validation_epoch_end(self) -> None:
        self._flush_stage_metrics("val")

    def on_test_epoch_end(self) -> None:
        all_metrics = self._flush_stage_metrics("test")
        headline_names = {
            "instance_precision",
            "instance_recall",
            "instance_f1",
            "instance_count_accuracy",
            "instance_count_mae",
            "instance_kp_mean_error_px",
            "instance_kp_pck_2px",
            "instance_kp_pck_4px",
            "matched_center_mean_error_px",
            "sim2_translation_error_px",
            "sim2_rotation_error_deg",
            "sim2_scale_relative_error",
        }
        metrics = {
            name: float(value)
            for name, value in all_metrics.items()
            if name in headline_names
        }
        diagnostic_metrics = {
            name: float(value)
            for name, value in all_metrics.items()
            if name not in headline_names
        }
        denominator = max(self._test_loss_samples, 1)
        diagnostic_metrics.update(
            {
                name: value / denominator
                for name, value in self._test_loss_sums.items()
            }
        )
        saved = self.save_test_predictions(
            metrics=metrics,
            diagnostic_metrics=diagnostic_metrics,
        )
        if saved is None:
            raise RuntimeError(
                "Court-alignment testing completed without a prediction payload."
            )
        print(f"[test] saved Court Alignment predictions -> {saved}")

    @staticmethod
    def _pad_instances(
        instances: CourtInstances,
        *,
        max_instances: int,
    ) -> dict[str, Tensor]:
        if any(sample.num_instances > max_instances for sample in instances.samples):
            raise ValueError("Decoded court count exceeds decoder.max_instances.")
        first = instances.samples[0]
        device = first.keypoints_px.device
        dtype = first.keypoints_px.dtype
        batch_size = len(instances)
        keypoints = torch.zeros(
            (batch_size, max_instances, 14, 2), device=device, dtype=dtype
        )
        scores = torch.zeros(
            (batch_size, max_instances, 14), device=device, dtype=dtype
        )
        valid = torch.zeros(
            (batch_size, max_instances, 14), device=device, dtype=torch.bool
        )
        centers = torch.zeros(
            (batch_size, max_instances, 2), device=device, dtype=dtype
        )
        semantic_count = torch.zeros(
            (batch_size, max_instances), device=device, dtype=torch.long
        )
        aggregate_confidence = torch.zeros(
            (batch_size, max_instances), device=device, dtype=dtype
        )
        geometry_residual = torch.zeros(
            (batch_size, max_instances), device=device, dtype=dtype
        )
        counts = torch.zeros((batch_size,), device=device, dtype=torch.long)
        for batch_index, sample in enumerate(instances.samples):
            if (
                sample.semantic_count is None
                or sample.aggregate_confidence is None
                or sample.geometry_residual_px is None
            ):
                raise RuntimeError(
                    "Decoded instances are missing ranking-quality diagnostics."
                )
            count = sample.num_instances
            counts[batch_index] = count
            keypoints[batch_index, :count] = sample.keypoints_px
            scores[batch_index, :count] = sample.scores
            valid[batch_index, :count] = sample.valid
            centers[batch_index, :count] = sample.centers_px
            semantic_count[batch_index, :count] = sample.semantic_count
            aggregate_confidence[
                batch_index, :count
            ] = sample.aggregate_confidence
            geometry_residual[
                batch_index, :count
            ] = sample.geometry_residual_px
        return {
            "pred_instance_keypoints_px": keypoints,
            "pred_instance_scores": scores,
            "pred_instance_valid": valid,
            "pred_instance_centers_px": centers,
            "pred_instance_semantic_count": semantic_count,
            "pred_instance_aggregate_confidence": aggregate_confidence,
            "pred_instance_geometry_residual_px": geometry_residual,
            "pred_num_instances": counts,
        }

    def test_prediction_payload(
        self,
        batch: Mapping[str, object],
        result: dict[str, object],
    ) -> dict[str, np.ndarray]:
        peaks = result.get("peaks")
        instances = result.get("instances")
        if not isinstance(peaks, CourtPeakDetections) or not isinstance(
            instances, CourtInstances
        ):
            raise TypeError("Test result requires decoded peaks and instances.")
        keypoints = self._tensor(batch, "keypoints")
        visibility = self._tensor(batch, "visibility")
        sample_ids = batch.get("sample_id")
        if not isinstance(sample_ids, (list, tuple)) or not all(
            isinstance(value, str) for value in sample_ids
        ):
            raise TypeError("sample_id must collate to a sequence of strings.")

        payload: dict[str, Any] = {
            "sample_id": np.asarray(sample_ids),
            "pred_peak_keypoints_px": peaks.keypoints_px,
            "pred_peak_scores": peaks.scores,
            "pred_peak_valid": peaks.valid,
            "pred_peak_center_votes_px": peaks.center_votes_px,
            "gt_peak_keypoints_px": keypoints.permute(0, 2, 1, 3),
            "gt_peak_valid": visibility.permute(0, 2, 1),
            "gt_instance_keypoints_px": keypoints,
            "gt_instance_valid": visibility,
            "gt_instance_centers_px": self._tensor(batch, "centers"),
            "gt_num_instances": self._tensor(batch, "num_courts"),
        }
        payload.update(
            self._pad_instances(instances, max_instances=self._max_instances)
        )
        return {name: self._to_numpy(value) for name, value in payload.items()}


__all__ = ["CourtAlignmentLightningModule"]
