"""Lightning lifecycle for LoRA-adapted DINO oriented-court detection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from hydra.utils import instantiate
from torch import Tensor, nn

from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.court_alignment.geometry.oriented_box import build_detr_court_targets
from src.tasks.court_alignment.inference.detr_decoder import (
    CourtDetrDetections,
    decode_detr_courts,
)
from src.tasks.court_alignment.models.dino_detector import validate_dino_court_output
from src.tasks.court_alignment.models.dino_input import validate_dino_heatmaps
from src.tasks.court_alignment.training.detr_losses import CourtDetrCriterion
from src.tasks.court_alignment.training.detr_metrics import CourtDetrMetrics
from src.utils.configuration import PathRole


class DinoCourtAlignmentLightningModule(BaseLightningModule):
    """Fine-tune DINO with LoRA and court-specific oriented-box heads."""

    def __init__(
        self,
        config: Any,
        *,
        model: nn.Module | None = None,
        criterion: CourtDetrCriterion | None = None,
    ) -> None:
        super().__init__(config)
        self.model = self._build_model(config) if model is None else model
        if not isinstance(self.model, nn.Module):
            raise TypeError("DINO court model must be a torch.nn.Module.")
        resolved_criterion = instantiate(config.loss) if criterion is None else criterion
        if not isinstance(resolved_criterion, CourtDetrCriterion):
            raise TypeError("loss._target_ must construct CourtDetrCriterion.")
        self.loss_fn = resolved_criterion
        self._metrics = {
            stage: instantiate(config.metrics)
            for stage in ("val", "test")
        }
        if any(not isinstance(metric, CourtDetrMetrics) for metric in self._metrics.values()):
            raise TypeError("metrics._target_ must construct CourtDetrMetrics.")
        self._decoder_threshold = float(config.decoder.threshold)
        self._decoder_class_index = int(config.decoder.class_index)
        self._decoder_top_k = int(config.decoder.top_k)
        self._test_loss_sums: dict[str, float] = {}
        self._test_loss_samples = 0

        trainable = tuple(
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
        )
        if not trainable:
            raise RuntimeError("DINO court fine-tuning has no trainable parameters.")
        trainable_count = sum(
            parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad
        )
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        print(
            "[dino] trainable parameters "
            f"{trainable_count}/{parameter_count} "
            f"({100.0 * trainable_count / max(parameter_count, 1):.4f}%): "
            + ", ".join(trainable)
        )

    def _build_model(self, config: Any) -> nn.Module:
        repository = self._model_input_path(
            str(config.model.repository),
            role=PathRole.EXTERNAL_ASSET,
        )
        checkpoint_path = self._model_input_path(
            str(config.model.checkpoint_path),
            # The released COCO initialization is a repository input asset,
            # not a task-run checkpoint used by resume/evaluation.
            role=PathRole.PROJECT,
        )
        return cast(
            nn.Module,
            instantiate(
                config.model,
                repository=repository,
                checkpoint_path=checkpoint_path,
            ),
        )

    def _model_input_path(self, raw: str, *, role: PathRole) -> Path:
        candidate = Path(raw).expanduser()
        if candidate.is_absolute():
            resolved: Path = self.path_resolver.validate(role, candidate)
        elif role is PathRole.PROJECT:
            # A pretrained asset may legitimately live below the repository's
            # ``ckpt/`` directory. Project-relative aliases are rejected by
            # the derived-child resolver, so build the explicit absolute path
            # and retain the same containment validation.
            resolved = self.path_resolver.validate(
                role,
                self.path_resolver.roots.project_root / candidate,
            )
        else:
            resolved = self.path_resolver.resolve(role, raw)
        return resolved

    def forward(
        self,
        image: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
    ) -> Mapping[str, object]:
        return cast(Mapping[str, object], self.model(image, targets))

    @staticmethod
    def _tensor(batch: Mapping[str, object], key: str) -> Tensor:
        value = batch.get(key)
        if not isinstance(value, Tensor):
            raise TypeError(f"Court-alignment batch {key!r} must be a Tensor.")
        return value

    def _targets(
        self,
        batch: Mapping[str, object],
        *,
        image_size: tuple[int, int],
    ) -> list[dict[str, Tensor]]:
        return cast(
            list[dict[str, Tensor]],
            build_detr_court_targets(
                self._tensor(batch, "keypoints"),
                self._tensor(batch, "visibility"),
                image_size=image_size,
                class_index=self._decoder_class_index,
            ),
        )

    def _decode(
        self,
        output: Mapping[str, object],
        *,
        image_size: tuple[int, int],
    ) -> CourtDetrDetections:
        tensors: dict[str, Tensor] = {}
        for name in ("pred_logits", "pred_boxes", "pred_court_boxes"):
            value = output.get(name)
            if not isinstance(value, Tensor):
                raise TypeError(f"DINO output {name!r} must be a Tensor.")
            tensors[name] = value
        return decode_detr_courts(
            tensors["pred_logits"],
            tensors["pred_boxes"],
            tensors["pred_court_boxes"],
            image_size=image_size,
            class_index=self._decoder_class_index,
            threshold=self._decoder_threshold,
            top_k=self._decoder_top_k,
        )

    def _shared_step(self, batch: Mapping[str, object], stage: str) -> Tensor:
        image = self._tensor(batch, "image")
        validate_dino_heatmaps(image)
        image_size = (int(image.shape[-2]), int(image.shape[-1]))
        targets = self._targets(batch, image_size=image_size)
        output = validate_dino_court_output(
            self(image, targets if stage == "train" else None)
        )
        self.loss_fn.validate_inputs(output, targets)
        losses = self.loss_fn(output, targets)
        total = losses.get("loss_total")
        if not isinstance(total, Tensor):
            raise TypeError("CourtDetrCriterion must return tensor loss_total.")
        batch_size = int(image.shape[0])
        self.log(
            f"{stage}/loss",
            total,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        for name, value in losses.items():
            if not isinstance(value, Tensor):
                raise TypeError(f"DETR loss {name!r} must be a Tensor.")
            self.log(
                f"{stage}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )

        detections: CourtDetrDetections | None = None
        if stage != "train":
            with torch.no_grad():
                detections = self._decode(output, image_size=image_size)
                self._metrics[stage].update(
                    detections,
                    targets,
                    image_size=image_size,
                )
        if stage == "test":
            if detections is None:
                raise RuntimeError("Test decoding did not produce DINO detections.")
            for name, value in losses.items():
                self._test_loss_sums[name] = self._test_loss_sums.get(name, 0.0) + (
                    float(value.detach()) * batch_size
                )
            self._test_loss_samples += batch_size
            self.collect_test_predictions(
                batch,
                {"detections": detections, "targets": targets},
            )
        return total

    def training_step(self, batch: Mapping[str, object], batch_idx: int) -> Tensor:
        del batch_idx
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Mapping[str, object], batch_idx: int) -> Tensor:
        del batch_idx
        return self._shared_step(batch, "val")

    def test_step(self, batch: Mapping[str, object], batch_idx: int) -> Tensor:
        del batch_idx
        return self._shared_step(batch, "test")

    def on_validation_epoch_start(self) -> None:
        self._metrics["val"].reset()

    def on_test_epoch_start(self) -> None:
        self._metrics["test"].reset()
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
                prog_bar=name in {"instance_f1", "matched_corner_mean_error_px"},
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
            "matched_center_mean_error_px",
            "matched_scale_relative_error",
            "matched_axial_angle_mean_error_deg",
            "matched_corner_mean_error_px",
        }
        denominator = max(self._test_loss_samples, 1)
        diagnostics = {
            name: float(value)
            for name, value in all_metrics.items()
            if name not in headline_names
        }
        diagnostics.update(
            {
                name: value / denominator
                for name, value in self._test_loss_sums.items()
            }
        )
        saved = self.save_test_predictions(
            metrics={
                name: float(value)
                for name, value in all_metrics.items()
                if name in headline_names
            },
            diagnostic_metrics=diagnostics,
        )
        if saved is None:
            raise RuntimeError("DINO testing completed without a prediction payload.")
        print(f"[test] saved DINO Court Alignment predictions -> {saved}")

    def optimizer_param_groups(self) -> list[dict[str, Any]]:
        parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError("DINO court fine-tuning has no optimizer parameters.")
        return [{"params": parameters}]

    @staticmethod
    def _pad_detections(
        detections: CourtDetrDetections,
        *,
        max_instances: int,
    ) -> dict[str, Tensor]:
        if any(sample.num_instances > max_instances for sample in detections.samples):
            raise ValueError("Decoded court count exceeds decoder.top_k.")
        first = detections.samples[0]
        device = first.scores.device
        dtype = first.scores.dtype
        batch_size = len(detections)
        payload = {
            "pred_scores": torch.zeros((batch_size, max_instances), device=device, dtype=dtype),
            "pred_query_indices": torch.full(
                (batch_size, max_instances), -1, device=device, dtype=torch.long
            ),
            "pred_aabb_cxcywh_normalized": torch.zeros(
                (batch_size, max_instances, 4), device=device, dtype=dtype
            ),
            "pred_centers_px": torch.zeros(
                (batch_size, max_instances, 2), device=device, dtype=dtype
            ),
            "pred_long_sides_px": torch.zeros(
                (batch_size, max_instances), device=device, dtype=dtype
            ),
            "pred_short_sides_px": torch.zeros(
                (batch_size, max_instances), device=device, dtype=dtype
            ),
            "pred_axial_vectors": torch.zeros(
                (batch_size, max_instances, 2), device=device, dtype=dtype
            ),
            "pred_corners_px": torch.zeros(
                (batch_size, max_instances, 4, 2), device=device, dtype=dtype
            ),
            "pred_rotation_rad": torch.zeros(
                (batch_size, max_instances), device=device, dtype=dtype
            ),
            "pred_scale_px_per_metre": torch.zeros(
                (batch_size, max_instances), device=device, dtype=dtype
            ),
            "pred_num_instances": torch.zeros(
                (batch_size,), device=device, dtype=torch.long
            ),
        }
        field_names = {
            "pred_scores": "scores",
            "pred_query_indices": "query_indices",
            "pred_aabb_cxcywh_normalized": "aabb_cxcywh_normalized",
            "pred_centers_px": "centers_px",
            "pred_long_sides_px": "long_sides_px",
            "pred_short_sides_px": "short_sides_px",
            "pred_axial_vectors": "axial_vectors",
            "pred_corners_px": "corners_px",
            "pred_rotation_rad": "rotation_rad",
            "pred_scale_px_per_metre": "scale_px_per_metre",
        }
        for batch_index, sample in enumerate(detections.samples):
            count = sample.num_instances
            payload["pred_num_instances"][batch_index] = count
            for output_name, field_name in field_names.items():
                payload[output_name][batch_index, :count] = getattr(sample, field_name)
        return payload

    @staticmethod
    def _pad_targets(
        targets: Sequence[Mapping[str, Tensor]],
        *,
        max_instances: int,
    ) -> dict[str, Tensor]:
        first = targets[0]
        first_boxes = first["boxes"]
        device = first_boxes.device
        dtype = first_boxes.dtype
        batch_size = len(targets)
        labels = torch.full(
            (batch_size, max_instances), -1, device=device, dtype=torch.long
        )
        boxes = torch.zeros(
            (batch_size, max_instances, 4), device=device, dtype=dtype
        )
        court_boxes = torch.zeros(
            (batch_size, max_instances, 5), device=device, dtype=dtype
        )
        counts = torch.zeros((batch_size,), device=device, dtype=torch.long)
        for batch_index, target in enumerate(targets):
            count = int(target["labels"].shape[0])
            if count > max_instances:
                raise ValueError("DETR target count exceeds padded dataset capacity.")
            counts[batch_index] = count
            labels[batch_index, :count] = target["labels"]
            boxes[batch_index, :count] = target["boxes"]
            court_boxes[batch_index, :count] = target["court_boxes"]
        return {
            "gt_labels": labels,
            "gt_aabb_cxcywh_normalized": boxes,
            "gt_court_boxes_normalized": court_boxes,
            "gt_num_instances": counts,
        }

    def test_prediction_payload(
        self,
        batch: Mapping[str, object],
        result: dict[str, object],
    ) -> dict[str, np.ndarray]:
        detections = result.get("detections")
        targets = result.get("targets")
        if not isinstance(detections, CourtDetrDetections):
            raise TypeError("Test result requires decoded DINO detections.")
        if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
            raise TypeError("Test result requires a sequence of DETR targets.")
        sample_ids = batch.get("sample_id")
        if not isinstance(sample_ids, (list, tuple)) or not all(
            isinstance(value, str) for value in sample_ids
        ):
            raise TypeError("sample_id must collate to a sequence of strings.")
        keypoints = self._tensor(batch, "keypoints")
        payload: dict[str, Any] = {
            "sample_id": np.asarray(sample_ids),
            "gt_instance_keypoints_px": keypoints,
            "gt_instance_valid": self._tensor(batch, "visibility"),
            "gt_num_courts_unfiltered": self._tensor(batch, "num_courts"),
        }
        payload.update(
            self._pad_detections(detections, max_instances=self._decoder_top_k)
        )
        payload.update(
            self._pad_targets(targets, max_instances=int(keypoints.shape[1]))
        )
        return {name: self._to_numpy(value) for name, value in payload.items()}


__all__ = ["DinoCourtAlignmentLightningModule"]
