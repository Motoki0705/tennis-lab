"""Per-head metrics for composable Court detection training."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Protocol, cast

import torch
from torch import Tensor

from src.tasks.court_detection.data.contracts import CourtTargetKind
from src.tasks.court_detection.geometry.pose import CourtDecodedPose
from src.tasks.court_detection.training.losses import rotation_geodesic_radians
from src.utils.data.heatmaps import heatmaps_to_peaks


class CourtDetectionMetrics:
    """Accumulate one metric under one resolved target-head contract."""

    def __init__(
        self,
        kind: CourtTargetKind,
        output_channels: int,
        *,
        singleton_kp: bool = False,
    ) -> None:
        if output_channels <= 0:
            raise ValueError("Court metric output_channels must be positive.")
        self.kind = kind
        self.output_channels = output_channels
        self.singleton_kp = singleton_kp
        self.reset()

    def update(
        self,
        logits: Tensor,
        target: object,
        *,
        image_size: Tensor,
    ) -> None:
        if logits.ndim != 4 or logits.shape[1] != self.output_channels:
            raise ValueError("Court metric logits disagree with the target head.")
        if image_size.shape != (logits.shape[0], 2) or image_size.dtype != torch.long:
            raise ValueError("Court metric image_size must be int64 [B,2].")
        if self.kind == "seg":
            self._update_seg(logits, cast(Tensor, target), image_size=image_size)
        elif self.kind == "kp":
            self._update_kp(
                logits,
                cast(Mapping[str, Tensor], target),
                image_size=image_size,
            )
        elif self.kind == "line":
            self._update_line(logits, cast(Tensor, target), image_size=image_size)
        else:  # pragma: no cover - typed construction rejects this
            raise ValueError(f"Unsupported Court metric target {self.kind!r}.")

    def _update_seg(
        self,
        logits: Tensor,
        target: Tensor,
        *,
        image_size: Tensor,
    ) -> None:
        predictions = logits.argmax(dim=1)
        for sample_index, size in enumerate(image_size.tolist()):
            height, width = (int(value) for value in size)
            prediction = predictions[sample_index, :height, :width]
            labels = target[sample_index, :height, :width]
            for class_index in range(self.output_channels):
                predicted = prediction == class_index
                expected = labels == class_index
                self._intersection[class_index] += float(
                    (predicted & expected).sum().item()
                )
                self._union[class_index] += float(
                    (predicted | expected).sum().item()
                )

    def _update_kp(
        self,
        logits: Tensor,
        target: Mapping[str, Tensor],
        *,
        image_size: Tensor,
    ) -> None:
        points = target["points_xy"]
        visible = target["point_visible"]
        if points.ndim != 4 or visible.shape != points.shape[:-1]:
            raise ValueError("Court KP metric target geometry is invalid.")
        if self.singleton_kp:
            self._update_singleton_kp(
                logits,
                points=points,
                visible=visible,
                image_size=image_size,
            )
            return
        max_peaks = points.shape[2]
        coordinates, _, predicted_valid = heatmaps_to_peaks(
            torch.sigmoid(logits),
            threshold=0.05,
            nms_kernel=7,
            max_peaks=max_peaks,
        )
        padded_height, padded_width = logits.shape[-2:]
        prediction_scale = logits.new_tensor(
            [
                float(max(padded_width - 1, 0)),
                float(max(padded_height - 1, 0)),
            ]
        )
        predicted_pixels = coordinates * prediction_scale
        for sample_index, size in enumerate(image_size.tolist()):
            height, width = (int(value) for value in size)
            target_scale = points.new_tensor(
                [
                    float(max(width - 1, 0)),
                    float(max(height - 1, 0)),
                ]
            )
            target_pixels = points[sample_index] * target_scale
            missing_penalty = float(torch.linalg.vector_norm(target_scale).item())
            for channel_index in range(self.output_channels):
                expected = target_pixels[channel_index][
                    visible[sample_index, channel_index]
                ]
                if expected.numel() == 0:
                    continue
                accepted = predicted_pixels[sample_index, channel_index][
                    predicted_valid[sample_index, channel_index]
                ]
                if accepted.numel() == 0:
                    self._kp_distances.extend(
                        [missing_penalty] * int(expected.shape[0])
                    )
                    continue
                pairwise = torch.cdist(
                    expected.to(dtype=torch.float32).unsqueeze(0),
                    accepted.to(dtype=torch.float32).unsqueeze(0),
                )[0]
                self._kp_distances.extend(
                    pairwise.amin(dim=1).detach().cpu().tolist()
                )

    def _update_singleton_kp(
        self,
        logits: Tensor,
        *,
        points: Tensor,
        visible: Tensor,
        image_size: Tensor,
    ) -> None:
        if points.shape != (logits.shape[0], self.output_channels, 1, 2):
            raise ValueError(
                "Singleton Court KP metric requires exact target shape (B,C,1,2)."
            )
        flat_index = logits.flatten(2).argmax(dim=-1)
        padded_height, padded_width = logits.shape[-2:]
        predicted_pixels = torch.stack(
            (
                (flat_index % padded_width).to(dtype=points.dtype),
                torch.div(flat_index, padded_width, rounding_mode="floor").to(
                    dtype=points.dtype
                ),
            ),
            dim=-1,
        )
        for sample_index, size in enumerate(image_size.tolist()):
            height, width = (int(value) for value in size)
            target_scale = points.new_tensor(
                [float(max(width - 1, 0)), float(max(height - 1, 0))]
            )
            target_pixels = points[sample_index, :, 0] * target_scale
            accepted = visible[sample_index, :, 0]
            distances = torch.linalg.vector_norm(
                predicted_pixels[sample_index] - target_pixels,
                dim=-1,
            )
            self._kp_distances.extend(distances[accepted].detach().cpu().tolist())

    def _update_line(
        self,
        logits: Tensor,
        target: Tensor,
        *,
        image_size: Tensor,
    ) -> None:
        predictions = torch.sigmoid(logits) > 0.5
        expected = target > 0.5
        for sample_index, size in enumerate(image_size.tolist()):
            height, width = (int(value) for value in size)
            prediction = predictions[sample_index, :, :height, :width]
            label = expected[sample_index, :, :height, :width]
            intersection = float((prediction & label).sum().item())
            union = float(prediction.sum().item() + label.sum().item())
            self._line_dice_sum += (2.0 * intersection + 1.0) / (union + 1.0)
            self._line_dice_count += 1

    def compute(self) -> dict[str, float]:
        if self.kind == "seg":
            values = [
                intersection / (union + 1.0e-8)
                for intersection, union in zip(
                    self._intersection,
                    self._union,
                    strict=True,
                )
            ]
            return {"miou": sum(values) / len(values)}
        if self.kind == "kp":
            return {
                "mean_dist": (
                    sum(self._kp_distances) / len(self._kp_distances)
                    if self._kp_distances
                    else 0.0
                )
            }
        return {
            "dice": (
                self._line_dice_sum / self._line_dice_count
                if self._line_dice_count
                else 0.0
            )
        }

    def reset(self) -> None:
        self._intersection = [0.0] * self.output_channels
        self._union = [0.0] * self.output_channels
        self._kp_distances: list[float] = []
        self._line_dice_sum = 0.0
        self._line_dice_count = 0


class _PoseMetricTarget(Protocol):
    @property
    def translation_m(self) -> Tensor: ...

    @property
    def rotation(self) -> Tensor: ...

    @property
    def log_focal(self) -> Tensor: ...


class CourtPoseMetrics:
    """Accumulate the explicit query pose metrics required by the checkpoint."""

    def __init__(self) -> None:
        self.reset()

    def update(
        self,
        prediction: CourtDecodedPose,
        target: _PoseMetricTarget,
    ) -> None:
        target_translation = target.translation_m.to(
            device=prediction.translation_m.device,
            dtype=prediction.translation_m.dtype,
        )
        target_rotation = target.rotation.to(
            device=prediction.rotation.device,
            dtype=prediction.rotation.dtype,
        )
        target_log = target.log_focal.to(
            device=prediction.log_focal.device,
            dtype=prediction.log_focal.dtype,
        )
        translation = torch.linalg.vector_norm(
            prediction.translation_m - target_translation,
            dim=-1,
        )
        rotation_deg = rotation_geodesic_radians(
            prediction.rotation,
            target_rotation,
        ) * (180.0 / math.pi)
        log_error = torch.abs(prediction.log_focal - target_log)
        target_focal = torch.exp(target_log)
        relative_focal = torch.abs(prediction.focal_px - target_focal) / target_focal
        self._translation.extend(translation.detach().cpu().tolist())
        self._rotation_deg.extend(rotation_deg.detach().cpu().tolist())
        self._focal_relative.extend(relative_focal.detach().cpu().tolist())
        self._log_focal.extend(log_error.detach().cpu().tolist())

    def compute(self) -> dict[str, float]:
        return {
            "translation_l2_m": self._mean(self._translation),
            "rotation_geodesic_deg": self._mean(self._rotation_deg),
            "focal_relative_error": self._mean(self._focal_relative),
            "log_focal_abs_error": self._mean(self._log_focal),
        }

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def reset(self) -> None:
        self._translation: list[float] = []
        self._rotation_deg: list[float] = []
        self._focal_relative: list[float] = []
        self._log_focal: list[float] = []


__all__ = ["CourtDetectionMetrics", "CourtPoseMetrics"]
