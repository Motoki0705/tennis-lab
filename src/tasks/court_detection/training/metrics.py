"""Per-head metrics for composable Court detection training."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch
from torch import Tensor

from src.tasks.court_detection.data.contracts import CourtTargetKind
from src.utils.data.heatmaps import heatmaps_to_peaks


class CourtDetectionMetrics:
    """Accumulate one metric under one resolved target-head contract."""

    def __init__(self, kind: CourtTargetKind, output_channels: int) -> None:
        if output_channels <= 0:
            raise ValueError("Court metric output_channels must be positive.")
        self.kind = kind
        self.output_channels = output_channels
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


__all__ = ["CourtDetectionMetrics"]
