"""Pixel and instance diagnostics for sigma ablation experiments."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from src.tasks.court_alignment.inference.decoder import (
    CourtInstanceBatch,
    CourtInstances,
    CourtPeakDetections,
    decode_court_instances,
    decode_keypoint_peaks,
    group_peak_votes,
)


def _target_layout(keypoints: Tensor, *, batch_size: int) -> tuple[Tensor, bool]:
    """Return targets as ``(B,14,N,2)`` and whether they were normalised."""
    if keypoints.ndim != 4 or keypoints.shape[0] != batch_size or keypoints.shape[-1] != 2:
        raise ValueError("keypoints must have shape (B,N,14,2) or (B,14,N,2).")
    if keypoints.shape[1] == 14:
        canonical = keypoints
    elif keypoints.shape[2] == 14:
        canonical = keypoints.permute(0, 2, 1, 3)
    else:
        raise ValueError("keypoints must have one axis of length fourteen.")
    if not keypoints.is_floating_point() or not bool(torch.isfinite(keypoints).all()):
        raise ValueError("keypoints must be finite floating point values.")
    normalised = bool(torch.all((canonical >= 0.0) & (canonical <= 1.0)))
    return canonical, normalised


def _visibility_layout(visibility: Tensor | None, *, target_shape: tuple[int, ...]) -> Tensor:
    if visibility is None:
        return torch.ones(target_shape, dtype=torch.bool)
    if visibility.ndim != 3:
        raise ValueError("visibility must have shape (B,N,14) or (B,14,N).")
    if visibility.shape == target_shape:
        result = visibility
    elif visibility.shape[0] == target_shape[0] and visibility.shape[2:] == target_shape[1:2] and visibility.shape[1] == target_shape[2]:
        result = visibility.permute(0, 2, 1)
    else:
        raise ValueError("visibility shape must match keypoints' court/semantic axes.")
    if result.dtype != torch.bool:
        raise TypeError("visibility must have boolean dtype.")
    return result


def _image_scales(
    image_size: Tensor | tuple[int, int] | None,
    *,
    batch_size: int,
    default_height: int,
    default_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if image_size is None:
        values = torch.tensor(
            (default_height, default_width), device=device, dtype=torch.long
        ).expand(batch_size, -1)
    elif isinstance(image_size, Tensor):
        if image_size.shape != (batch_size, 2) or image_size.dtype not in {
            torch.int32,
            torch.int64,
        }:
            raise ValueError("image_size must have shape (B,2) and integer dtype.")
        if image_size.device != device:
            raise ValueError("image_size must share the target device.")
        values = image_size
    else:
        if len(image_size) != 2:
            raise ValueError("image_size must be a (height,width) pair.")
        values = torch.tensor(image_size, device=device, dtype=torch.long).expand(
            batch_size, -1
        )
    if bool(torch.any(values <= 0)):
        raise ValueError("image_size values must be positive.")
    return values.to(dtype=dtype)[:, [1, 0]] - 1.0


def _as_detections(predictions: CourtPeakDetections | Tensor, center_votes: Tensor | None, **decode_options: object) -> CourtPeakDetections:
    if isinstance(predictions, CourtPeakDetections):
        return predictions
    if center_votes is None:
        center_votes = torch.zeros((predictions.shape[0], 2, *predictions.shape[-2:]), dtype=predictions.dtype, device=predictions.device)
    return decode_keypoint_peaks(predictions, center_votes, **decode_options)  # type: ignore[arg-type]


def peak_metrics(
    predictions: CourtPeakDetections | Tensor,
    keypoints: Tensor,
    visibility: Tensor | None = None,
    *,
    center_votes: Tensor | None = None,
    image_size: Tensor | tuple[int, int] | None = None,
    target_normalized: bool | None = None,
    threshold: float = 0.25,
    nms_kernel: int = 3,
    max_peaks: int = 8,
) -> dict[str, float]:
    """Measure nearest-prediction error and recall at 2/4 pixels per KP."""
    if isinstance(predictions, Tensor) and predictions.ndim != 4:
        raise ValueError("prediction logits must have shape (B,14,H,W).")
    batch_size = predictions.keypoints_px.shape[0] if isinstance(predictions, CourtPeakDetections) else predictions.shape[0]
    targets, inferred_normalized = _target_layout(keypoints, batch_size=batch_size)
    visible = _visibility_layout(visibility, target_shape=targets.shape[:3])
    if visible.device != targets.device:
        raise ValueError("keypoints and visibility must share a device.")
    detections = _as_detections(
        predictions,
        center_votes,
        threshold=threshold,
        nms_kernel=nms_kernel,
        max_peaks=max_peaks,
        subpixel_refine=True,
    )
    if isinstance(predictions, Tensor):
        map_height, map_width = predictions.shape[-2:]
    else:
        map_height, map_width = 1, 1
    scale = _image_scales(
        image_size,
        batch_size=batch_size,
        default_height=map_height,
        default_width=map_width,
        device=targets.device,
        dtype=targets.dtype,
    )
    use_normalized = inferred_normalized if target_normalized is None else target_normalized
    target_px = targets * scale[:, None, None] if use_normalized else targets
    errors: list[Tensor] = []
    for batch_index in range(batch_size):
        for channel in range(14):
            expected = target_px[batch_index, channel][visible[batch_index, channel]]
            accepted = detections.keypoints_px[batch_index, channel][detections.valid[batch_index, channel]]
            if expected.numel() == 0:
                continue
            if accepted.numel() == 0:
                penalty = torch.linalg.vector_norm(scale[batch_index])
                errors.append(penalty.expand(expected.shape[0]))
            else:
                errors.append(torch.cdist(expected[None], accepted[None])[0].amin(dim=1))
    if not errors:
        result = {"peak_mean_error_px": 0.0, "recall_at_2px": 0.0, "recall_at_4px": 0.0, "peak_count": 0.0}
    else:
        all_errors = torch.cat(errors)
        result = {
            "peak_mean_error_px": float(all_errors.mean().detach()),
            "recall_at_2px": float((all_errors <= 2.0).float().mean().detach()),
            "recall_at_4px": float((all_errors <= 4.0).float().mean().detach()),
            "peak_count": float(all_errors.numel()),
        }
    # Common aliases make dashboards from old Court metrics and this prototype
    # directly comparable.
    result["kp_mean_distance_px"] = result["peak_mean_error_px"]
    result["recall_2px"] = result["recall_at_2px"]
    result["recall_4px"] = result["recall_at_4px"]
    return result


def instance_grouping_metrics(
    predictions: CourtInstances | Sequence[CourtInstanceBatch],
    centers: Tensor,
    *,
    num_courts: Tensor | None = None,
    image_size: Tensor | tuple[int, int] | None = None,
    target_normalized: bool | None = None,
) -> dict[str, float]:
    """Report count and center-association diagnostics for multiple courts."""
    samples = predictions.samples if isinstance(predictions, CourtInstances) else tuple(predictions)
    if centers.ndim != 3 or centers.shape[-1] != 2 or centers.shape[0] != len(samples):
        raise ValueError("centers must have shape (B,N,2) matching predictions.")
    if not centers.is_floating_point() or not bool(torch.isfinite(centers).all()):
        raise ValueError("centers must be finite floating point values.")
    if num_courts is not None:
        if num_courts.shape != (len(samples),) or num_courts.dtype not in {
            torch.int32,
            torch.int64,
        }:
            raise ValueError("num_courts must have shape (B,) and integer dtype.")
        if num_courts.device != centers.device or bool(torch.any(num_courts < 0)):
            raise ValueError("num_courts must be non-negative on the centers device.")
        if bool(torch.any(num_courts > centers.shape[1])):
            raise ValueError("num_courts cannot exceed the centers padding axis.")
    inferred_normalized = bool(torch.all((centers >= 0.0) & (centers <= 1.0)))
    use_normalized = inferred_normalized if target_normalized is None else target_normalized
    if use_normalized:
        if image_size is None:
            raise ValueError("image_size is required for normalised center targets.")
        if isinstance(image_size, Tensor):
            scales = image_size.to(device=centers.device, dtype=centers.dtype)[:, [1, 0]] - 1.0
        else:
            scales = centers.new_tensor((image_size[1] - 1.0, image_size[0] - 1.0)).expand(len(samples), -1)
        centers = centers * scales[:, None]
    count_errors: list[float] = []
    matched_errors: list[Tensor] = []
    matched = 0
    total = 0
    for sample_index, sample in enumerate(samples):
        target_count = (
            int(num_courts[sample_index].item())
            if num_courts is not None
            else centers.shape[1]
        )
        target = centers[sample_index, :target_count]
        count_errors.append(abs(sample.num_instances - target.shape[0]))
        if sample.num_instances == 0 or target.shape[0] == 0:
            total += int(target.shape[0])
            continue
        distances = torch.cdist(target, sample.centers_px)
        nearest = distances.amin(dim=1)
        matched_errors.append(nearest)
        matched += int((nearest <= 4.0).sum())
        total += int(target.shape[0])
    target_counts = [
        int(num_courts[index].item()) if num_courts is not None else centers[index].shape[0]
        for index in range(len(samples))
    ]
    result = {
        "instance_count_error": float(sum(count_errors) / max(len(count_errors), 1)),
        "instance_count_accuracy": float(sum(error == 0 for error in count_errors) / max(len(count_errors), 1)),
        "grouping_center_mean_error_px": float(torch.cat(matched_errors).mean()) if matched_errors else 0.0,
        "grouping_recall_at_4px": float(matched / max(total, 1)),
        "predicted_instance_count": float(sum(sample.num_instances for sample in samples) / max(len(samples), 1)),
        "target_instance_count": float(sum(target_counts) / max(len(samples), 1)),
    }
    result["grouping_mean_error_px"] = result["grouping_center_mean_error_px"]
    return result


def compute_alignment_metrics(
    predictions: CourtPeakDetections | Tensor,
    keypoints: Tensor,
    visibility: Tensor | None = None,
    *,
    center_votes: Tensor | None = None,
    centers: Tensor | None = None,
    num_courts: Tensor | None = None,
    image_size: Tensor | tuple[int, int] | None = None,
    threshold: float = 0.25,
    nms_kernel: int = 3,
    max_peaks: int = 8,
    subpixel_refine: bool = True,
    cluster_distance_px: float = 12.0,
    max_instances: int | None = None,
) -> dict[str, float]:
    """Compute peak metrics and optional instance grouping diagnostics."""
    result = peak_metrics(
        predictions,
        keypoints,
        visibility,
        center_votes=center_votes,
        image_size=image_size,
        threshold=threshold,
        nms_kernel=nms_kernel,
        max_peaks=max_peaks,
    )
    if centers is not None:
        if isinstance(predictions, Tensor):
            instances = decode_court_instances(
                predictions,
                center_votes
                if center_votes is not None
                else torch.zeros(
                    (predictions.shape[0], 2, *predictions.shape[-2:]),
                    device=predictions.device,
                    dtype=predictions.dtype,
                ),
                threshold=threshold,
                nms_kernel=nms_kernel,
                max_peaks=max_peaks,
                subpixel_refine=subpixel_refine,
                cluster_distance_px=cluster_distance_px,
                max_instances=max_instances,
            )
        else:
            grouped = group_peak_votes(
                predictions.keypoints_px,
                predictions.center_votes_px,
                predictions.valid,
                predictions.scores,
                cluster_distance_px=cluster_distance_px,
                max_instances=max_instances,
            )
            instances = (
                CourtInstances((grouped,))
                if isinstance(grouped, CourtInstanceBatch)
                else grouped
            )
        result.update(
            instance_grouping_metrics(
                instances,
                centers,
                num_courts=num_courts,
                image_size=image_size,
            )
        )
    return result


class CourtAlignmentMetrics:
    """Accumulate alignment metrics over an epoch."""

    def __init__(
        self,
        threshold: float = 0.25,
        nms_kernel: int = 3,
        max_peaks: int = 8,
        subpixel_refine: bool = True,
        cluster_distance_px: float = 12.0,
        max_instances: int | None = None,
    ) -> None:
        self.threshold = threshold
        self.nms_kernel = nms_kernel
        self.max_peaks = max_peaks
        self.subpixel_refine = subpixel_refine
        self.cluster_distance_px = cluster_distance_px
        self.max_instances = max_instances
        self._values: list[dict[str, float]] = []

    def reset(self) -> None:
        self._values.clear()

    def update(
        self,
        predictions: CourtPeakDetections | Tensor,
        keypoints: Tensor,
        visibility: Tensor | None = None,
        *,
        center_votes: Tensor | None = None,
        centers: Tensor | None = None,
        image_size: Tensor | tuple[int, int] | None = None,
    ) -> None:
        self._values.append(
            compute_alignment_metrics(
                predictions,
                keypoints,
                visibility,
                center_votes=center_votes,
                centers=centers,
                image_size=image_size,
                threshold=self.threshold,
                nms_kernel=self.nms_kernel,
                max_peaks=self.max_peaks,
                subpixel_refine=self.subpixel_refine,
                cluster_distance_px=self.cluster_distance_px,
                max_instances=self.max_instances,
            )
        )

    def compute(self) -> dict[str, float]:
        if not self._values:
            return {"peak_mean_error_px": 0.0, "recall_at_2px": 0.0, "recall_at_4px": 0.0}
        keys = self._values[0].keys()
        return {key: sum(value[key] for value in self._values if key in value) / max(sum(key in value for value in self._values), 1) for key in keys}


__all__ = [
    "CourtAlignmentMetrics",
    "compute_alignment_metrics",
    "instance_grouping_metrics",
    "peak_metrics",
]
