"""Bundle-aware padding collate for the sole Court dataset."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch
from torch import Tensor

from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetKind,
)


def _align8(value: int) -> int:
    return ((value + 7) // 8) * 8


def _pad_image(value: Tensor, *, height: int, width: int) -> Tensor:
    if value.ndim != 3:
        raise ValueError("Court image tensors must have shape [C,H,W].")
    output = value.new_zeros((value.shape[0], height, width))
    output[:, : value.shape[-2], : value.shape[-1]] = value
    return output


def _pad_spatial_2d(value: Tensor, *, height: int, width: int) -> Tensor:
    if value.ndim != 2:
        raise ValueError("Court 2-D targets must have shape [H,W].")
    output = value.new_zeros((height, width))
    output[: value.shape[-2], : value.shape[-1]] = value
    return output


def _pad_spatial_3d(value: Tensor, *, height: int, width: int) -> Tensor:
    if value.ndim != 3:
        raise ValueError("Court dense targets must have shape [C,H,W].")
    output = value.new_zeros((value.shape[0], height, width))
    output[:, : value.shape[-2], : value.shape[-1]] = value
    return output


def _collate_keypoints(
    payloads: list[Mapping[str, object]],
    *,
    height: int,
    width: int,
) -> dict[str, Tensor]:
    heatmaps = [cast(Tensor, payload["heatmap"]) for payload in payloads]
    points = [cast(Tensor, payload["points_xy"]) for payload in payloads]
    visible = [cast(Tensor, payload["point_visible"]) for payload in payloads]
    physical = [cast(Tensor, payload["physical_indices"]) for payload in payloads]
    channels = heatmaps[0].shape[0]
    if any(value.shape[0] != channels for value in heatmaps):
        raise ValueError("Court KP channel count changed within one data source.")
    max_points = max(value.shape[1] for value in points)
    padded_points: list[Tensor] = []
    padded_visible: list[Tensor] = []
    padded_physical: list[Tensor] = []
    for point_value, visible_value, physical_value in zip(
        points, visible, physical, strict=True
    ):
        point_output = point_value.new_zeros((channels, max_points, 2))
        visible_output = torch.zeros((channels, max_points), dtype=torch.bool)
        physical_output = torch.full(
            (channels, max_points), -1, dtype=torch.long
        )
        count = point_value.shape[1]
        point_output[:, :count] = point_value
        visible_output[:, :count] = visible_value
        physical_output[:, :count] = physical_value
        padded_points.append(point_output)
        padded_visible.append(visible_output)
        padded_physical.append(physical_output)
    return {
        "heatmap": torch.stack(
            [_pad_spatial_3d(value, height=height, width=width) for value in heatmaps]
        ),
        "points_xy": torch.stack(padded_points),
        "point_visible": torch.stack(padded_visible),
        "physical_indices": torch.stack(padded_physical),
    }


def court_detection_collate(
    batch: list[dict[str, object]],
    *,
    bundle: CourtTargetBundleSpec,
) -> dict[str, object]:
    """Pad all selected heads against the same batch spatial envelope."""
    if not batch:
        raise ValueError("Court collate requires a non-empty batch.")
    images = [cast(Tensor, sample["image"]) for sample in batch]
    height = _align8(max(value.shape[-2] for value in images))
    width = _align8(max(value.shape[-1] for value in images))
    target_mappings = [
        cast(Mapping[CourtTargetKind, object], sample["targets"]) for sample in batch
    ]
    targets: dict[CourtTargetKind, object] = {}
    for kind in bundle.kinds:
        values = [mapping[kind] for mapping in target_mappings]
        if kind == "kp":
            targets[kind] = _collate_keypoints(
                [cast(Mapping[str, object], value) for value in values],
                height=height,
                width=width,
            )
        elif kind == "seg":
            targets[kind] = torch.stack(
                [
                    _pad_spatial_2d(cast(Tensor, value), height=height, width=width)
                    for value in values
                ]
            )
        elif kind == "line":
            targets[kind] = torch.stack(
                [
                    _pad_spatial_3d(cast(Tensor, value), height=height, width=width)
                    for value in values
                ]
            )
        else:  # pragma: no cover - bundle validation rejects this
            raise ValueError(f"Unsupported Court target kind: {kind!r}.")
    return {
        "image": torch.stack(
            [_pad_image(value, height=height, width=width) for value in images]
        ),
        "targets": targets,
        "image_size": torch.stack(
            [cast(Tensor, sample["image_size"]) for sample in batch]
        ),
        "sample_id": [cast(str, sample["sample_id"]) for sample in batch],
        "metadata": [
            cast(Mapping[str, object], sample["metadata"]) for sample in batch
        ],
    }


__all__ = ["court_detection_collate"]
