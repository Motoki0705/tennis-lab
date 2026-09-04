"""Query decoder for DINO oriented court detections.

The decoder performs confidence thresholding and optional top-k truncation
only.  It deliberately does not apply NMS: DINO's one-to-one set objective is
responsible for duplicate suppression, and multiple nearby court queries must
remain independently observable during evaluation.

Coordinates are pixel ``(x,y)``.  ``rotation_rad`` is the canonical court
rotation in ``[0,pi)``: zero means the physical 23.77 m long edge is parallel
to positive image ``y``.  Since a rectangle cannot distinguish its two ends,
rotation is axial and a 180-degree reversal is the same prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from src.tasks.court_alignment.geometry.court import COURT_LENGTH_M
from src.tasks.court_alignment.geometry.oriented_box import (
    COURT_SHORT_TO_LONG_RATIO,
    court_rotation_from_axial_vector,
    decode_raw_court_boxes,
    oriented_box_corners,
)


@dataclass(frozen=True, slots=True)
class CourtDetrDetectionBatch:
    """Variable-length DINO court detections for one input image."""

    scores: Tensor
    query_indices: Tensor
    aabb_cxcywh_normalized: Tensor
    centers_px: Tensor
    long_sides_px: Tensor
    short_sides_px: Tensor
    axial_vectors: Tensor
    corners_px: Tensor
    rotation_rad: Tensor
    scale_px_per_metre: Tensor

    def __post_init__(self) -> None:
        count = self.scores.shape[0]
        expected = {
            "scores": (count,),
            "query_indices": (count,),
            "aabb_cxcywh_normalized": (count, 4),
            "centers_px": (count, 2),
            "long_sides_px": (count,),
            "short_sides_px": (count,),
            "axial_vectors": (count, 2),
            "corners_px": (count, 4, 2),
            "rotation_rad": (count,),
            "scale_px_per_metre": (count,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}.")
            if value.device != self.scores.device:
                raise ValueError("All decoded detection tensors must share a device.")
        if self.query_indices.dtype != torch.long:
            raise TypeError("query_indices must have int64 dtype.")
        for name in expected:
            if name == "query_indices":
                continue
            value = getattr(self, name)
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite floating point.")

    @property
    def num_instances(self) -> int:
        return int(self.scores.shape[0])

    @property
    def translation_px(self) -> Tensor:
        """Alias exposing the court translation as OBB center pixels."""

        return self.centers_px


@dataclass(frozen=True, slots=True)
class CourtDetrDetections:
    """Batch wrapper around variable-length DINO court detections."""

    samples: tuple[CourtDetrDetectionBatch, ...]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> CourtDetrDetectionBatch:
        return self.samples[index]

    @property
    def num_instances(self) -> Tensor:
        device = self.samples[0].scores.device if self.samples else None
        return torch.tensor(
            [sample.num_instances for sample in self.samples],
            dtype=torch.long,
            device=device,
        )


def _image_size(image_size: int | tuple[int, int]) -> tuple[float, float]:
    if isinstance(image_size, bool):
        raise TypeError("image_size must be a positive integer or (height,width).")
    if isinstance(image_size, int):
        if image_size <= 0:
            raise ValueError("image_size must be positive.")
        return float(image_size), float(image_size)
    if len(image_size) != 2 or any(
        isinstance(value, bool) or int(value) != value or value <= 0
        for value in image_size
    ):
        raise ValueError("image_size must contain two positive integers.")
    return float(image_size[0]), float(image_size[1])


def decode_detr_courts(
    pred_logits: Tensor,
    pred_boxes: Tensor,
    pred_court_boxes: Tensor,
    *,
    image_size: int | tuple[int, int],
    class_index: int = 0,
    threshold: float = 0.5,
    top_k: int | None = None,
) -> CourtDetrDetections:
    """Decode multiple court queries to OBB corners and similarity pose.

    ``pred_logits`` are unnormalized sigmoid focal logits ``[B,Q,C]``;
    ``pred_boxes`` are normalized DINO AABBs ``[B,Q,4]``; and
    ``pred_court_boxes`` are raw ``[long_logit, axis_x, axis_y]`` values.
    Long/short sides are restored using the isotropic runtime reference
    ``max(height,width)`` and scale is ``long_side_px / 23.77m``.  No NMS or
    cross-query clustering is applied.
    """

    if pred_logits.ndim != 3 or pred_logits.shape[-1] <= 0:
        raise ValueError("pred_logits must have shape [B,Q,C] with C positive.")
    if pred_boxes.shape != (*pred_logits.shape[:2], 4):
        raise ValueError("pred_boxes must have shape [B,Q,4].")
    if pred_court_boxes.shape != (*pred_logits.shape[:2], 3):
        raise ValueError("pred_court_boxes must have shape [B,Q,3].")
    for name, value in (
        ("pred_logits", pred_logits),
        ("pred_boxes", pred_boxes),
        ("pred_court_boxes", pred_court_boxes),
    ):
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point.")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values.")
        if value.device != pred_logits.device:
            raise ValueError("All DINO predictions must share a device.")
    if bool(((pred_boxes < 0.0) | (pred_boxes > 1.0)).any()):
        raise ValueError("pred_boxes must be normalized to [0,1].")
    if isinstance(class_index, bool) or not isinstance(class_index, int):
        raise TypeError("class_index must be an integer.")
    if class_index < 0 or class_index >= pred_logits.shape[-1]:
        raise ValueError("class_index is outside pred_logits classes.")
    if not math.isfinite(float(threshold)) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0,1].")
    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0
    ):
        raise ValueError("top_k must be a positive integer or None.")
    height, width = _image_size(image_size)
    isotropic_size = max(height, width)

    scores = pred_logits[..., class_index].sigmoid()
    decoded_court = decode_raw_court_boxes(pred_court_boxes)
    image_scale = pred_boxes.new_tensor((width, height))
    samples: list[CourtDetrDetectionBatch] = []
    for batch_index in range(pred_logits.shape[0]):
        query_indices = torch.nonzero(
            scores[batch_index] >= threshold, as_tuple=False
        ).flatten()
        if query_indices.numel() > 0:
            order = torch.argsort(scores[batch_index, query_indices], descending=True)
            query_indices = query_indices[order]
            if top_k is not None:
                query_indices = query_indices[:top_k]
        selected_scores = scores[batch_index, query_indices]
        selected_aabb = pred_boxes[batch_index, query_indices]
        selected_court = decoded_court[batch_index, query_indices]
        centers_px = selected_aabb[:, :2] * image_scale
        long_sides_px = selected_court[:, 0] * isotropic_size
        short_sides_px = long_sides_px * COURT_SHORT_TO_LONG_RATIO
        axial_vectors = selected_court[:, 1:]
        corners_px = oriented_box_corners(
            centers_px,
            long_sides_px,
            axial_vectors,
        )
        rotation_rad = court_rotation_from_axial_vector(axial_vectors)
        scale_px_per_metre = long_sides_px / COURT_LENGTH_M
        samples.append(
            CourtDetrDetectionBatch(
                scores=selected_scores,
                query_indices=query_indices,
                aabb_cxcywh_normalized=selected_aabb,
                centers_px=centers_px,
                long_sides_px=long_sides_px,
                short_sides_px=short_sides_px,
                axial_vectors=axial_vectors,
                corners_px=corners_px,
                rotation_rad=rotation_rad,
                scale_px_per_metre=scale_px_per_metre,
            )
        )
    return CourtDetrDetections(tuple(samples))


__all__ = [
    "CourtDetrDetectionBatch",
    "CourtDetrDetections",
    "decode_detr_courts",
]
