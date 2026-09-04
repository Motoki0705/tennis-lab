"""Oriented-box geometry for DINO court-alignment targets.

All pixel coordinates use ``(x, y)`` order.  A court orientation is axial:
the long-edge directions ``phi`` and ``phi + pi`` describe the same box.  It
is therefore stored as the unit vector ``(cos(2*phi), sin(2*phi))``.  The
reported court rotation is the rotation of the canonical court whose long
axis points along positive ``y``; it is ``(phi - pi/2) mod pi``.

The DINO box head keeps its standard normalized ``(cx, cy, width, height)``
axis-aligned box.  The additional raw court head has exactly three values:
``(long_side_logit, axial_x_raw, axial_y_raw)``.  Decoding applies sigmoid to
the first value and L2 normalization to the final two values.  Target
``court_boxes`` contain five decoded values:
``(cx_normalized, cy_normalized, long_side_normalized, axial_x, axial_y)``.
Long/short-side normalization uses the isotropic runtime reference
``max(height, width)``.  This differs deliberately from DINO AABB center/size
normalization, which uses width for ``x`` and height for ``y``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from src.tasks.court_alignment.geometry.court import (
    COURT_LENGTH_M,
    DOUBLES_WIDTH_M,
    GROUND_COURT_DOUBLES_FOOTPRINT_INDEX,
)

COURT_SHORT_TO_LONG_RATIO = DOUBLES_WIDTH_M / COURT_LENGTH_M


def _image_size(image_size: int | tuple[int, int]) -> tuple[float, float]:
    """Return ``(height, width)`` after strict positive-size validation."""

    if isinstance(image_size, bool):
        raise TypeError("image_size must be a positive integer or (height,width).")
    if isinstance(image_size, int):
        if image_size <= 0:
            raise ValueError("image_size must be positive.")
        return float(image_size), float(image_size)
    if len(image_size) != 2:
        raise ValueError("image_size must be a positive integer or (height,width).")
    height, width = image_size
    if (
        isinstance(height, bool)
        or isinstance(width, bool)
        or int(height) != height
        or int(width) != width
        or height <= 0
        or width <= 0
    ):
        raise ValueError("image_size must contain two positive integers.")
    return float(height), float(width)


def decode_raw_court_boxes(raw_court_boxes: Tensor, *, eps: float = 1.0e-8) -> Tensor:
    """Decode raw court-head values to ``(long_normalized, axial_x, axial_y)``.

    The zero raw axial vector is deterministically mapped to ``(1, 0)`` so
    startup inference and empty-target training remain finite.  Everywhere
    else the returned axial vector has unit L2 norm and preserves gradients.
    """

    if raw_court_boxes.ndim < 1 or raw_court_boxes.shape[-1] != 3:
        raise ValueError("raw_court_boxes must have final dimension three.")
    if not raw_court_boxes.is_floating_point():
        raise TypeError("raw_court_boxes must be floating point.")
    if not bool(torch.isfinite(raw_court_boxes).all()):
        raise ValueError("raw_court_boxes must contain only finite values.")
    epsilon = float(eps)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("eps must be finite and positive.")

    long_side = raw_court_boxes[..., :1].sigmoid()
    raw_axis = raw_court_boxes[..., 1:]
    norm = torch.linalg.vector_norm(raw_axis, dim=-1, keepdim=True)
    normalized = raw_axis / norm.clamp_min(epsilon)
    fallback = torch.zeros_like(raw_axis)
    fallback[..., 0] = 1.0
    axis = torch.where(norm > epsilon, normalized, fallback)
    return torch.cat((long_side, axis), dim=-1)


def court_rotation_from_axial_vector(axial_vector: Tensor) -> Tensor:
    """Return canonical-court rotation in ``[0, pi)`` from a long-edge axis."""

    if axial_vector.ndim < 1 or axial_vector.shape[-1] != 2:
        raise ValueError("axial_vector must have final dimension two.")
    if not axial_vector.is_floating_point():
        raise TypeError("axial_vector must be floating point.")
    if not bool(torch.isfinite(axial_vector).all()):
        raise ValueError("axial_vector must contain only finite values.")
    norm = torch.linalg.vector_norm(axial_vector, dim=-1)
    if bool((norm <= torch.finfo(axial_vector.dtype).eps).any()):
        raise ValueError("axial_vector must be non-zero.")
    axis = axial_vector / norm.unsqueeze(-1)
    long_edge_angle = 0.5 * torch.atan2(axis[..., 1], axis[..., 0])
    return torch.remainder(long_edge_angle - math.pi / 2.0, math.pi)


def oriented_box_corners(
    center_xy_px: Tensor,
    long_side_px: Tensor,
    axial_vector: Tensor,
    *,
    short_to_long_ratio: float = COURT_SHORT_TO_LONG_RATIO,
) -> Tensor:
    """Construct cyclic court corners ``[...,4,2]`` from axial OBB values.

    The returned order is far-left, far-right, near-right, near-left for one
    of the two equivalent long-axis signs.  Because the representation is
    axial, a sign change may cyclically swap far and near but never changes
    the rectangle or its court rotation modulo ``pi``.
    """

    if center_xy_px.ndim < 1 or center_xy_px.shape[-1] != 2:
        raise ValueError("center_xy_px must have final dimension two.")
    if axial_vector.shape != center_xy_px.shape:
        raise ValueError("axial_vector must have the same shape as center_xy_px.")
    if long_side_px.shape != center_xy_px.shape[:-1]:
        raise ValueError("long_side_px must match the center leading dimensions.")
    for name, value in (
        ("center_xy_px", center_xy_px),
        ("long_side_px", long_side_px),
        ("axial_vector", axial_vector),
    ):
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point.")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values.")
    if bool((long_side_px <= 0.0).any()):
        raise ValueError("long_side_px must be positive.")
    ratio = float(short_to_long_ratio)
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("short_to_long_ratio must be finite and positive.")
    axis_norm = torch.linalg.vector_norm(axial_vector, dim=-1, keepdim=True)
    if bool((axis_norm <= torch.finfo(axial_vector.dtype).eps).any()):
        raise ValueError("axial_vector must be non-zero.")
    axis = axial_vector / axis_norm
    phi = 0.5 * torch.atan2(axis[..., 1], axis[..., 0])
    long_direction = torch.stack((phi.cos(), phi.sin()), dim=-1)
    short_direction = torch.stack(
        (long_direction[..., 1], -long_direction[..., 0]), dim=-1
    )
    half_long = 0.5 * long_side_px.unsqueeze(-1)
    half_short = half_long * ratio
    long_delta = half_long * long_direction
    short_delta = half_short * short_direction
    return torch.stack(
        (
            center_xy_px + long_delta - short_delta,
            center_xy_px + long_delta + short_delta,
            center_xy_px - long_delta + short_delta,
            center_xy_px - long_delta - short_delta,
        ),
        dim=-2,
    )


def build_detr_court_targets(
    keypoints: Tensor,
    visibility: Tensor,
    *,
    image_size: int | tuple[int, int],
    class_index: int = 0,
) -> list[dict[str, Tensor]]:
    """Convert padded KP14 batches to variable-length DINO court targets.

    ``keypoints`` must be pixel ``(x,y)`` coordinates with shape
    ``[B,M,14,2]`` and ``visibility`` must be bool ``[B,M,14]``.  Only an
    instance whose doubles-court footprint indices ``(0,1,3,2)`` are all
    visible is retained.  Invisible padded instances are omitted rather than
    represented as zero-area boxes.

    Each returned mapping contains ``labels [N]``, DINO ``boxes [N,4]`` in
    normalized ``cxcywh`` form, and decoded ``court_boxes [N,5]`` containing
    normalized center, normalized physical long side, and the unit axial
    vector.  Long/short sides use the isotropic denominator
    ``max(height,width)``; the short side is fixed to
    ``long * 10.97 / 23.77`` downstream.
    """

    if keypoints.ndim != 4 or keypoints.shape[-2:] != (14, 2):
        raise ValueError("keypoints must have shape [B,M,14,2].")
    if visibility.shape != keypoints.shape[:-1]:
        raise ValueError("visibility must have shape [B,M,14].")
    if not keypoints.is_floating_point():
        raise TypeError("keypoints must be floating point.")
    if visibility.dtype != torch.bool:
        raise TypeError("visibility must have boolean dtype.")
    if keypoints.device != visibility.device:
        raise ValueError("keypoints and visibility must share a device.")
    if not bool(torch.isfinite(keypoints).all()):
        raise ValueError("keypoints must contain only finite values.")
    if (
        isinstance(class_index, bool)
        or not isinstance(class_index, int)
        or class_index < 0
    ):
        raise ValueError("class_index must be a non-negative integer.")
    height, width = _image_size(image_size)
    isotropic_size = max(height, width)

    indices = torch.tensor(
        GROUND_COURT_DOUBLES_FOOTPRINT_INDEX,
        dtype=torch.long,
        device=keypoints.device,
    )
    footprint = keypoints.index_select(2, indices)
    footprint_visible = visibility.index_select(2, indices).all(dim=-1)
    normalizer_xy = keypoints.new_tensor((width, height))
    targets: list[dict[str, Tensor]] = []
    for batch_index in range(keypoints.shape[0]):
        selected = footprint[batch_index, footprint_visible[batch_index]]
        if selected.shape[0] == 0:
            targets.append(
                {
                    "labels": torch.empty(0, dtype=torch.long, device=keypoints.device),
                    "boxes": keypoints.new_empty((0, 4)),
                    "court_boxes": keypoints.new_empty((0, 5)),
                }
            )
            continue

        far_midpoint = 0.5 * (selected[:, 0] + selected[:, 1])
        near_midpoint = 0.5 * (selected[:, 2] + selected[:, 3])
        long_vector = far_midpoint - near_midpoint
        long_side_px = torch.linalg.vector_norm(long_vector, dim=-1)
        if bool((long_side_px <= torch.finfo(keypoints.dtype).eps).any()):
            raise ValueError(
                "A fully visible court footprint has zero long-side length."
            )
        long_angle = torch.atan2(long_vector[:, 1], long_vector[:, 0])
        axial = torch.stack(
            ((2.0 * long_angle).cos(), (2.0 * long_angle).sin()), dim=-1
        )
        center_px = selected.mean(dim=1)
        minimum = selected.amin(dim=1)
        maximum = selected.amax(dim=1)
        aabb_center = 0.5 * (minimum + maximum)
        aabb_size = maximum - minimum
        if bool((aabb_size <= 0.0).any()):
            raise ValueError("A fully visible court footprint has a degenerate AABB.")
        boxes = torch.cat(
            (aabb_center / normalizer_xy, aabb_size / normalizer_xy), dim=-1
        )
        court_boxes = torch.cat(
            (
                center_px / normalizer_xy,
                (long_side_px / isotropic_size).unsqueeze(-1),
                axial,
            ),
            dim=-1,
        )
        labels = torch.full(
            (selected.shape[0],),
            class_index,
            dtype=torch.long,
            device=keypoints.device,
        )
        targets.append({"labels": labels, "boxes": boxes, "court_boxes": court_boxes})
    return targets


__all__ = [
    "COURT_SHORT_TO_LONG_RATIO",
    "build_detr_court_targets",
    "court_rotation_from_axial_vector",
    "decode_raw_court_boxes",
    "oriented_box_corners",
]
