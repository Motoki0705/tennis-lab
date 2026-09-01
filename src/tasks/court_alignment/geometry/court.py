"""Versioned canonical ground-court geometry used by court alignment.

The alignment model consumes a ground-UV raster and predicts the following
explicit, task-local KP14 semantic order.  This is deliberately a separate
contract from image-space Court Detection labels: no camera-side or image
augmentation semantics are inferred here.

Coordinates are metric metres in a right-handed ground plane.  ``x`` is the
left/right axis and ``y`` points towards the far baseline.  Pixel coordinates
used by the renderer are always ``(x, y)`` while tensors are ``[H, W]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from src.utils.schema.court import (
    CAMERA_VIEW_HALF_TURN_INDEX,
    CENTER_MARK_LENGTH,
    COURT_KP_NAMES,
    COURT_SKELETON,
    DOUBLES_WIDTH,
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    HALF_SINGLES_WIDTH,
    SERVICE_LINE_DISTANCE,
    SINGLES_WIDTH,
    STANDARD_COURT_CONFIG,
    court_keypoints_3d,
)
from src.utils.schema.court import (
    NUM_GROUND_COURT_KP as AUTHORITY_NUM_GROUND_COURT_KP,
)

# This versioned alias is the public alignment contract.  The names and
# semantic order are sourced from the repository's single court authority;
# this module does not define a second image-space label order.
GROUND_COURT_KP14_SCHEMA = "ground_court_kp14_v1"
GROUND_COURT_KP14_NAMES: tuple[str, ...] = COURT_KP_NAMES[:AUTHORITY_NUM_GROUND_COURT_KP]

GROUND_COURT_KP14_COUNT = AUTHORITY_NUM_GROUND_COURT_KP
# The four doubles-court corners form the physical footprint used when
# sampling multiple synthetic courts.  The order is cyclic around the
# rectangle (far-left, far-right, near-right, near-left), rather than the
# semantic KP order's far/near interleaving.
GROUND_COURT_DOUBLES_FOOTPRINT_INDEX: tuple[int, ...] = (0, 1, 3, 2)
# Concise aliases used by downstream training code.  They refer to this
# versioned ground contract, not to the image-space schema of Court Detection.
GROUND_COURT_KP_NAMES = GROUND_COURT_KP14_NAMES
NUM_GROUND_COURT_KP = GROUND_COURT_KP14_COUNT

# ITF regulation dimensions in metres.
COURT_LENGTH_M = 2.0 * HALF_LENGTH
HALF_LENGTH_M = HALF_LENGTH
SINGLES_WIDTH_M = SINGLES_WIDTH
HALF_SINGLES_WIDTH_M = HALF_SINGLES_WIDTH
DOUBLES_WIDTH_M = DOUBLES_WIDTH
HALF_DOUBLES_WIDTH_M = HALF_DOUBLES_WIDTH
SERVICE_LINE_DISTANCE_M = SERVICE_LINE_DISTANCE
CENTER_MARK_LENGTH_M = CENTER_MARK_LENGTH

# Ground line connectivity.  The final two segments are center marks and are
# represented by explicit points in ``canonical_court_line_segments``.
GROUND_COURT_LINE_EDGES: tuple[tuple[int, int], ...] = tuple(
    (first, second)
    for first, second in COURT_SKELETON
    if first < GROUND_COURT_KP14_COUNT and second < GROUND_COURT_KP14_COUNT
)
GROUND_COURT_HALF_TURN_INDEX: tuple[int, ...] = CAMERA_VIEW_HALF_TURN_INDEX
GROUND_COURT_KP14_HALF_TURN_INDEX = GROUND_COURT_HALF_TURN_INDEX


@dataclass(frozen=True, slots=True)
class GroundCourtInstance:
    """One court's 2-D similarity transform in output pixel coordinates.

    ``center_xy_px`` is the transformed court origin (the net centre),
    ``rotation_rad`` rotates canonical metric points counter-clockwise, and
    ``scale_px_per_metre`` is a positive uniform scale.  ``instance_id`` is a
    fixed integer so padded dataset batches can carry identity without string
    collation or implicit ordering.
    """

    instance_id: int
    center_xy_px: tuple[float, float]
    rotation_rad: float
    scale_px_per_metre: float

    def __post_init__(self) -> None:
        if isinstance(self.instance_id, bool) or not isinstance(self.instance_id, int):
            raise TypeError("Ground court instance_id must be an integer.")
        if len(self.center_xy_px) != 2:
            raise ValueError("Ground court center_xy_px must have two coordinates.")
        if not all(math.isfinite(float(value)) for value in self.center_xy_px):
            raise ValueError("Ground court center_xy_px must be finite.")
        if not math.isfinite(float(self.rotation_rad)):
            raise ValueError("Ground court rotation_rad must be finite.")
        if not math.isfinite(float(self.scale_px_per_metre)) or self.scale_px_per_metre <= 0.0:
            raise ValueError("Ground court scale_px_per_metre must be finite and positive.")


def canonical_court_keypoints(
    *, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None
) -> Tensor:
    """Return canonical metric ground KP14 as ``[14, 2]`` in the task order."""

    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("Canonical court keypoints require a floating-point dtype.")
    return court_keypoints_3d(STANDARD_COURT_CONFIG)[:GROUND_COURT_KP14_COUNT, :2].to(
        dtype=dtype, device=device
    )


def canonical_court_line_segments(
    *, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None
) -> Tensor:
    """Return all ground line segments, including the two baseline marks.

    The result has shape ``[15, 2, 2]`` and metric ``(x, y)`` coordinates.
    Baselines are represented once through the KP edge, matching the existing
    ITF CourtKP14 skeleton.
    """

    points = canonical_court_keypoints(dtype=dtype, device=device)
    edges = points.new_tensor(GROUND_COURT_LINE_EDGES, dtype=torch.long)
    segments = points.index_select(0, edges.reshape(-1)).reshape(-1, 2, 2)
    marks = points.new_tensor(
        (
            ((0.0, HALF_LENGTH_M), (0.0, HALF_LENGTH_M - CENTER_MARK_LENGTH_M)),
            ((0.0, -HALF_LENGTH_M), (0.0, -HALF_LENGTH_M + CENTER_MARK_LENGTH_M)),
        )
    )
    return torch.cat((segments, marks), dim=0)


def court_keypoints_for_instance(instance: GroundCourtInstance) -> Tensor:
    """Transform canonical KP14 into pixel ``(x, y)`` coordinates."""

    canonical = canonical_court_keypoints(dtype=torch.float32)
    angle = float(instance.rotation_rad)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = canonical.new_tensor(((cosine, -sine), (sine, cosine)))
    transformed = canonical @ rotation.T
    transformed = transformed * float(instance.scale_px_per_metre)
    return transformed + transformed.new_tensor(instance.center_xy_px)


def court_doubles_footprint_for_instance(instance: GroundCourtInstance) -> Tensor:
    """Return the transformed doubles-court footprint as ``[4,2]`` pixels."""

    keypoints = court_keypoints_for_instance(instance)
    indices = torch.tensor(
        GROUND_COURT_DOUBLES_FOOTPRINT_INDEX,
        dtype=torch.long,
        device=keypoints.device,
    )
    return keypoints.index_select(0, indices)


def _validate_footprint(footprint: Tensor, *, name: str) -> Tensor:
    if footprint.shape != (4, 2):
        raise ValueError(f"{name} must have shape [4,2].")
    if not footprint.is_floating_point():
        raise TypeError(f"{name} must be floating point.")
    if not bool(torch.isfinite(footprint).all()):
        raise ValueError(f"{name} must contain finite values.")
    return footprint


def doubles_footprints_overlap(
    first: Tensor,
    second: Tensor,
    *,
    tolerance_px: float = 0.0,
) -> bool:
    """Return whether two convex doubles footprints overlap materially.

    The four vertices must be supplied in cyclic order.  A separating-axis
    test avoids geometry dependencies and treats touching, as well as
    penetration no greater than ``tolerance_px`` on at least one axis, as
    non-overlap.  Thus only positive-area overlap beyond the explicit pixel
    tolerance is rejected by the sampler.
    """

    first = _validate_footprint(first, name="first footprint")
    second = _validate_footprint(second, name="second footprint").to(
        device=first.device, dtype=first.dtype
    )
    tolerance = float(tolerance_px)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance_px must be finite and non-negative.")
    axes: list[Tensor] = []
    for footprint in (first, second):
        edges = torch.roll(footprint, shifts=-1, dims=0) - footprint
        # Either orientation of a perpendicular is valid for SAT.  Normalize
        # so tolerance remains expressed in pixels on every axis.
        normals = torch.stack((-edges[:, 1], edges[:, 0]), dim=1)
        lengths = torch.linalg.vector_norm(normals, dim=1, keepdim=True)
        axes.extend(normals / lengths.clamp_min(torch.finfo(first.dtype).eps))
    for axis in axes:
        first_projection = first @ axis
        second_projection = second @ axis
        first_min, first_max = first_projection.aminmax()
        second_min, second_max = second_projection.aminmax()
        overlap_depth = torch.minimum(first_max, second_max) - torch.maximum(
            first_min, second_min
        )
        if float(overlap_depth) <= tolerance:
            return False
    return True


def court_line_segments_for_instance(instance: GroundCourtInstance) -> Tensor:
    """Transform canonical line segments into output pixel coordinates."""

    canonical = canonical_court_line_segments(dtype=torch.float32)
    angle = float(instance.rotation_rad)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = canonical.new_tensor(((cosine, -sine), (sine, cosine)))
    transformed = canonical @ rotation.T
    transformed = transformed * float(instance.scale_px_per_metre)
    return transformed + transformed.new_tensor(instance.center_xy_px)


__all__ = [
    "CENTER_MARK_LENGTH_M",
    "COURT_LENGTH_M",
    "DOUBLES_WIDTH_M",
    "GROUND_COURT_KP14_COUNT",
    "GROUND_COURT_DOUBLES_FOOTPRINT_INDEX",
    "GROUND_COURT_KP14_HALF_TURN_INDEX",
    "GROUND_COURT_KP14_NAMES",
    "GROUND_COURT_KP14_SCHEMA",
    "GROUND_COURT_KP_NAMES",
    "GROUND_COURT_LINE_EDGES",
    "GROUND_COURT_HALF_TURN_INDEX",
    "HALF_DOUBLES_WIDTH_M",
    "HALF_LENGTH_M",
    "HALF_SINGLES_WIDTH_M",
    "NUM_GROUND_COURT_KP",
    "SERVICE_LINE_DISTANCE_M",
    "SINGLES_WIDTH_M",
    "GroundCourtInstance",
    "canonical_court_keypoints",
    "canonical_court_line_segments",
    "court_keypoints_for_instance",
    "court_doubles_footprint_for_instance",
    "court_line_segments_for_instance",
    "doubles_footprints_overlap",
]
