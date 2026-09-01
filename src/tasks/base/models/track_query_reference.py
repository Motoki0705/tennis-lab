"""Task-aware spatial M-RoPE contract for reference-conditioned track queries."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

import torch
from torch import Tensor

from src.tasks.base.data.track_query_reference import (
    ReferenceViewBatchError,
    validate_reference_view_index,
)


class TrackQueryReferenceModelError(ValueError):
    """Base error for the track-query reference model contract."""


class TrackQueryRopeDimensionError(TrackQueryReferenceModelError):
    """Raised when rotary dimensions cannot represent the selected contract."""


class ReferenceContextMaskError(TrackQueryReferenceModelError):
    """Raised when a supervised frame masks its reference context view."""


class TrackQueryRopeContract(StrEnum):
    """Explicit spatial-coordinate semantics independent of weight shape."""

    TIME_CAMERA_ROLE_V1 = "time_camera_role_v1"
    TIME_CAMERA_REFERENCE_SELECTOR_V1 = "time_camera_reference_selector_v1"


class ReferenceSelectorMode(StrEnum):
    """Canonical reference-selector behavior."""

    REFERENCE = "reference"


ROLE_ROPE_CONTRACT: Final = TrackQueryRopeContract.TIME_CAMERA_ROLE_V1
REFERENCE_SELECTOR_ROPE_CONTRACT: Final = (
    TrackQueryRopeContract.TIME_CAMERA_REFERENCE_SELECTOR_V1
)


def resolve_track_query_rope_contract(value: str) -> TrackQueryRopeContract:
    """Resolve one exact marker without inspecting flags, shapes, or weights."""
    try:
        return TrackQueryRopeContract(value)
    except ValueError as error:
        raise TrackQueryReferenceModelError(
            f"Unknown track-query RoPE contract {value!r}; expected one of "
            f"{tuple(item.value for item in TrackQueryRopeContract)!r}."
        ) from error


def resolve_reference_selector_mode(value: str) -> ReferenceSelectorMode:
    """Resolve the canonical reference-selector mode without a default."""
    try:
        return ReferenceSelectorMode(value)
    except ValueError as error:
        raise TrackQueryReferenceModelError(
            f"Unknown reference selector mode {value!r}; expected one of "
            f"{tuple(item.value for item in ReferenceSelectorMode)!r}."
        ) from error


def validate_track_query_rope_dimensions(
    *,
    contract: TrackQueryRopeContract,
    rope_dim: int,
    head_dim: int,
) -> None:
    """Validate even rotary width, head fit, and v2 coverage of all three axes."""
    if not isinstance(contract, TrackQueryRopeContract):
        raise TypeError("contract must be TrackQueryRopeContract.")
    if type(rope_dim) is not int or rope_dim <= 0 or rope_dim % 2 != 0:
        raise TrackQueryRopeDimensionError(
            f"rope_dim must be a positive even int, got {rope_dim!r}."
        )
    if type(head_dim) is not int or head_dim <= 0:
        raise TrackQueryRopeDimensionError(
            f"head_dim must be a positive int, got {head_dim!r}."
        )
    if rope_dim > head_dim:
        raise TrackQueryRopeDimensionError(
            f"rope_dim ({rope_dim}) must not exceed head_dim ({head_dim})."
        )
    if contract is ROLE_ROPE_CONTRACT:
        return
    if rope_dim < 6:
        raise TrackQueryRopeDimensionError(
            "Reference-selector v2 requires rope_dim >= 6 so time, camera, and "
            f"selector each receive a rotary pair; got {rope_dim}."
        )
    allocated_axes = {pair_index % 3 for pair_index in range(rope_dim // 2)}
    if allocated_axes != {0, 1, 2}:
        raise TrackQueryRopeDimensionError(
            "Reference-selector v2 rotary allocation must cover all three axes."
        )


def _require_positive_int(value: int, *, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise TrackQueryReferenceModelError(
            f"{name} must be a positive int, got {value!r}."
        )


def _build_track_query_spatial_coordinates(
    reference_view_index: Tensor,
    *,
    num_frames: int,
    num_views: int,
    num_queries: int,
    object_tokens_per_view: int,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Build exact v2 ``(B*T, Q+V*P, 3)`` spatial coordinates.

    Query slots are first, followed by view-major object tokens.  The leading
    ``B*T`` axis is batch-major with time contiguous, matching ``flatten(0, 1)``
    in the existing BLCS/PLCS track-query models.
    """
    for name, value in (
        ("num_frames", num_frames),
        ("num_views", num_views),
        ("num_queries", num_queries),
        ("object_tokens_per_view", object_tokens_per_view),
    ):
        _require_positive_int(value, name=name)
    if not isinstance(reference_view_index, Tensor):
        raise TypeError("reference_view_index must be a torch.Tensor.")
    inferred_batch_size = (
        int(reference_view_index.shape[0]) if reference_view_index.ndim else 0
    )
    if batch_size is not None:
        _require_positive_int(batch_size, name="batch_size")
        if batch_size != inferred_batch_size:
            raise TrackQueryReferenceModelError(
                f"batch_size={batch_size} does not match reference_view_index "
                f"shape {tuple(reference_view_index.shape)}."
            )
    else:
        batch_size = inferred_batch_size
    coordinate_device = torch.device(device or reference_view_index.device)
    try:
        validate_reference_view_index(
            reference_view_index,
            batch_size=batch_size,
            num_views=num_views,
            device=coordinate_device,
        )
    except ReferenceViewBatchError as error:
        raise TrackQueryReferenceModelError(str(error)) from error

    time = torch.arange(num_frames, device=coordinate_device, dtype=torch.int64)
    queries = torch.zeros(
        batch_size,
        num_frames,
        num_queries,
        3,
        dtype=torch.int64,
        device=coordinate_device,
    )
    queries[..., 0] = time.view(1, num_frames, 1)

    objects = torch.zeros(
        batch_size,
        num_frames,
        num_views,
        object_tokens_per_view,
        3,
        dtype=torch.int64,
        device=coordinate_device,
    )
    objects[..., 0] = time.view(1, num_frames, 1, 1)
    objects[..., 1] = torch.arange(
        1,
        num_views + 1,
        dtype=torch.int64,
        device=coordinate_device,
    ).view(1, 1, num_views, 1)
    selector = torch.ones(
        batch_size,
        num_views,
        dtype=torch.int64,
        device=coordinate_device,
    )
    selector.scatter_(1, reference_view_index[:, None], 0)
    objects[..., 2] = selector[:, None, :, None]
    return torch.cat((queries, objects.flatten(2, 3)), dim=2).flatten(0, 1)


def build_compressed_track_query_spatial_coordinates(
    reference_view_index: Tensor,
    *,
    num_frames: int,
    num_views: int,
    num_queries: int,
) -> Tensor:
    """Build the layer-end compressed ``Q+V`` spatial width."""
    return _build_track_query_spatial_coordinates(
        reference_view_index,
        num_frames=num_frames,
        num_views=num_views,
        num_queries=num_queries,
        object_tokens_per_view=1,
    )


def validate_reference_context_mask(
    reference_view_index: Tensor,
    context_valid: Tensor,
    *,
    supervised_time_mask: Tensor | None = None,
) -> None:
    """Require an unmasked reference context token at every supervised time.

    ``context_valid`` is ``bool[B,V,T]`` and deliberately excludes detection
    visibility.  Empty detections and all-false visibility therefore cannot
    remove the reference camera's context token.
    """
    if not isinstance(context_valid, Tensor):
        raise TypeError("context_valid must be a torch.Tensor.")
    if context_valid.dtype != torch.bool:
        raise ReferenceContextMaskError("context_valid must have bool dtype.")
    if context_valid.ndim != 3 or any(size == 0 for size in context_valid.shape):
        raise ReferenceContextMaskError(
            "context_valid must have nonempty shape (B,V,T), got "
            f"{tuple(context_valid.shape)}."
        )
    batch_size, num_views, num_frames = context_valid.shape
    try:
        validate_reference_view_index(
            reference_view_index,
            batch_size=batch_size,
            num_views=num_views,
            device=context_valid.device,
        )
    except ReferenceViewBatchError as error:
        raise ReferenceContextMaskError(str(error)) from error
    if supervised_time_mask is None:
        supervised = context_valid.any(dim=1)
    else:
        if not isinstance(supervised_time_mask, Tensor):
            raise TypeError("supervised_time_mask must be a torch.Tensor.")
        if supervised_time_mask.dtype != torch.bool:
            raise ReferenceContextMaskError(
                "supervised_time_mask must have bool dtype."
            )
        if supervised_time_mask.shape != (batch_size, num_frames):
            raise ReferenceContextMaskError(
                f"supervised_time_mask must have shape ({batch_size}, {num_frames})."
            )
        if supervised_time_mask.device != context_valid.device:
            raise ReferenceContextMaskError(
                "supervised_time_mask must share the context mask device."
            )
        supervised = supervised_time_mask
    reference_valid = context_valid.gather(
        1,
        reference_view_index[:, None, None].expand(-1, 1, num_frames),
    ).squeeze(1)
    invalid = supervised & ~reference_valid
    if invalid.any().item():
        failing = invalid.nonzero(as_tuple=False).tolist()
        raise ReferenceContextMaskError(
            "Every supervised time must retain an unmasked reference context "
            f"token; invalid (batch,time) entries are {failing!r}."
        )


__all__ = [
    "REFERENCE_SELECTOR_ROPE_CONTRACT",
    "ROLE_ROPE_CONTRACT",
    "ReferenceContextMaskError",
    "ReferenceSelectorMode",
    "TrackQueryReferenceModelError",
    "TrackQueryRopeContract",
    "TrackQueryRopeDimensionError",
    "build_compressed_track_query_spatial_coordinates",
    "resolve_reference_selector_mode",
    "resolve_track_query_rope_contract",
    "validate_reference_context_mask",
    "validate_track_query_rope_dimensions",
]
