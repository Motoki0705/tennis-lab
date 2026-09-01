"""Shared task-aware model contracts."""

from src.tasks.base.models.track_query_reference import (
    REFERENCE_SELECTOR_ROPE_CONTRACT,
    ROLE_ROPE_CONTRACT,
    ReferenceContextMaskError,
    ReferenceSelectorMode,
    TrackQueryReferenceModelError,
    TrackQueryRopeContract,
    TrackQueryRopeDimensionError,
    build_compressed_track_query_spatial_coordinates,
    resolve_reference_selector_mode,
    resolve_track_query_rope_contract,
    validate_reference_context_mask,
    validate_track_query_rope_dimensions,
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
