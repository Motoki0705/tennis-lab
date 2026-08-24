"""PLCS checkpoint binding for versioned track-query semantics."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping

from src.tasks.base.generate_dataset import CourtKeypointContract
from src.tasks.base.model_io import (
    TrackQueryReferenceContract,
    validate_checkpoint_track_query_reference_contract,
    write_checkpoint_track_query_reference_contract,
)
from src.tasks.base.models import resolve_reference_selector_mode
from src.tasks.plcs.configuration import PLCSModelConfig

_LEGACY_MODEL_NAMES = frozenset(
    {"plcs_track_query", "plcs_track_query_ablation"}
)
_REFERENCE_MODEL_NAMES = frozenset(
    {
        "plcs_track_query_reference",
        "plcs_track_query_reference_ablation",
    }
)


def resolve_plcs_track_query_reference_contract(
    model: PLCSModelConfig,
    court_keypoints: CourtKeypointContract,
) -> TrackQueryReferenceContract:
    """Resolve exact PLCS model type and independent semantic markers."""
    if model.name in _LEGACY_MODEL_NAMES:
        contract = TrackQueryReferenceContract.legacy_v1()
    elif model.name in _REFERENCE_MODEL_NAMES:
        contract = TrackQueryReferenceContract.reference_v2(
            resolve_reference_selector_mode(
                model.string("reference_selector_mode")
            )
        )
        if model.string("target_frame_contract") != (
            contract.target_frame_contract
        ):
            raise ValueError(
                "PLCS model target-frame marker does not match reference-v2."
            )
        if model.string("track_query_rope_contract") != (
            contract.track_query_rope_contract.value
        ):
            raise ValueError(
                "PLCS model RoPE marker does not match reference-v2."
            )
    else:
        raise ValueError(
            f"PLCS model {model.name!r} is not a track-query architecture."
        )
    if court_keypoints.contract_id != contract.court_keypoint_contract:
        raise ValueError(
            "PLCS track-query CourtKP20 marker does not match its model/target/"
            "RoPE/selector contract."
        )
    return contract


def write_plcs_checkpoint_track_query_reference(
    checkpoint: MutableMapping[str, object],
    contract: TrackQueryReferenceContract,
) -> None:
    """Persist independent PLCS court/target/RoPE/selector markers."""
    write_checkpoint_track_query_reference_contract(
        checkpoint,
        contract,
        location="PLCS checkpoint",
    )


def validate_plcs_checkpoint_track_query_reference(
    checkpoint: Mapping[str, object],
    contract: TrackQueryReferenceContract,
) -> None:
    """Reject semantic mismatch before PLCS model state restoration."""
    validate_checkpoint_track_query_reference_contract(
        checkpoint,
        contract,
        explicit_legacy_v1=(contract == TrackQueryReferenceContract.legacy_v1()),
        location="PLCS checkpoint",
    )


__all__ = [
    "resolve_plcs_track_query_reference_contract",
    "validate_plcs_checkpoint_track_query_reference",
    "write_plcs_checkpoint_track_query_reference",
]
