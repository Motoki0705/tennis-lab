"""Fail-closed BLCS checkpoint metadata contract."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, open_dict

from src.tasks.base.generate_dataset import (
    CourtKeypointContract,
    CourtKeypointContractMismatchError,
    MissingCourtKeypointMetadataError,
    resolve_court_keypoint_contract,
)
from src.tasks.base.model_io import (
    TrackQueryReferenceContract,
    TrackQueryReferenceContractMismatchError,
    extract_track_query_reference_contract_metadata,
    resolve_model_artifact_court_keypoint_contract,
    validate_checkpoint_track_query_reference_contract,
    validate_model_artifact_court_keypoint_contract,
    write_checkpoint_track_query_reference_contract,
)
from src.tasks.base.models import (
    REFERENCE_SELECTOR_ROPE_CONTRACT,
    resolve_reference_selector_mode,
    resolve_track_query_rope_contract,
)
from src.tasks.blcs.configuration import parse_court_keypoint_contract
from src.utils.schema.court_normalization import load_and_validate_checkpoint


@dataclass(frozen=True, slots=True)
class BLCSCheckpointRuntime:
    """Checkpoint composition and its validated CourtKP20 contract."""

    config: Any
    court_keypoint_contract: CourtKeypointContract
    track_query_reference_contract: TrackQueryReferenceContract | None
    legacy_metadata_free: bool


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    checkpoint: Mapping[str, Any] = load_and_validate_checkpoint(path)
    return checkpoint


def _checkpoint_config(checkpoint: Mapping[str, Any]) -> Any:
    hyper_parameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyper_parameters, Mapping) or "config" not in hyper_parameters:
        raise RuntimeError(
            "BLCS checkpoint is incompatible: hyper_parameters.config is required "
            "to compose its typed model I/O contract."
        )
    return hyper_parameters["config"]


def _qualify_metadata_free_non_tracking_config(config: Any) -> Any:
    """Add explicit physical-v1 config only for non-tracking checkpoints."""
    copied = deepcopy(config)
    if isinstance(copied, DictConfig):
        with open_dict(copied):
            if "court_keypoints" not in copied:
                copied["court_keypoints"] = {"selector": "physical_v1"}
        return copied
    if isinstance(copied, Mapping):
        result = dict(copied)
        result.setdefault("court_keypoints", {"selector": "physical_v1"})
        return result
    raise RuntimeError(
        "BLCS checkpoint hyper_parameters.config must be a mapping-like config."
    )


def _has_court_keypoint_section(config: Any) -> bool:
    return isinstance(config, (DictConfig, Mapping)) and "court_keypoints" in config


def resolve_config_track_query_reference_contract(
    config: Any,
) -> TrackQueryReferenceContract | None:
    """Resolve exact BLCS track-query semantics from a saved/runtime config."""
    if not isinstance(config, (DictConfig, Mapping)):
        raise RuntimeError("BLCS checkpoint config must be a mapping.")
    raw_model = config.get("model")
    if raw_model is None:
        return None
    if not isinstance(raw_model, (DictConfig, Mapping)):
        raise RuntimeError("BLCS checkpoint config.model must be a mapping.")
    name = raw_model.get("name")
    if name == "blcs_track_query":
        return TrackQueryReferenceContract.physical_v1()
    if name in {None, "blcs", "blcs_multiview_axial"}:
        return None
    if name != "blcs_track_query_reference":
        raise RuntimeError(f"Unsupported BLCS checkpoint model name {name!r}.")
    required = {
        "target_frame_contract",
        "track_query_rope_contract",
        "reference_selector_mode",
    }
    missing = sorted(required - set(raw_model))
    if missing:
        raise RuntimeError(
            "BLCS reference-v2 checkpoint config.model is missing exact semantic "
            f"field(s): {missing!r}."
        )
    values = {key: raw_model[key] for key in required}
    if any(type(value) is not str for value in values.values()):
        raise RuntimeError(
            "BLCS reference-v2 checkpoint semantic fields must be strings."
        )
    selector = resolve_reference_selector_mode(
        cast("str", values["reference_selector_mode"])
    )
    rope = resolve_track_query_rope_contract(
        cast("str", values["track_query_rope_contract"])
    )
    result = TrackQueryReferenceContract.reference_v2(selector)
    if (
        rope is not REFERENCE_SELECTOR_ROPE_CONTRACT
        or values["target_frame_contract"] != result.target_frame_contract
    ):
        raise RuntimeError(
            "BLCS reference-v2 checkpoint config has incompatible target-frame "
            "or track-query RoPE semantics."
        )
    return result


def resolve_blcs_track_query_reference_contract(
    config: Any,
) -> TrackQueryReferenceContract:
    """Resolve one BLCS track-query model and its exact Court/target/RoPE tuple."""
    contract = resolve_config_track_query_reference_contract(config)
    if contract is None:
        raise ValueError("BLCS config is not a track-query architecture.")
    court_keypoints = parse_court_keypoint_contract(config)
    if (
        court_keypoints.contract_id != contract.court_keypoint_contract
        or court_keypoints.target_frame_id != contract.target_frame_contract
    ):
        raise CourtKeypointContractMismatchError(
            "BLCS track-query CourtKP20 marker does not match its model/target/"
            "RoPE/selector contract."
        )
    return contract


def write_blcs_checkpoint_track_query_reference(
    checkpoint: MutableMapping[str, object],
    contract: TrackQueryReferenceContract,
) -> None:
    """Persist exact BLCS court/target/RoPE/selector checkpoint metadata."""
    write_checkpoint_track_query_reference_contract(
        checkpoint,
        contract,
        location="BLCS checkpoint",
    )


def validate_blcs_checkpoint_track_query_reference(
    checkpoint: Mapping[str, object],
    contract: TrackQueryReferenceContract,
) -> None:
    """Reject BLCS semantic mismatch before restoring any model state."""
    validate_checkpoint_track_query_reference_contract(
        checkpoint,
        contract,
        location="BLCS checkpoint",
    )


def load_checkpoint_config(path: Path) -> Any:
    """Load the explicit configuration required to compose a BLCS checkpoint."""
    return _checkpoint_config(_load_checkpoint(path))


def load_checkpoint_runtime(
    path: Path,
    *,
    runtime_court_keypoints: CourtKeypointContract | str | None = None,
    runtime_track_query_reference: TrackQueryReferenceContract | None = None,
) -> BLCSCheckpointRuntime:
    """Restore exact CourtKP semantics or require explicit physical-v1 legacy use."""
    checkpoint = _load_checkpoint(path)
    requested = (
        None
        if runtime_court_keypoints is None
        else (
            runtime_court_keypoints
            if isinstance(runtime_court_keypoints, CourtKeypointContract)
            else resolve_court_keypoint_contract(runtime_court_keypoints)
        )
    )
    if requested is None:
        checkpoint_contract = resolve_model_artifact_court_keypoint_contract(
            checkpoint,
            location=str(path),
        )
    else:
        checkpoint_contract = validate_model_artifact_court_keypoint_contract(
            checkpoint,
            requested,
            location=str(path),
        )

    config = _checkpoint_config(checkpoint)
    config_track_query_contract = resolve_config_track_query_reference_contract(config)
    if checkpoint_contract.legacy_metadata_free:
        if config_track_query_contract is not None:
            raise MissingCourtKeypointMetadataError(
                f"{path}: canonical BLCS track-query checkpoints require explicit "
                "CourtKP20 metadata; metadata-free pre-promotion checkpoints are "
                "incompatible."
            )
        config = _qualify_metadata_free_non_tracking_config(config)
    elif not _has_court_keypoint_section(config):
        raise RuntimeError(
            f"{path}: metadata-bearing BLCS checkpoint config must include "
            "court_keypoints."
        )
    config_contract = parse_court_keypoint_contract(config)
    if config_contract != checkpoint_contract.contract:
        raise CourtKeypointContractMismatchError(
            f"{path}: checkpoint config CourtKP contract "
            f"{config_contract.contract_id!r} does not match checkpoint/runtime "
            f"{checkpoint_contract.contract.contract_id!r}."
        )
    stored_track_query_metadata = extract_track_query_reference_contract_metadata(
        checkpoint,
        location=str(path),
    )
    if config_track_query_contract is None:
        if runtime_track_query_reference is not None:
            raise TrackQueryReferenceContractMismatchError(
                f"{path}: non-track-query BLCS config cannot use a track-query "
                "runtime contract."
            )
        if stored_track_query_metadata is not None:
            raise TrackQueryReferenceContractMismatchError(
                f"{path}: non-track-query BLCS checkpoint must not contain "
                "track-query semantic metadata."
            )
        resolved_track_query_contract = None
    else:
        if (
            config_contract.contract_id
            != config_track_query_contract.court_keypoint_contract
            or config_contract.target_frame_id
            != config_track_query_contract.target_frame_contract
        ):
            raise TrackQueryReferenceContractMismatchError(
                f"{path}: checkpoint config CourtKP20/target-frame semantics do "
                "not match its track-query RoPE/selector model type."
            )
        requested_track_query_contract = (
            config_track_query_contract
            if runtime_track_query_reference is None
            else runtime_track_query_reference
        )
        if requested_track_query_contract != config_track_query_contract:
            raise TrackQueryReferenceContractMismatchError(
                f"{path}: checkpoint config track-query contract "
                f"{config_track_query_contract!r} does not match runtime "
                f"{requested_track_query_contract!r}."
            )
        compatibility = validate_checkpoint_track_query_reference_contract(
            checkpoint,
            requested_track_query_contract,
            location=str(path),
        )
        resolved_track_query_contract = compatibility.contract
    return BLCSCheckpointRuntime(
        config=config,
        court_keypoint_contract=checkpoint_contract.contract,
        track_query_reference_contract=resolved_track_query_contract,
        legacy_metadata_free=checkpoint_contract.legacy_metadata_free,
    )


def validate_checkpoint_path(
    path: Path,
    runtime_court_keypoints: CourtKeypointContract | str,
    runtime_track_query_reference: TrackQueryReferenceContract | None = None,
) -> None:
    """Validate a resume/init checkpoint CourtKP contract before model loading."""
    load_checkpoint_runtime(
        path,
        runtime_court_keypoints=runtime_court_keypoints,
        runtime_track_query_reference=runtime_track_query_reference,
    )


__all__ = [
    "BLCSCheckpointRuntime",
    "load_checkpoint_config",
    "load_checkpoint_runtime",
    "resolve_blcs_track_query_reference_contract",
    "resolve_config_track_query_reference_contract",
    "validate_blcs_checkpoint_track_query_reference",
    "validate_checkpoint_path",
    "write_blcs_checkpoint_track_query_reference",
]
