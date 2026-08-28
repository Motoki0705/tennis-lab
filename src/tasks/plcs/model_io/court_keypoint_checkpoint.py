"""PLCS checkpoint binding for the shared CourtKP20 semantic contract."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping

from omegaconf import DictConfig, OmegaConf

from src.tasks.base.generate_dataset import (
    PHYSICAL_V1_SELECTOR,
    CourtKeypointContract,
    CourtKeypointContractMismatchError,
    MissingCourtKeypointMetadataError,
)
from src.tasks.base.model_io import (
    resolve_model_artifact_court_keypoint_contract,
    validate_model_artifact_court_keypoint_contract,
    write_model_artifact_court_keypoint_contract,
)
from src.tasks.plcs.court_keypoint_contract import PLCSCourtKeypointRuntimeConfig


class _MissingRuntime:
    pass


_MISSING_RUNTIME = _MissingRuntime()


def write_plcs_checkpoint_court_keypoints(
    checkpoint: MutableMapping[str, object],
    contract: CourtKeypointContract,
) -> None:
    """Persist exact CourtKP20 semantics beside PLCS model state."""
    write_model_artifact_court_keypoint_contract(
        checkpoint,
        contract,
        location="PLCS checkpoint",
    )


def validate_plcs_checkpoint_court_keypoints(
    checkpoint: Mapping[str, object],
    contract: CourtKeypointContract,
) -> None:
    """Reject CourtKP20 mismatch before Lightning restores state."""
    validate_model_artifact_court_keypoint_contract(
        checkpoint,
        contract,
        location="PLCS checkpoint",
    )


def _stored_config(checkpoint: Mapping[str, object]) -> DictConfig:
    """Copy the Lightning constructor config without mutating the checkpoint."""
    hyper_parameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyper_parameters, Mapping):
        raise ValueError(
            "PLCS checkpoint is missing the required hyper_parameters mapping."
        )
    config = hyper_parameters.get("config")
    if isinstance(config, DictConfig):
        copied = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    elif isinstance(config, Mapping):
        copied = OmegaConf.create(dict(config))
    else:
        raise ValueError(
            "PLCS checkpoint hyper_parameters is missing the required config mapping."
        )
    if not isinstance(copied, DictConfig):
        raise TypeError("PLCS checkpoint config must compose to a mapping.")
    return copied


def prepare_plcs_checkpoint_court_keypoint_config(
    checkpoint: Mapping[str, object],
    config_or_runtime: DictConfig | CourtKeypointContract | None,
    runtime_contract: CourtKeypointContract | None | _MissingRuntime = _MISSING_RUNTIME,
    *,
    location: str = "PLCS checkpoint",
) -> tuple[DictConfig, CourtKeypointContract]:
    """Resolve checkpoint semantics and validate its saved selector exactly.

    Checkpoint factories pass only the optional runtime contract and let this
    boundary restore the saved Lightning config. Tests and lower-level callers
    may pass an already copied ``DictConfig`` explicitly.
    """
    if isinstance(runtime_contract, _MissingRuntime):
        if isinstance(config_or_runtime, DictConfig):
            raise TypeError(
                "A PLCS checkpoint config requires an explicit runtime contract "
                "argument (which may be None)."
            )
        config = _stored_config(checkpoint)
        effective_runtime = config_or_runtime
    else:
        if not isinstance(config_or_runtime, DictConfig):
            raise TypeError("PLCS checkpoint config must be a DictConfig.")
        config = config_or_runtime
        effective_runtime = runtime_contract
    compatibility = (
        resolve_model_artifact_court_keypoint_contract(
            checkpoint,
            explicit_legacy_runtime=None,
            location=location,
        )
        if effective_runtime is None
        else validate_model_artifact_court_keypoint_contract(
            checkpoint,
            effective_runtime,
            location=location,
        )
    )
    contract = compatibility.contract
    if "court_keypoints" not in config:
        if not compatibility.legacy_metadata_free:
            raise MissingCourtKeypointMetadataError(
                f"{location}: checkpoint metadata is versioned but its saved "
                "config has no court_keypoints section."
            )
        if contract.selector != PHYSICAL_V1_SELECTOR:
            raise MissingCourtKeypointMetadataError(
                f"{location}: only explicit physical_v1 may load legacy config."
            )
        merged = OmegaConf.merge(
            config,
            {"court_keypoints": {"selector": contract.selector}},
        )
        if not isinstance(merged, DictConfig):
            raise TypeError("PLCS checkpoint config must remain a DictConfig.")
        return merged, contract

    stored = PLCSCourtKeypointRuntimeConfig.from_config(config).contract
    if stored != contract:
        raise CourtKeypointContractMismatchError(
            f"{location}: saved config CourtKP20 contract "
            f"{stored.contract_id!r} does not match checkpoint/runtime "
            f"{contract.contract_id!r}."
        )
    return config, contract


__all__ = [
    "prepare_plcs_checkpoint_court_keypoint_config",
    "validate_plcs_checkpoint_court_keypoints",
    "write_plcs_checkpoint_court_keypoints",
]
