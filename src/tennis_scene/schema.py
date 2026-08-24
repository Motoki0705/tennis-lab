"""Canonical schema for an integrated tennis-scene reconstruction result."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.base.generate_dataset import (
    PHYSICAL_V1_SELECTOR,
    CourtKeypointContract,
    CourtKeypointContractMismatchError,
    CourtReferenceFrameProvenance,
    MissingCourtKeypointMetadataError,
    build_physical_court_provenance,
)
from src.tasks.base.model_io import (
    validate_model_artifact_court_keypoint_contract,
    write_model_artifact_court_keypoint_contract,
)

COURT_REFERENCE_PROVENANCE_KEY = "court_reference_provenance"


@dataclass
class SceneResult:
    """Result of tennis scene 3D reconstruction.

    Player-related arrays use ``(P, T, ...)`` as the canonical shape. Camera
    observations use a leading camera axis ``N``.

    ``player_position`` and ``ball_3d`` are in court coordinates: XY is the
    court plane and +Z is up. ``smpl_vertices_local`` and SMPL pose parameters
    are stored in the GVHMR/SMPL body convention; rendering root-centers the
    vertices and explicitly converts Y-up SMPL geometry to court Z-up before
    applying ``player_yaw``.

    Archive persistence is intentionally separate from this schema. Use
    :func:`src.tennis_scene.archive.save_scene_result` and
    :func:`src.tennis_scene.archive.load_scene_result`.
    """

    num_frames: int
    fps: float
    width: int
    height: int

    court_kp: NDArray[np.float32]  # (N, T, K, 2)
    court_vis: NDArray[np.float32]  # (N, T, K)

    player_position: NDArray[np.float32]  # (P, T, 3)
    player_yaw: NDArray[np.float32]  # (P, T)

    smpl_body_pose: NDArray[np.float32]  # (P, T, 63)
    smpl_global_orient: NDArray[np.float32]  # (P, T, 3)
    smpl_betas: NDArray[np.float32]  # (P, 10)
    smpl_vertices_local: NDArray[np.float32] | None = None  # (P, T, V, 3)

    ball_uv: NDArray[np.float32] | None = None  # (N, T, 2)
    ball_vis: NDArray[np.bool_] | None = None  # (N, T)
    ball_3d: NDArray[np.float32] | None = None  # (T, 3)

    human_kp_2d: NDArray[np.float32] | None = None  # (P, N, T, 17, 2)
    human_kp_vis: NDArray[np.float32] | None = None  # (P, N, T, 17)

    player_track_ids: NDArray[np.int32] | None = None
    player_kp_3d: NDArray[np.float32] | None = None  # (P, T, J, 3)

    metadata: dict[str, Any] = field(default_factory=dict)


def attach_court_keypoint_provenance(
    document: dict[str, Any],
    contract: CourtKeypointContract,
    provenance: CourtReferenceFrameProvenance,
    *,
    location: str,
) -> dict[str, object]:
    """Attach exact model-frame provenance without changing physical arrays."""
    if provenance.contract != contract:
        raise CourtKeypointContractMismatchError(
            f"{location}: reference provenance contract "
            f"{provenance.contract_id!r} does not match runtime "
            f"{contract.contract_id!r}."
        )
    result: dict[str, object] = dict(document)
    write_model_artifact_court_keypoint_contract(
        result,
        contract,
        location=location,
    )
    existing = result.get(COURT_REFERENCE_PROVENANCE_KEY)
    if existing is not None:
        parsed = CourtReferenceFrameProvenance.from_mapping(
            existing,
            location=f"{location}.{COURT_REFERENCE_PROVENANCE_KEY}",
        )
        if parsed != provenance:
            raise CourtKeypointContractMismatchError(
                f"{location}: refusing to replace conflicting Court reference "
                "provenance."
            )
    result[COURT_REFERENCE_PROVENANCE_KEY] = provenance.to_dict()
    return result


def validate_court_keypoint_provenance(
    document: dict[str, Any],
    contract: CourtKeypointContract,
    *,
    location: str,
) -> CourtReferenceFrameProvenance:
    """Validate a result document before its model-frame arrays are consumed."""
    validate_model_artifact_court_keypoint_contract(
        document,
        contract,
        location=location,
    )
    raw = document.get(COURT_REFERENCE_PROVENANCE_KEY)
    if raw is None:
        if contract.selector != PHYSICAL_V1_SELECTOR:
            raise MissingCourtKeypointMetadataError(
                f"{location}: camera_view_v2 result is missing "
                f"{COURT_REFERENCE_PROVENANCE_KEY}."
            )
        return build_physical_court_provenance()
    provenance = CourtReferenceFrameProvenance.from_mapping(
        raw,
        location=f"{location}.{COURT_REFERENCE_PROVENANCE_KEY}",
    )
    if provenance.contract != contract:
        raise CourtKeypointContractMismatchError(
            f"{location}: reference provenance contract "
            f"{provenance.contract_id!r} does not match runtime "
            f"{contract.contract_id!r}."
        )
    return provenance


def attach_scene_result_court_keypoint_provenance(
    result: SceneResult,
    contract: CourtKeypointContract,
    provenance: CourtReferenceFrameProvenance,
) -> None:
    """Record how physical SceneResult arrays were restored from model space."""
    result.metadata = attach_court_keypoint_provenance(
        result.metadata,
        contract,
        provenance,
        location="SceneResult.metadata",
    )


def validate_scene_result_court_keypoint_provenance(
    result: SceneResult,
    contract: CourtKeypointContract,
    *,
    location: str = "SceneResult",
) -> CourtReferenceFrameProvenance:
    """Validate SceneResult Court semantics and reversible model provenance."""
    return validate_court_keypoint_provenance(
        result.metadata,
        contract,
        location=f"{location}.metadata",
    )


__all__ = [
    "COURT_REFERENCE_PROVENANCE_KEY",
    "SceneResult",
    "attach_court_keypoint_provenance",
    "attach_scene_result_court_keypoint_provenance",
    "validate_court_keypoint_provenance",
    "validate_scene_result_court_keypoint_provenance",
]
