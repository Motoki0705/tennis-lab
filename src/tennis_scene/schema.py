"""Canonical schema for an integrated tennis-scene reconstruction result."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.base.data.court_coordinate_contract import (
    CourtCoordinateContractMismatchError,
    CourtCoordinateNormalizationMetadata,
    MissingCourtCoordinateMetadataError,
    extract_court_coordinate_normalization_metadata,
    inject_court_coordinate_normalization_metadata,
)
from src.utils.schema.court_normalization import CourtCoordinateNormalization

SCENE_RESULT_POSITION_UNIT = "m"


@dataclass
class SceneResult:
    """Result of tennis scene 3D reconstruction.

    Player-related arrays use ``(P, T, ...)`` as the canonical shape. Camera
    observations use a leading camera axis ``N``.

    ``player_position`` and ``ball_3d`` are physical metres in court
    coordinates: XY is the court plane and +Z is up. A normalization contract
    recorded in ``metadata`` is provenance for the task models that produced
    these values; it never changes the public metre-valued arrays.
    ``smpl_vertices_local`` and SMPL pose parameters are stored in the
    GVHMR/SMPL body convention; rendering root-centers the vertices and
    explicitly converts Y-up SMPL geometry to court Z-up before applying
    ``player_yaw``.

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

    player_position: NDArray[np.float32]  # (P, T, 3), court/world [m]
    player_yaw: NDArray[np.float32]  # (P, T)

    smpl_body_pose: NDArray[np.float32]  # (P, T, 63)
    smpl_global_orient: NDArray[np.float32]  # (P, T, 3)
    smpl_betas: NDArray[np.float32]  # (P, 10)
    smpl_vertices_local: NDArray[np.float32] | None = None  # (P, T, V, 3)

    ball_uv: NDArray[np.float32] | None = None  # (N, T, 2)
    ball_vis: NDArray[np.bool_] | None = None  # (N, T)
    ball_3d: NDArray[np.float32] | None = None  # (T, 3), court/world [m]

    human_kp_2d: NDArray[np.float32] | None = None  # (P, N, T, 17, 2)
    human_kp_vis: NDArray[np.float32] | None = None  # (P, N, T, 17)

    player_track_ids: NDArray[np.int32] | None = None
    player_kp_3d: NDArray[np.float32] | None = None  # (P, T, J, 3)

    metadata: dict[str, Any] = field(default_factory=dict)


def attach_scene_result_court_coordinate_provenance(
    result: SceneResult,
    contract: CourtCoordinateNormalization,
) -> None:
    """Attach normalization provenance without modifying SceneResult arrays."""
    result.metadata = attach_court_coordinate_provenance(
        result.metadata,
        contract,
        location="SceneResult.metadata",
    )


def attach_court_coordinate_provenance(
    document: dict[str, Any],
    contract: CourtCoordinateNormalization,
    *,
    location: str,
) -> dict[str, object]:
    """Return an artifact document with canonical normalization provenance."""
    result: dict[str, object] = inject_court_coordinate_normalization_metadata(
        document,
        contract,
        location=location,
    )
    return result


def validate_court_coordinate_provenance(
    document: dict[str, Any],
    runtime_contract: CourtCoordinateNormalization,
    *,
    location: str,
) -> CourtCoordinateNormalizationMetadata | None:
    """Validate one metre-valued tennis-scene artifact's provenance.

    Metadata-free artifacts are accepted only by an explicit v1
    runtime.  New metadata must match both the runtime version and canonical
    scale.  No version is inferred from position values or array shapes.
    """
    metadata = extract_court_coordinate_normalization_metadata(
        document,
        location=location,
    )
    if metadata is None:
        if runtime_contract.version != "v1":
            raise MissingCourtCoordinateMetadataError(
                f"{location}: normalization provenance is absent. Metadata-free "
                "tennis-scene artifacts are legacy v1 only, but runtime selected "
                f"{runtime_contract.version!r}."
            )
        return None
    expected = CourtCoordinateNormalizationMetadata.from_contract(runtime_contract)
    if metadata != expected:
        raise CourtCoordinateContractMismatchError(
            f"{location}: artifact normalization {metadata.to_dict()!r} does "
            f"not match runtime {expected.to_dict()!r}."
        )
    return metadata


def validate_scene_result_court_coordinate_provenance(
    result: SceneResult,
    runtime_contract: CourtCoordinateNormalization,
    *,
    location: str = "SceneResult",
) -> CourtCoordinateNormalizationMetadata | None:
    """Validate provenance before normalizing SceneResult metre arrays."""
    return validate_court_coordinate_provenance(
        result.metadata,
        runtime_contract,
        location=f"{location}.metadata",
    )


__all__ = [
    "SCENE_RESULT_POSITION_UNIT",
    "SceneResult",
    "attach_court_coordinate_provenance",
    "attach_scene_result_court_coordinate_provenance",
    "validate_court_coordinate_provenance",
    "validate_scene_result_court_coordinate_provenance",
]
