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

    For legacy v1, ``player_position`` / ``player_yaw`` are the PLCS placement and
    ``gvhmr_aligned_*`` fields are the alternative placement obtained by
    aligning GVHMR world motion to PLCS. Both are retained so downstream
    consumers choose the representation they need.

    V2 uses triangulated hips and calibrated incam body recovery for the primary
    placement. Local vertices are root-centered canonical posed vertices and
    global_orient is a renderer compensation rotation; raw incam parameters live
    in the body stage artifact. All reconstruction validity is explicit.

    ``player_position``, ``gvhmr_aligned_player_position`` and ``ball_3d`` are
    in court coordinates: XY is the court plane and +Z is up.
    ``smpl_vertices_local`` and SMPL pose parameters are stored in the
    GVHMR/SMPL body convention; rendering root-centers the vertices and
    explicitly converts Y-up SMPL geometry to court Z-up before applying the
    corresponding player yaw.

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

    smpl_body_pose: NDArray[np.float32] | None = None  # (P, T, 63)
    smpl_global_orient: NDArray[np.float32] | None = None  # (P, T, 3)
    smpl_betas: NDArray[np.float32] | None = None  # (P, 10)
    smpl_vertices_local: NDArray[np.float32] | None = None  # (P, T, V, 3)

    gvhmr_aligned_player_position: NDArray[np.float32] | None = (
        None  # (P, T, 3), court coordinates
    )
    gvhmr_aligned_player_yaw: NDArray[np.float32] | None = None  # (P, T)
    gvhmr_aligned_smpl_global_orient: NDArray[np.float32] | None = None  # (P, T, 3)
    gvhmr_aligned_smpl_vertices_local: NDArray[np.float32] | None = None  # (P, T, V, 3)

    ball_uv: NDArray[np.float32] | None = None  # (N, T, 2)
    ball_vis: NDArray[np.bool_] | None = None  # (N, T)
    ball_3d: NDArray[np.float32] | None = None  # (T, 3)

    human_kp_2d: NDArray[np.float32] | None = None  # (P, N, T, 17, 2)
    human_kp_vis: NDArray[np.float32] | None = None  # (P, N, T, 17)

    player_track_ids: NDArray[np.int32] | None = None
    player_kp_3d: NDArray[np.float32] | None = None  # (P, T, J, 3)

    metadata: dict[str, Any] = field(default_factory=dict)

    player_canonical_pose: NDArray[np.float32] | None = (
        None  # (P,T,17,3), root-relative court Z-up
    )

    player_observed: NDArray[np.bool_] | None = None
    player_valid: NDArray[np.bool_] | None = None
    player_heading_valid: NDArray[np.bool_] | None = None
    player_kp_3d_vis: NDArray[np.bool_] | None = None
    player_smpl_valid: NDArray[np.bool_] | None = None
    ball_3d_valid: NDArray[np.bool_] | None = None
    player_rejection_code: NDArray[np.uint8] | None = None
    player_kp_3d_rejection_code: NDArray[np.uint8] | None = None
    ball_rejection_code: NDArray[np.uint8] | None = None

    @property
    def schema_version(self) -> int:
        if not isinstance(self.metadata, dict):
            raise TypeError("Scene metadata must be a dictionary")
        value = self.metadata.get("scene_schema_version", 1)
        if type(value) is not int or value not in (1, 2):
            raise ValueError("Unsupported SceneResult schema version")
        return value


SCENE_MASK_FIELDS = (
    "player_observed", "player_valid", "player_heading_valid",
    "player_kp_3d_vis", "player_smpl_valid", "ball_3d_valid",
)
SCENE_REASON_FIELDS = (
    "player_rejection_code", "player_kp_3d_rejection_code", "ball_rejection_code",
)


def validate_scene_result_arrays(result: SceneResult) -> None:
    """V2 requires explicit reconstruction validity, including empty scenes.

    Versionless historical archives are v1 and retain their original contract.
    Validity is never inferred from zero-filled geometry or 2D visibility.
    """
    if result.schema_version == 1:
        present = [name for name in (*SCENE_MASK_FIELDS, *SCENE_REASON_FIELDS) if getattr(result, name) is not None]
        if present:
            raise ValueError(f"v1 scenes have no reconstruction validity fields: {present}")
        return
    frames = result.num_frames
    if frames < 1 or not np.isfinite(result.fps) or result.fps <= 0 or min(result.width, result.height) < 1:
        raise ValueError("Invalid v2 scene timeline/image metadata")
    if result.player_position.ndim != 3 or result.player_position.shape[1:] != (frames, 3):
        raise ValueError("Invalid v2 player position shape")
    players = result.player_position.shape[0]
    views = result.court_kp.shape[0]
    masks: dict[str, NDArray[np.bool_]] = {}
    for name in SCENE_MASK_FIELDS:
        value = getattr(result, name)
        shape: tuple[int, ...] = (frames,) if name == "ball_3d_valid" else ((players, frames, 17) if name == "player_kp_3d_vis" else (players, frames))
        if not isinstance(value, np.ndarray) or value.dtype != np.bool_ or value.shape != shape:
            raise ValueError(f"v2 {name} must be boolean {shape}")
        masks[name] = value
    for name, mask_name in zip(SCENE_REASON_FIELDS, ("player_valid", "player_kp_3d_vis", "ball_3d_valid"), strict=True):
        codes = getattr(result, name)
        if not isinstance(codes, np.ndarray) or codes.dtype != np.uint8 or codes.shape != masks[mask_name].shape:
            raise ValueError(f"v2 {name} must be uint8 with the associated validity shape")
        if not np.array_equal(codes == 0, masks[mask_name]):
            raise ValueError(f"v2 {name}: zero reason must mean valid")
    shapes = {
        "court_kp": (views, frames, 14, 2), "court_vis": (views, frames, 14),
        "player_yaw": (players, frames), "player_kp_3d": (players, frames, 17, 3),
        "human_kp_2d": (players, views, frames, 17, 2), "human_kp_vis": (players, views, frames, 17),
        "ball_uv": (views, frames, 2), "ball_vis": (views, frames), "ball_3d": (frames, 3),
    }
    for name, shape in shapes.items():
        value = getattr(result, name)
        expected_dtype = np.bool_ if name == "ball_vis" else np.float32
        if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != expected_dtype or not np.isfinite(value).all():
            raise ValueError(f"Invalid v2 {name}: expected finite {expected_dtype} {shape}")
    if result.player_position.dtype != np.float32 or not np.isfinite(result.player_position).all():
        raise ValueError("v2 player positions must be finite float32")
    optional_shapes = {
        "smpl_body_pose": (players, frames, 63), "smpl_global_orient": (players, frames, 3),
        "smpl_betas": (players, 10), "player_canonical_pose": (players, frames, 17, 3),
    }
    for name, shape in optional_shapes.items():
        value = getattr(result, name)
        if value is not None and (value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all()):
            raise ValueError(f"Invalid v2 {name} shape/dtype/finite values")
    vertices = result.smpl_vertices_local
    if vertices is not None and (vertices.ndim != 4 or vertices.shape[:2] != (players, frames) or vertices.shape[-1] != 3 or vertices.dtype != np.float32):
        raise ValueError("v2 SMPL vertices must have float32 shape (P,T,V,3)")
    for name, mask_name in (
        ("player_position", "player_valid"), ("player_yaw", "player_heading_valid"),
        ("player_kp_3d", "player_kp_3d_vis"), ("ball_3d", "ball_3d_valid"),
        ("smpl_vertices_local", "player_smpl_valid"), ("smpl_global_orient", "player_smpl_valid"),
        ("smpl_body_pose", "player_smpl_valid"),
    ):
        value = getattr(result, name)
        mask = masks[mask_name]
        if value is None:
            if mask.any():
                raise ValueError(f"v2 {name} missing for valid frames")
            continue
        if value.shape[:mask.ndim] != mask.shape or not np.isfinite(value).all():
            raise ValueError(f"v2 {name} shape/finite contract failed")
        nonzero = np.any(value != 0, axis=tuple(range(mask.ndim, value.ndim)))
        if (nonzero & ~mask).any():
            raise ValueError(f"v2 {name} must be zero in invalid frames")
    if (masks["player_heading_valid"] & ~masks["player_valid"]).any() or (masks["player_smpl_valid"] & ~(masks["player_valid"] & masks["player_heading_valid"])).any():
        raise ValueError("v2 body/heading validity requires a valid player root")
    if ((masks["player_valid"] | masks["player_kp_3d_vis"].any(-1)) & ~masks["player_observed"]).any():
        raise ValueError("v2 player reconstruction lacks accepted observations")
    if result.ball_vis is None or (masks["ball_3d_valid"] & (result.ball_vis.sum(0) < 2)).any():
        raise ValueError("v2 ball 3D requires at least two views")
    if (masks["player_valid"].any() or masks["player_kp_3d_vis"].any() or masks["ball_3d_valid"].any()) and not isinstance(result.metadata.get("court_reference"), dict):
        raise ValueError("v2 valid geometry requires a resolved camera reference")


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
