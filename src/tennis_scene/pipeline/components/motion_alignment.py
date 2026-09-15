"""Align GVHMR world motion to PLCS for storage alongside the PLCS placement.

The module always fits one gravity-fixed similarity transform shared by every
frame of a track. The aligned root, yaw and SMPL parameters are returned as an
alternative representation; the original PLCS and camera-frame GVHMR fields
remain unchanged in :class:`SceneResult`. The existing renderer rule
``Rz(yaw) @ A @ R(G)^T @ (V_local - root(V_local)) + position`` reproduces the
direct similarity transform of the world vertices.

Aligned SMPL field contract
---------------------------

With ``A = SMPL_Y_UP_TO_COURT_Z_UP``, ``s``/``psi``/``b`` the fitted scale,
yaw and translation, ``V_can`` the canonical (orientation-free) posed vertices,
``c`` the canonical joint-0 root and ``R_world`` the world rotation, the aligned
fields are exactly:

* ``player_position = s * Rz(psi) @ p_src + b``
* ``player_yaw = wrap(theta_G + psi)``
* ``smpl_vertices_local = s * (V_can - c)``
* ``smpl_global_orient = matrix_to_axis_angle(R_world^T @ A^T @ Rz(theta_G) @ A)``
* ``smpl_body_pose`` and ``smpl_betas`` unchanged.

``theta_G`` is the heading of the court-frame world rotation ``A @ R_world``.
No silent fallback exists: an incomplete GVHMR world artifact, an unobservable
track or a solver failure raises instead of degrading to the PLCS track.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

from src.tennis_scene.motion_alignment.similarity import (
    SimilarityConfig,
    SimilarityFitResult,
    SimilarityTransform,
    fit_similarity,
    wrap_to_pi,
)
from src.utils.geometry.matrices import SMPL_Y_UP_TO_COURT_Z_UP
from src.utils.geometry.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.tennis_scene.pipeline.components.player_association import (
        PlayerAssociationApplied,
    )

_TORSO_JOINTS = (5, 6, 11, 12)
_HIP_JOINTS = (11, 12)


def load_joint_regressor(path: str | Path) -> NDArray[np.float64]:
    """Load the SMPL neutral joint regressor that defines the joint-0 root.

    Accepts the full dense ``(24, V)`` regressor or a bare ``(V,)`` row for
    joint 0. A missing file, a non-tensor payload or any other shape raises.
    """
    regressor_path = Path(path)
    if not regressor_path.is_file():
        raise FileNotFoundError(
            f"SMPL joint regressor asset is missing: {regressor_path}"
        )
    loaded = torch.load(regressor_path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, torch.Tensor):
        raise TypeError(
            f"SMPL joint regressor asset must contain a tensor: {regressor_path}"
        )
    if loaded.ndim == 1:
        valid = loaded.shape[0] > 0
    elif loaded.ndim == 2:
        valid = loaded.shape[0] >= 1 and loaded.shape[1] > 0
    else:
        valid = False
    if not valid:
        raise ValueError(
            "SMPL joint regressor must have shape (24, V) or (V,), got "
            f"{tuple(loaded.shape)}."
        )
    values = np.asarray(loaded.to(dtype=torch.float64).numpy(), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(
            f"SMPL joint regressor contains non-finite values: {regressor_path}"
        )
    return values


def _root_regressor_row(regressor: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the ``(V,)`` joint-0 row from a full or single-row regressor."""
    values = np.asarray(regressor, dtype=np.float64)
    if values.ndim == 1:
        return values
    if values.ndim == 2 and values.shape[0] >= 1:
        return values[0]
    raise ValueError(
        f"joint_regressor must have shape (24, V) or (V,), got {values.shape}."
    )


def _as_rotation_matrices(
    value: NDArray[np.float64], *, name: str
) -> NDArray[np.float64]:
    values = np.asarray(value, dtype=np.float64)
    if values.ndim != 2 or values.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (T, 3), got {values.shape}.")
    return np.asarray(
        axis_angle_to_matrix(torch.from_numpy(values)).numpy(), dtype=np.float64
    )


def _as_frame_array(
    value: NDArray[np.float64], *, name: str, num_frames: int
) -> NDArray[np.float64]:
    values = np.asarray(value, dtype=np.float64)
    if values.shape != (num_frames, 3):
        raise ValueError(
            f"{name} must have shape ({num_frames}, 3), got {values.shape}."
        )
    return values


def _frame_rotations_z(yaw: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return batched ``Rz(yaw)`` matrices as ``(T, 3, 3)`` float64."""
    angles = np.asarray(yaw, dtype=np.float64)
    cos = np.cos(angles)
    sin = np.sin(angles)
    zeros = np.zeros_like(cos)
    ones = np.ones_like(cos)
    return np.stack(
        (
            np.stack((cos, -sin, zeros), axis=-1),
            np.stack((sin, cos, zeros), axis=-1),
            np.stack((zeros, zeros, ones), axis=-1),
        ),
        axis=-2,
    )


@dataclass(frozen=True, slots=True)
class PlayerMotionConfig:
    """Validated GVHMR-to-PLCS alignment settings for one pipeline run."""

    scale_mode: Literal["fixed", "free"]
    smpl_joint_regressor: Path
    similarity: SimilarityConfig

    def __post_init__(self) -> None:
        if self.scale_mode not in {"fixed", "free"}:
            raise ValueError(
                "player_motion.scale_mode must be 'fixed' or 'free', got "
                f"{self.scale_mode!r}."
            )
        if not isinstance(self.smpl_joint_regressor, Path):
            raise TypeError(
                "player_motion.smpl_joint_regressor must be a pathlib.Path."
            )
        if not isinstance(self.similarity, SimilarityConfig):
            raise TypeError("player_motion.similarity must be a SimilarityConfig.")

    def fit_config(self) -> SimilarityConfig:
        """Return the similarity config for the selected scale mode."""
        return replace(
            self.similarity,
            fixed_scale=1.0 if self.scale_mode == "fixed" else None,
        )


@dataclass(frozen=True, slots=True)
class CanonicalMotion:
    """Orientation-free GVHMR motion derived once per player track."""

    canonical_vertices: NDArray[np.float64]  # (T, V, 3), root-relative later
    canonical_root: NDArray[np.float64]  # (T, 3)
    world_root: NDArray[np.float64]  # (T, 3), Y-up
    world_rotation: NDArray[np.float64]  # (T, 3, 3), Y-up
    source_position: NDArray[np.float64]  # (T, 3), court frame
    source_heading: NDArray[np.float64]  # (T,)
    rotation_court: NDArray[np.float64]  # (T, 3, 3), court frame
    observed: NDArray[np.bool_]  # (T,)


@dataclass(frozen=True, slots=True)
class AlignmentWeights:
    """Final per-frame fit weights plus the confidence-clipping audit flag."""

    position: NDArray[np.float64]
    heading: NDArray[np.float64]
    observed: NDArray[np.bool_]
    confidence_clipped: bool


@dataclass(frozen=True, slots=True)
class AlignedTrack:
    """Aligned court-frame fields for one player track."""

    player_position: NDArray[np.float64]
    player_yaw: NDArray[np.float64]
    smpl_vertices_local: NDArray[np.float64]
    smpl_global_orient: NDArray[np.float64]
    transform: SimilarityTransform
    fit: SimilarityFitResult
    source_position: NDArray[np.float64]
    source_heading: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PlayerMotionApplied:
    """Aligned player motion fields plus their diagnostic metadata."""

    player_position: NDArray[np.float32]
    player_yaw: NDArray[np.float32]
    smpl_global_orient: NDArray[np.float32]
    smpl_vertices_local: NDArray[np.float32]
    metadata: dict[str, Any]


def derive_canonical_motion(
    *,
    vertices_incam: NDArray[np.float64],
    global_orient_incam: NDArray[np.float64],
    transl_incam: NDArray[np.float64],
    global_orient_world: NDArray[np.float64],
    transl_world: NDArray[np.float64],
    joint_regressor: NDArray[np.float64],
) -> CanonicalMotion:
    """Derive the orientation-free motion and its court-frame source track.

    ``vertices_incam`` is ``(T, V, 3)``; the axis-angle orientations and
    translations are ``(T, 3)``. ``joint_regressor`` is the dense ``(24, V)``
    regressor or its ``(V,)`` joint-0 row. No array is clipped or repaired: the
    observed mask only records which frames are finite.
    """
    vertices = np.asarray(vertices_incam, dtype=np.float64)
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(
            f"vertices_incam must have shape (T, V, 3), got {vertices.shape}."
        )
    num_frames = int(vertices.shape[0])
    if num_frames == 0:
        raise ValueError("vertices_incam must contain at least one frame.")
    in_cam_translation = _as_frame_array(
        transl_incam, name="transl_incam", num_frames=num_frames
    )
    world_translation = _as_frame_array(
        transl_world, name="transl_world", num_frames=num_frames
    )
    rotation_incam = _as_rotation_matrices(
        global_orient_incam, name="global_orient_incam"
    )
    rotation_world = _as_rotation_matrices(
        global_orient_world, name="global_orient_world"
    )
    for name, values in (
        ("global_orient_incam", rotation_incam),
        ("global_orient_world", rotation_world),
    ):
        if values.shape[0] != num_frames:
            raise ValueError(
                f"{name} must have {num_frames} frames, got {values.shape[0]}."
            )
    root_row = _root_regressor_row(joint_regressor)
    if root_row.shape[0] != vertices.shape[1]:
        raise ValueError(
            "joint_regressor row must match the vertex count "
            f"{vertices.shape[1]}, got {root_row.shape[0]}."
        )

    canonical = np.einsum(
        "tji,tvj->tvi",
        rotation_incam,
        vertices - in_cam_translation[:, None, :],
    )
    canonical_root = np.einsum("v,tvc->tc", root_row, canonical)
    world_root = (
        np.einsum("tij,tj->ti", rotation_world, canonical_root) + world_translation
    )
    basis = np.asarray(SMPL_Y_UP_TO_COURT_Z_UP, dtype=np.float64)
    source_position = world_root @ basis.T
    rotation_court = np.einsum("ij,tjk->tik", basis, rotation_world)
    source_heading = np.arctan2(rotation_court[:, 1, 0], rotation_court[:, 0, 0])
    observed = (
        np.isfinite(source_position).all(axis=-1)
        & np.isfinite(world_root).all(axis=-1)
        & np.isfinite(in_cam_translation).all(axis=-1)
        & np.isfinite(world_translation).all(axis=-1)
    )
    return CanonicalMotion(
        canonical_vertices=canonical,
        canonical_root=canonical_root,
        world_root=world_root,
        world_rotation=rotation_world,
        source_position=source_position,
        source_heading=source_heading,
        rotation_court=rotation_court,
        observed=observed,
    )


def derive_alignment_weights(
    *,
    target_visibility_ref: NDArray[np.float64],
    source_visibility_ref: NDArray[np.float64],
    court_visibility: NDArray[np.float64],
    rotation_court: NDArray[np.float64],
    observed: NDArray[np.bool_],
) -> AlignmentWeights:
    """Return the PR's confidence-proxy weights for one player track.

    ``target_visibility_ref`` and ``source_visibility_ref`` are the reference
    camera's ``(T, 17)`` 2D confidences for the PLCS and GVHMR sides, and
    ``court_visibility`` is ``(N, T, K)``. No separate PLCS uncertainty exists,
    so both sides use the SceneResult/GVHMR detection confidences. Confidences
    are clipped into ``[0, 1]`` and the clipping is recorded in the audit flag.
    ``rotation_court`` supplies the horizontal projection
    ``||R_court[:, :2, 0]||`` that discounts tilted headings.
    """
    target_raw = np.asarray(target_visibility_ref, dtype=np.float64)
    source_raw = np.asarray(source_visibility_ref, dtype=np.float64)
    court_raw = np.asarray(court_visibility, dtype=np.float64)
    rotations = np.asarray(rotation_court, dtype=np.float64)
    if target_raw.ndim != 2 or target_raw.shape[1] != 17:
        raise ValueError(
            f"target_visibility_ref must have shape (T, 17), got {target_raw.shape}."
        )
    if source_raw.shape != target_raw.shape:
        raise ValueError(
            "source_visibility_ref must match target_visibility_ref shape "
            f"{target_raw.shape}, got {source_raw.shape}."
        )
    if court_raw.ndim != 3:
        raise ValueError(
            f"court_visibility must have shape (N, T, K), got {court_raw.shape}."
        )
    num_frames = int(target_raw.shape[0])
    if court_raw.shape[1] != num_frames:
        raise ValueError(
            "court_visibility frame axis must match the confidence frames, got "
            f"{court_raw.shape[1]} and {num_frames}."
        )
    if rotations.shape != (num_frames, 3, 3):
        raise ValueError(
            "rotation_court must have shape "
            f"({num_frames}, 3, 3), got {rotations.shape}."
        )
    observed_mask = np.asarray(observed, dtype=np.bool_)
    if observed_mask.shape != (num_frames,):
        raise ValueError(
            f"observed must have shape ({num_frames},), got {observed_mask.shape}."
        )
    for name, values in (
        ("target_visibility_ref", target_raw),
        ("source_visibility_ref", source_raw),
        ("court_visibility", court_raw),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains NaN or infinity.")

    clipped = bool(
        (target_raw < 0.0).any()
        or (target_raw > 1.0).any()
        or (source_raw < 0.0).any()
        or (source_raw > 1.0).any()
        or (court_raw < 0.0).any()
        or (court_raw > 1.0).any()
    )
    target = np.clip(target_raw, 0.0, 1.0)
    source = np.clip(source_raw, 0.0, 1.0)
    court = np.clip(court_raw, 0.0, 1.0).max(axis=0).mean(axis=-1)
    target_hips = target[:, 11:13].mean(axis=-1)
    target_torso = target[:, list(_TORSO_JOINTS)].min(axis=-1)
    source_hips = source[:, list(_HIP_JOINTS)].mean(axis=-1)
    source_torso = source[:, list(_TORSO_JOINTS)].min(axis=-1)
    horizontal = np.linalg.norm(rotations[:, :2, 0], axis=-1)
    return AlignmentWeights(
        position=np.asarray(
            np.clip(source_hips, 0.0, 1.0) * target_hips * court * observed_mask,
            dtype=np.float64,
        ),
        heading=np.asarray(
            source_torso * target_torso * court * observed_mask * horizontal**2,
            dtype=np.float64,
        ),
        observed=observed_mask,
        confidence_clipped=clipped,
    )


def align_track_to_plcs(
    *,
    motion: CanonicalMotion,
    target_position: NDArray[np.float64],
    target_yaw: NDArray[np.float64],
    weights: AlignmentWeights,
    config: SimilarityConfig,
) -> AlignedTrack:
    """Fit one similarity transform and rewrite the renderer's input fields."""
    num_frames = int(motion.canonical_vertices.shape[0])
    target = _as_frame_array(
        target_position, name="target_position", num_frames=num_frames
    )
    target_yaw_values = np.asarray(target_yaw, dtype=np.float64)
    if target_yaw_values.shape != (num_frames,):
        raise ValueError(
            f"target_yaw must have shape ({num_frames},), got "
            f"{target_yaw_values.shape}."
        )
    position_weights = np.asarray(weights.position, dtype=np.float64)
    heading_weights = np.asarray(weights.heading, dtype=np.float64)
    for name, values in (
        ("position_weights", position_weights),
        ("heading_weights", heading_weights),
    ):
        if values.shape != (num_frames,):
            raise ValueError(
                f"{name} must have shape ({num_frames},), got {values.shape}."
            )
    finite_observed = (
        motion.observed
        & np.isfinite(target).all(axis=-1)
        & np.isfinite(target_yaw_values)
    )
    fit = fit_similarity(
        motion.source_position,
        target,
        motion.source_heading,
        target_yaw_values,
        position_weights * finite_observed,
        heading_weights * finite_observed,
        config=config,
    )
    transform = fit.transform
    player_position = transform.apply(motion.source_position)
    player_yaw = transform.apply_heading(motion.source_heading)
    vertices_local = transform.scale * (
        motion.canonical_vertices - motion.canonical_root[:, None, :]
    )
    basis = np.asarray(SMPL_Y_UP_TO_COURT_Z_UP, dtype=np.float64)
    # R(G_out) = R_world^T @ A^T @ Rz(theta_G) @ A removes the world rotation and
    # leaves exactly the residual tilt the renderer rule must re-apply.
    gravity_rotation = _frame_rotations_z(motion.source_heading)
    court_alignment = np.einsum("ki,tkl,lj->tij", basis, gravity_rotation, basis)
    output_rotation = np.einsum("tji,tjk->tik", motion.world_rotation, court_alignment)
    global_orient = np.asarray(
        matrix_to_axis_angle(torch.from_numpy(output_rotation)).numpy(),
        dtype=np.float64,
    )
    return AlignedTrack(
        player_position=player_position,
        player_yaw=player_yaw,
        smpl_vertices_local=vertices_local,
        smpl_global_orient=global_orient,
        transform=transform,
        fit=fit,
        source_position=motion.source_position,
        source_heading=motion.source_heading,
    )


def _summarize_residuals(
    values: NDArray[np.float64],
) -> dict[str, float | int]:
    """Return ``{median, rmse, p90, count}`` for one residual vector."""
    residuals = np.asarray(values, dtype=np.float64)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0:
        return {"median": 0.0, "rmse": 0.0, "p90": 0.0, "count": 0}
    return {
        "median": float(np.median(residuals)),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "p90": float(np.percentile(residuals, 90)),
        "count": int(residuals.size),
    }


def _player_metadata(
    *,
    track: AlignedTrack,
    track_id: int,
    reference_position: NDArray[np.float64],
    reference_yaw: NDArray[np.float64],
    confidence_clipped: bool,
) -> dict[str, Any]:
    """Build the JSON-serializable per-player alignment diagnostics."""
    position_residual = np.linalg.norm(
        track.player_position - reference_position, axis=-1
    )
    heading_residual = np.abs(np.rad2deg(wrap_to_pi(track.player_yaw - reference_yaw)))
    diagnostics = track.fit.diagnostics
    return {
        "track_id": int(track_id),
        "scale": float(track.transform.scale),
        "yaw_rad": float(track.transform.yaw),
        "yaw_deg": float(np.rad2deg(track.transform.yaw)),
        "position_residual_m": _summarize_residuals(
            position_residual[track.fit.position_mask]
        ),
        "heading_residual_deg": _summarize_residuals(
            heading_residual[track.fit.heading_mask]
        ),
        "fit": {
            "success": bool(track.fit.success),
            "nfev": int(diagnostics.nfev),
            "optimality": float(diagnostics.optimality),
            "jacobian_rank": int(diagnostics.jacobian_rank),
            "n_free_parameters": int(diagnostics.n_free_parameters),
            "initializer": str(diagnostics.initializer),
        },
        "confidence_clipped": bool(confidence_clipped),
    }


class MotionAlignmentModule:
    """Build the GVHMR-aligned alternative fields for a ``SceneResult``."""

    def __init__(self, config: PlayerMotionConfig) -> None:
        self.config = config
        self._joint_regressor = load_joint_regressor(config.smpl_joint_regressor)

    def process(
        self,
        *,
        associated: PlayerAssociationApplied,
        plcs_position: NDArray[np.float32],
        plcs_yaw: NDArray[np.float32],
        court_visibility: NDArray[np.float32],
        reference_camera_index: int,
    ) -> PlayerMotionApplied:
        """Return the aligned alternative fields plus their metadata."""
        return self._align(
            associated=associated,
            plcs_position=plcs_position,
            plcs_yaw=plcs_yaw,
            court_visibility=court_visibility,
            reference_camera_index=reference_camera_index,
        )

    def _align(
        self,
        *,
        associated: PlayerAssociationApplied,
        plcs_position: NDArray[np.float32],
        plcs_yaw: NDArray[np.float32],
        court_visibility: NDArray[np.float32],
        reference_camera_index: int,
    ) -> PlayerMotionApplied:
        joint_regressor = self._joint_regressor
        missing = [
            name
            for name, value in (
                ("smpl_vertices_local", associated.smpl_vertices_local),
                ("smpl_transl_incam", associated.smpl_transl_incam),
                ("smpl_transl_world", associated.smpl_transl_world),
                ("smpl_global_orient_world", associated.smpl_global_orient_world),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "GVHMR alignment requires the world-motion fields "
                f"{sorted(missing)}, but they are absent. "
                "The GVHMR artifact was produced before world-motion support; "
                "regenerate it with the current pipeline."
            )
        vertices_local_input = associated.smpl_vertices_local
        transl_incam = associated.smpl_transl_incam
        transl_world = associated.smpl_transl_world
        orient_world = associated.smpl_global_orient_world
        assert vertices_local_input is not None
        assert transl_incam is not None
        assert transl_world is not None
        assert orient_world is not None

        position = np.asarray(plcs_position, dtype=np.float32)
        yaw = np.asarray(plcs_yaw, dtype=np.float32)
        if position.ndim != 3 or position.shape[-1] != 3:
            raise ValueError(
                f"plcs_position must have shape (P, T, 3), got {position.shape}."
            )
        if yaw.shape != position.shape[:2]:
            raise ValueError(
                f"plcs_yaw must have shape {position.shape[:2]}, got {yaw.shape}."
            )
        num_players = int(position.shape[0])
        num_frames = int(position.shape[1])
        if vertices_local_input.shape[:2] != (num_players, num_frames):
            raise ValueError(
                "aligned SMPL vertices must match the PLCS (P, T) axes, got "
                f"{vertices_local_input.shape[:2]} and "
                f"{(num_players, num_frames)}."
            )

        positions: list[NDArray[np.float64]] = []
        yaws: list[NDArray[np.float64]] = []
        vertices: list[NDArray[np.float64]] = []
        orients: list[NDArray[np.float64]] = []
        players: list[dict[str, Any]] = []
        for player_index in range(num_players):
            motion = derive_canonical_motion(
                vertices_incam=np.asarray(
                    vertices_local_input[player_index], dtype=np.float64
                ),
                global_orient_incam=np.asarray(
                    associated.smpl_global_orient[player_index], dtype=np.float64
                ),
                transl_incam=np.asarray(transl_incam[player_index], dtype=np.float64),
                global_orient_world=np.asarray(
                    orient_world[player_index], dtype=np.float64
                ),
                transl_world=np.asarray(transl_world[player_index], dtype=np.float64),
                joint_regressor=joint_regressor,
            )
            target_position = np.asarray(position[player_index], dtype=np.float64)
            target_yaw = np.asarray(yaw[player_index], dtype=np.float64)
            weights = derive_alignment_weights(
                target_visibility_ref=np.asarray(
                    associated.human_kp_vis[player_index, reference_camera_index],
                    dtype=np.float64,
                ),
                source_visibility_ref=np.asarray(
                    associated.human_kp_vis[player_index, reference_camera_index],
                    dtype=np.float64,
                ),
                court_visibility=np.asarray(court_visibility, dtype=np.float64),
                rotation_court=motion.rotation_court,
                observed=motion.observed,
            )
            track = align_track_to_plcs(
                motion=motion,
                target_position=target_position,
                target_yaw=target_yaw,
                weights=weights,
                config=self.config.fit_config(),
            )
            positions.append(track.player_position)
            yaws.append(track.player_yaw)
            vertices.append(track.smpl_vertices_local)
            orients.append(track.smpl_global_orient)
            players.append(
                _player_metadata(
                    track=track,
                    track_id=int(associated.track_ids[player_index]),
                    reference_position=target_position,
                    reference_yaw=target_yaw,
                    confidence_clipped=weights.confidence_clipped,
                )
            )
        metadata = {
            "gvhmr_alignment": {
                "scale_mode": self.config.scale_mode,
                "players": players,
            }
        }
        return PlayerMotionApplied(
            player_position=np.stack(positions).astype(np.float32),
            player_yaw=np.stack(yaws).astype(np.float32),
            smpl_global_orient=np.stack(orients).astype(np.float32),
            smpl_vertices_local=np.stack(vertices).astype(np.float32),
            metadata=metadata,
        )


__all__ = [
    "AlignedTrack",
    "AlignmentWeights",
    "CanonicalMotion",
    "MotionAlignmentModule",
    "PlayerMotionApplied",
    "PlayerMotionConfig",
    "align_track_to_plcs",
    "derive_alignment_weights",
    "derive_canonical_motion",
    "load_joint_regressor",
]
