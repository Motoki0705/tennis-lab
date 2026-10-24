"""Triangulate identified people, then place independently recovered bodies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.motion_alignment.mesh_placement import (
    CanonicalBody,
    canonicalize_incam_body,
    interpolate_rotations,
    place_canonical_body,
)
from src.tennis_scene.motion_alignment.temporal import (
    PlacementRejection,
    TemporalPlacementConfig,
    fit_supported_track,
)
from src.tennis_scene.pipeline.body_types import (
    BodyGeometry,
    BodyParameters,
)
from src.tennis_scene.pipeline.components.gvhmr import GVHMROutput, RecoveredBody
from src.tennis_scene.pipeline.observation_types import (
    GroupedObservations,
    ObjectObservations,
)
from src.utils.geometry.triangulation import (
    PinholeCamera,
    PointTriangulationConfig,
    TriangulatedPoints,
    reject_excessive_speed,
    triangulate_points,
)


class BodyGeometryModel(Protocol):
    @property
    def root_regressor(self) -> NDArray[np.float64]: ...
    def reconstruct(self, parameters: BodyParameters) -> BodyGeometry: ...
    def unload(self) -> None: ...


@dataclass(frozen=True)
class PlayerSkeleton:
    positions: NDArray[np.float32]  # (P,T,17,3)
    valid: NDArray[np.bool_]
    reasons: NDArray[np.uint8]
    inliers: NDArray[np.bool_]  # (P,V,T,17)
    reprojection_px: NDArray[np.float32]


@dataclass(frozen=True)
class PlacementTrace:
    """Canonical body inputs and the fitted object transform for one identity.

    ``frames`` lists every frame covered by a recovered GVHMR segment.  The
    arrays are indexed by that list, so frames outside every segment (the
    existing rejection code 1) are absent.

    ``vertices_local`` is ``u_t = R_body,t @ v_t + root_from_hip_t``: the posed
    body at unit scale, hip-centered and court-oriented.  It is populated for
    every frame where body geometry was reconstructed, including frames whose
    placement was rejected; frames without a reconstructed body are zero.
    ``joints_local`` follows the same rule.  The public transform is
    ``world = hip_position + scale * Rz(yaw_correction) @ u_t``; callers must
    gate on ``valid`` and never interpolate across invalid frames.

    ``target_joints`` marks triangulation missing values with NaN, and
    ``target_weights`` is 0 there.  Consumers must gate on ``target_weights``
    and never treat NaN as a coordinate.
    ``hip_position``/``yaw_correction`` are the fit output, so frames rejected
    by the fit (101-105) are 0; the excessive-speed gate changes only
    ``valid``/``reasons``.  Those are the final root validity for this identity:
    0 is accepted, 6 is excessive speed, and 101-105 are ``PlacementRejection``
    values.
    """

    identity: int
    frames: NDArray[np.int64]  # (F,) frames covered by recovered segments
    segment_ids: NDArray[np.int64]  # (F,) GVHMR segment ID
    segment_camera_ids: tuple[str, ...]  # segment ID -> GVHMR camera
    vertices_local: NDArray[np.float32]  # (F,V,3) u_t, zero without a body
    joints_local: NDArray[np.float64]  # (F,17,3) J_src
    target_joints: NDArray[np.float32]  # (F,17,3) triangulated COCO17, NaN invalid
    target_weights: NDArray[np.float64]  # (F,17) placement_weights
    scale: float | None
    hip_position: NDArray[np.float64]  # (F,3) q_t, zero on rejected frames
    yaw_correction: NDArray[np.float64]  # (F,) psi_t, zero on rejected frames
    valid: NDArray[np.bool_]  # (F,)
    reasons: NDArray[np.uint8]  # (F,) final root rejection code


@dataclass(frozen=True)
class _PlayerPlacement(PlacementTrace):
    """A ``PlacementTrace`` plus the production-only arrays for one player."""

    source_camera: str
    segment_diagnostics: tuple[dict[str, Any], ...]
    player_diagnostics: dict[str, Any] | None
    betas: NDArray[np.float32]
    poses: NDArray[np.float32]
    position: NDArray[np.float32]  # fitted placement, before the speed gate
    yaw: NDArray[np.float32]  # placed body yaw, before the speed gate
    root_valid: NDArray[np.bool_]  # fitted validity, before the speed gate
    root_reasons: NDArray[np.uint8]  # fitted reasons, before the speed gate
    heading_valid: NDArray[np.bool_]
    global_orient: NDArray[np.float32]
    mesh_error: NDArray[np.float32]
    error: NDArray[np.float64]  # reprojection residual, inf behind a camera
    vertices: NDArray[np.float32] | None  # (T,V,3) placed local vertices


@dataclass(frozen=True)
class ReconstructedPlayers:
    position: NDArray[np.float32]
    yaw: NDArray[np.float32]
    root_valid: NDArray[np.bool_]
    heading_valid: NDArray[np.bool_]
    smpl_valid: NDArray[np.bool_]
    body_pose: NDArray[np.float32]
    global_orient: NDArray[np.float32]
    betas: NDArray[np.float32]
    vertices_local: NDArray[np.float32] | None
    root_reasons: NDArray[np.uint8]
    mesh_reprojection_px: NDArray[np.float32]
    diagnostics: dict[str, Any]


def triangulate_players(
    grouped: GroupedObservations,
    cameras: tuple[PinholeCamera, ...],
    *,
    reprojection_px: float,
    joint_confidence: float = 0.3,
) -> PlayerSkeleton:
    players, views, frames, joints, _ = grouped.uv_px.shape
    if joints != 17:
        raise ValueError("Player triangulation requires COCO17")
    results = [
        triangulate_points(
            grouped.uv_px[p],
            grouped.visibility[p] & (grouped.confidence[p] >= joint_confidence),
            cameras,
            config=PointTriangulationConfig(reprojection_px, (-0.5, 4.0)),
            confidence=grouped.confidence[p],
        )
        for p in range(players)
    ]
    return PlayerSkeleton(
        np.stack([r.positions for r in results])
        if results
        else np.zeros((0, frames, 17, 3), np.float32),
        np.stack([r.valid for r in results])
        if results
        else np.zeros((0, frames, 17), bool),
        np.stack([r.reasons for r in results])
        if results
        else np.zeros((0, frames, 17), np.uint8),
        np.stack([r.inliers for r in results])
        if results
        else np.zeros((0, views, frames, 17), bool),
        np.stack([r.reprojection_px for r in results])
        if results
        else np.zeros((0, views, frames, 17), np.float32),
    )


def trace_player_placement(
    recovered_person: RecoveredBody,
    skeleton: PlayerSkeleton,
    grouped: GroupedObservations,
    cameras: tuple[PinholeCamera, ...],
    *,
    player: int,
    body: BodyGeometryModel,
    fps: float,
    placement_config: TemporalPlacementConfig,
) -> PlacementTrace:
    """Return the canonical body and fitted transform for one identity.

    This is the pure per-player placement computation behind
    ``reconstruct_player_bodies``.  ``player`` indexes ``grouped`` and
    ``skeleton``; ``recovered_person.person_id`` must match that identity.  It
    does not unload the body model, mutate the observations, or apply the mesh
    reprojection gate.  ``valid`` and ``reasons`` include the existing
    excessive-speed rejection, so they match
    ``ReconstructedPlayers.root_valid``/``root_reasons`` on ``frames``.
    """
    return _place_player(
        recovered_person,
        skeleton,
        grouped,
        cameras,
        player=player,
        body=body,
        fps=fps,
        placement_config=placement_config,
    )


def _empty_player_placement(
    identity: int,
    frames: int,
    source_camera: str,
    segment_diagnostics: tuple[dict[str, Any], ...],
    betas: NDArray[np.float32],
    poses: NDArray[np.float32],
) -> _PlayerPlacement:
    """An identity with no recovered GVHMR segment contributes no placement."""
    return _PlayerPlacement(
        identity,
        np.zeros(0, np.int64),
        np.zeros(0, np.int64),
        (),
        np.zeros((0, 0, 3), np.float32),
        np.zeros((0, 17, 3), np.float64),
        np.zeros((0, 17, 3), np.float32),
        np.zeros((0, 17), np.float64),
        None,
        np.zeros((0, 3), np.float64),
        np.zeros(0, np.float64),
        np.zeros(0, bool),
        np.zeros(0, np.uint8),
        source_camera,
        segment_diagnostics,
        None,
        betas,
        poses,
        np.zeros((frames, 3), np.float32),
        np.zeros(frames, np.float32),
        np.zeros(frames, bool),
        np.ones(frames, np.uint8),
        np.zeros(frames, bool),
        np.zeros((frames, 3), np.float32),
        np.zeros(frames, np.float32),
        np.zeros(frames, np.float64),
        None,
    )


def _place_player(
    recovered_person: RecoveredBody,
    skeleton: PlayerSkeleton,
    grouped: GroupedObservations,
    cameras: tuple[PinholeCamera, ...],
    *,
    player: int,
    body: BodyGeometryModel,
    fps: float,
    placement_config: TemporalPlacementConfig,
) -> _PlayerPlacement:
    """Canonicalize, fit and place one identity; the sole per-player path."""
    identity = int(grouped.identities[player])
    if int(recovered_person.person_id) != identity:
        raise ValueError("Recovered body does not match grouped identity")
    frames = int(grouped.raw_indices.shape[2])
    view = tuple(c.camera_id for c in cameras).index(recovered_person.camera_id)
    predicted = [
        (segment.source_frames, segment.parameters, segment.observed_samples)
        for segment in recovered_person.segments
    ]
    segment_diagnostics = tuple(
        {
            "identity": identity,
            "camera_id": recovered_person.camera_id,
            "start_frame": int(segment.source_frames[0]),
            "end_frame": int(segment.source_frames[-1]) + 1,
            "observed_samples": segment.observed_samples,
        }
        for segment in recovered_person.segments
    )
    segment_camera_ids = tuple(recovered_person.camera_id for _ in predicted)
    betas: NDArray[np.float32] = np.zeros(10, np.float32)
    poses: NDArray[np.float32] = np.zeros((frames, 63), np.float32)
    if not predicted:
        return _empty_player_placement(
            identity,
            frames,
            recovered_person.camera_id,
            segment_diagnostics,
            betas,
            poses,
        )
    betas[:] = np.average(
        np.stack([p.betas.mean(0) for _, p, _ in predicted]),
        axis=0,
        weights=[n for _, _, n in predicted],
    )
    local_joints: NDArray[np.float64] = np.zeros((frames, 17, 3), np.float64)
    offsets: NDArray[np.float64] = np.zeros((frames, 3), np.float64)
    rotations: NDArray[np.float64] = np.zeros((frames, 3, 3), np.float64)
    segment_ids: NDArray[np.int64] = np.full(frames, -1, np.int64)
    canonical_vertices: NDArray[np.float32] | None = None
    for segment_id, (source, params, _) in enumerate(predicted):
        dense_frames: NDArray[np.int64] = np.arange(
            int(source[0]), int(source[-1]) + 1, dtype=np.int64
        )
        segment_ids[dense_frames] = segment_id
        eligible = (
            skeleton.valid[player, dense_frames].sum(-1) >= placement_config.min_joints
        )
        dense_frames = dense_frames[eligible]
        for offset in range(0, len(dense_frames), 32):
            selected = dense_frames[offset : offset + 32]
            dense_params = BodyParameters(
                interpolate_rotations(params.body_pose, source, selected),
                interpolate_rotations(params.global_orient, source, selected),
                np.repeat(betas[None], len(selected), axis=0),
                np.stack(
                    [
                        np.interp(selected, source, params.transl[:, c])
                        for c in range(3)
                    ],
                    -1,
                ).astype(np.float32),
            )
            geometry = body.reconstruct(dense_params)
            canonical = canonicalize_incam_body(
                geometry.vertices,
                geometry.coco17,
                dense_params.global_orient,
                cameras[view],
                body.root_regressor,
            )
            if canonical_vertices is None:
                canonical_vertices = np.zeros(
                    (frames, geometry.vertices.shape[1], 3), np.float32
                )
            canonical_vertices[selected] = canonical.vertices
            (
                local_joints[selected],
                offsets[selected],
                rotations[selected],
            ) = (
                canonical.joints,
                canonical.root_from_hip,
                canonical.rotation,
            )
            segment_ids[selected] = segment_id
            poses[selected] = dense_params.body_pose
    weights = placement_weights(skeleton, grouped, player, placement_config)
    fitted = fit_supported_track(
        local_joints,
        np.asarray(skeleton.positions[player], np.float64),
        weights,
        segment_ids,
        fps=fps,
        config=placement_config,
    )
    player_diagnostics = {
        "scale": fitted.scale,
        "scale_pairs": fitted.scale_pairs,
        "intervals": fitted.intervals,
        "weighted_joint_frames": int((weights > 0).sum()),
        "rejection_counts": {
            str(int(code)): int((fitted.reasons == code).sum())
            for code in np.unique(fitted.reasons)
        },
    }
    position: NDArray[np.float32] = np.zeros((frames, 3), np.float32)
    yaw: NDArray[np.float32] = np.zeros(frames, np.float32)
    root_valid: NDArray[np.bool_] = np.zeros(frames, bool)
    heading_valid: NDArray[np.bool_] = np.zeros(frames, bool)
    orient: NDArray[np.float32] = np.zeros((frames, 3), np.float32)
    mesh_error: NDArray[np.float32] = np.zeros(frames, np.float32)
    error: NDArray[np.float64] = np.zeros(frames, np.float64)
    roots_reason = fitted.reasons.copy()
    placed_vertices = (
        None if canonical_vertices is None else np.zeros_like(canonical_vertices)
    )
    accepted_frames = np.flatnonzero(fitted.valid)
    if fitted.scale is not None and canonical_vertices is not None:
        for offset in range(0, len(accepted_frames), 32):
            selected = accepted_frames[offset : offset + 32]
            canonical = CanonicalBody(
                canonical_vertices[selected],
                local_joints[selected],
                offsets[selected],
                rotations[selected],
            )
            placed = place_canonical_body(
                canonical,
                fitted.hip_position[selected],
                fitted.yaw_correction[selected],
                fitted.scale,
            )
            errors: NDArray[np.float64] = np.zeros(len(selected), np.float64)
            for camera_index, camera in enumerate(cameras):
                projected, front = camera.project(placed.joints_court)
                visible = skeleton.inliers[player, camera_index, selected].copy()
                torso = np.zeros_like(visible)
                torso[:, [5, 6, 11, 12]] = True
                visible &= torso
                residual = np.linalg.norm(
                    projected - grouped.uv_px[player, camera_index, selected],
                    axis=-1,
                )
                camera_error = np.where(
                    visible, np.where(front, residual, np.inf), 0
                ).max(-1)
                errors = np.maximum(errors, camera_error)
            root_valid[selected] = True
            roots_reason[selected] = 0
            heading_valid[selected] = placed.heading_valid
            position[selected] = placed.position
            yaw[selected] = placed.yaw
            error[selected] = errors
            mesh_error[selected] = np.where(
                np.isfinite(errors), errors, np.finfo(np.float32).max
            )
            orient[selected] = placed.global_orient
            assert placed_vertices is not None
            placed_vertices[selected] = placed.vertices_local
    # Expose final root validity, including the excessive-speed gate that
    # reconstruct_player_bodies applies after placing every player.
    traced = reject_excessive_speed(
        TriangulatedPoints(
            position,
            root_valid,
            roots_reason,
            np.zeros((len(cameras), frames), bool),
            np.zeros((len(cameras), frames), np.float32),
        ),
        fps=fps,
        max_speed_mps=12.0,
    )
    trace_frames = np.flatnonzero(segment_ids >= 0).astype(np.int64, copy=False)
    target_joints = np.where(
        skeleton.valid[player, trace_frames, :, None],
        skeleton.positions[player, trace_frames],
        np.nan,
    ).astype(np.float32)
    if canonical_vertices is None:
        vertices_local: NDArray[np.float32] = np.zeros(
            (len(trace_frames), 0, 3), np.float32
        )
    else:
        vertices_local = (
            np.einsum(
                "tij,tvj->tvi",
                rotations[trace_frames],
                canonical_vertices[trace_frames].astype(np.float64),
            )
            + offsets[trace_frames, None, :]
        ).astype(np.float32)
    return _PlayerPlacement(
        identity,
        trace_frames,
        segment_ids[trace_frames].copy(),
        segment_camera_ids,
        vertices_local,
        local_joints[trace_frames].copy(),
        target_joints,
        weights[trace_frames].copy(),
        fitted.scale,
        fitted.hip_position[trace_frames].copy(),
        fitted.yaw_correction[trace_frames].copy(),
        traced.valid[trace_frames].copy(),
        traced.reasons[trace_frames].copy(),
        recovered_person.camera_id,
        segment_diagnostics,
        player_diagnostics,
        betas,
        poses,
        position,
        yaw,
        root_valid,
        roots_reason,
        heading_valid,
        orient,
        mesh_error,
        error,
        placed_vertices,
    )


def reconstruct_player_bodies(
    grouped: GroupedObservations,
    raw: ObjectObservations,
    skeleton: PlayerSkeleton,
    cameras: tuple[PinholeCamera, ...],
    recovered: GVHMROutput,
    *,
    body: BodyGeometryModel | None,
    reprojection_px: float,
    placement_config: TemporalPlacementConfig,
) -> ReconstructedPlayers:
    players, views, frames = grouped.raw_indices.shape
    position = np.zeros((players, frames, 3), np.float32)
    yaw = np.zeros((players, frames), np.float32)
    root_valid = np.zeros((players, frames), bool)
    heading_valid = root_valid.copy()
    mesh_valid = root_valid.copy()
    poses = np.zeros((players, frames, 63), np.float32)
    orient = np.zeros((players, frames, 3), np.float32)
    betas = np.zeros((players, 10), np.float32)
    roots_reason = np.ones((players, frames), np.uint8)
    mesh_error = np.zeros((players, frames), np.float32)
    vertices: NDArray[np.float32] | None = None
    diagnostics: dict[str, Any] = {
        "source_cameras": {},
        "segments": [],
        "players": {},
        "placement": "coco17_temporal_position_yaw_v1",
        "config": asdict(placement_config),
        "rejection_codes": {reason.name: int(reason) for reason in PlacementRejection},
        "raw_parameters": [],
    }
    if body is not None and players:
        if raw.boxes_xys is None:
            raise ValueError("Body recovery requires recorded detector boxes")
        try:
            for player, identity in enumerate(grouped.identities):
                matches = [
                    item for item in recovered.bodies if item.person_id == int(identity)
                ]
                if not matches:
                    continue
                if len(matches) != 1:
                    raise ValueError("Body artifacts contain duplicate person IDs")
                placement = trace_player_placement(
                    matches[0],
                    skeleton,
                    grouped,
                    cameras,
                    player=player,
                    body=body,
                    fps=raw.fps,
                    placement_config=placement_config,
                )
                if not isinstance(placement, _PlayerPlacement):
                    raise TypeError("Placement trace is missing production arrays")
                diagnostics["source_cameras"][str(int(identity))] = (
                    placement.source_camera
                )
                diagnostics["segments"].extend(placement.segment_diagnostics)
                if placement.player_diagnostics is None:
                    continue
                diagnostics["players"][str(int(identity))] = (
                    placement.player_diagnostics
                )
                betas[player] = placement.betas
                poses[player] = placement.poses
                position[player] = placement.position
                yaw[player] = placement.yaw
                root_valid[player] = placement.root_valid
                roots_reason[player] = placement.root_reasons
                heading_valid[player] = placement.heading_valid
                mesh_error[player] = placement.mesh_error
                accepted = placement.heading_valid & (
                    placement.error <= reprojection_px
                )
                mesh_valid[player] = accepted
                orient[player, accepted] = placement.global_orient[accepted]
                if placement.vertices is not None:
                    if vertices is None:
                        vertices = np.zeros(
                            (
                                players,
                                frames,
                                placement.vertices.shape[1],
                                3,
                            ),
                            np.float32,
                        )
                    vertices[player, accepted] = placement.vertices[accepted]
        finally:
            body.unload()
    for player in range(players):
        result = reject_excessive_speed(
            TriangulatedPoints(
                position[player],
                root_valid[player],
                roots_reason[player],
                np.zeros((views, frames), bool),
                np.zeros((views, frames), np.float32),
            ),
            fps=raw.fps,
            max_speed_mps=12.0,
        )
        position[player], root_valid[player], roots_reason[player] = (
            result.positions,
            result.valid,
            result.reasons,
        )
        heading_valid[player] &= result.valid
        mesh_valid[player] &= result.valid
        yaw[player, ~heading_valid[player]] = 0
        poses[player, ~mesh_valid[player]] = 0
        orient[player, ~mesh_valid[player]] = 0
        if vertices is not None:
            vertices[player, ~mesh_valid[player]] = 0
    return ReconstructedPlayers(
        position,
        yaw,
        root_valid,
        heading_valid,
        mesh_valid,
        poses,
        orient,
        betas,
        vertices,
        roots_reason,
        mesh_error,
        diagnostics,
    )


def placement_weights(
    skeleton: PlayerSkeleton,
    grouped: GroupedObservations,
    player: int,
    config: TemporalPlacementConfig,
) -> NDArray[np.float64]:
    """Use triangulation inliers, confidence and reprojection; absent values stay absent."""
    used = skeleton.inliers[player]
    count = used.sum(0)
    rms = np.sqrt(
        np.sum(
            np.where(used, skeleton.reprojection_px[player].astype(np.float64) ** 2, 0),
            axis=0,
        )
        / np.maximum(count, 1)
    )
    confidence = np.sum(
        np.where(used, grouped.confidence[player], 0), axis=0
    ) / np.maximum(count, 1)
    height = skeleton.positions[player, :, :, 2]
    valid = (
        skeleton.valid[player]
        & (count >= 2)
        & (rms <= config.max_reprojection_rms_px)
        & (height >= -0.25)
        & (height < 3.5)
    )
    weights = np.where(
        valid, confidence / (1 + (rms / config.reprojection_weight_sigma_px) ** 2), 0
    )
    weights[:, :5] *= 0.5
    return np.asarray(weights, np.float64)
