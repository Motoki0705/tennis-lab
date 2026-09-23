"""Triangulate identified people, then place independently recovered bodies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
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
from src.tennis_scene.pipeline.model_io.body import (
    BodyGeometry,
    BodyParameters,
    BodyRecoveryRequest,
)
from src.tennis_scene.pipeline.model_io.observations import (
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


class BodyRecovery(Protocol):
    @property
    def root_regressor(self) -> NDArray[np.float64]: ...
    def recover(self, request: BodyRecoveryRequest) -> BodyParameters: ...
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


def _segments(
    support: NDArray[np.bool_],
    source_frames: NDArray[np.int64],
    observed_any: NDArray[np.bool_],
    fps: float,
) -> list[NDArray[np.int64]]:
    active = np.flatnonzero(support)
    if not len(active):
        return []
    starts = [0]
    for index in range(1, len(active)):
        previous, current = int(active[index - 1]), int(active[index])
        missing = np.arange(previous + 1, current)
        # An unassigned actual detection may be another identity: do not bridge it.
        ambiguous = bool(observed_any[source_frames[missing]].any())
        gap = (source_frames[current] - source_frames[previous]) / fps
        if missing.size and (ambiguous or gap > 0.1):
            starts.append(index)
    ends = [*starts[1:], len(active)]
    return [
        np.arange(active[start], active[end - 1] + 1, dtype=np.int64)
        for start, end in zip(starts, ends, strict=True)
        if end - start >= 2
    ]


def reconstruct_player_bodies(
    grouped: GroupedObservations,
    raw: ObjectObservations,
    skeleton: PlayerSkeleton,
    cameras: tuple[PinholeCamera, ...],
    video_paths: tuple[Path, ...],
    sample_frames: NDArray[np.int64],
    *,
    body: BodyRecovery | None,
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
                coverage = grouped.visibility[player].any(-1).sum(-1)
                scores = grouped.confidence[player].mean(axis=(1, 2))
                view = min(
                    range(views),
                    key=lambda v: (
                        -int(coverage[v]),
                        -float(scores[v]),
                        cameras[v].camera_id,
                    ),
                )
                diagnostics["source_cameras"][str(int(identity))] = cameras[
                    view
                ].camera_id
                raw_rows = grouped.raw_indices[player, view, sample_frames]
                support = raw_rows >= 0
                ranges = _segments(
                    support, sample_frames, raw.observed[view].any(-1), raw.fps
                )
                predicted: list[tuple[NDArray[np.int64], BodyParameters, int]] = []
                for segment in ranges:
                    source = sample_frames[segment]
                    present = support[segment]
                    original_rows = raw_rows[segment[present]]
                    boxes: NDArray[np.float32] = np.empty((len(segment), 3), np.float32)
                    keypoints: NDArray[np.float32] = np.zeros(
                        (len(segment), 17, 3), np.float32
                    )
                    actual_boxes = raw.boxes_xys[view, source[present], original_rows]
                    for coordinate in range(3):
                        boxes[:, coordinate] = np.interp(
                            source, source[present], actual_boxes[:, coordinate]
                        )
                    keypoints[present, :, :2] = raw.uv_px[
                        view, source[present], original_rows
                    ]
                    keypoints[present, :, 2] = np.where(
                        grouped.visibility[player, view, source[present]],
                        raw.confidence[view, source[present], original_rows],
                        0,
                    )
                    params = body.recover(
                        BodyRecoveryRequest(
                            video_paths[view],
                            source,
                            keypoints,
                            boxes,
                            raw.size,
                            cameras[view].intrinsic,
                        )
                    )
                    predicted.append((source, params, int(present.sum())))
                    diagnostics["segments"].append(
                        {
                            "identity": int(identity),
                            "camera_id": cameras[view].camera_id,
                            "start_frame": int(source[0]),
                            "end_frame": int(source[-1]) + 1,
                            "observed_samples": int(present.sum()),
                        }
                    )
                    diagnostics["raw_parameters"].append(
                        {
                            "identity": int(identity),
                            "camera_id": cameras[view].camera_id,
                            "source_frames": source,
                            "body_pose": params.body_pose,
                            "global_orient": params.global_orient,
                            "betas": params.betas,
                            "transl": params.transl,
                        }
                    )
                if not predicted:
                    continue
                betas[player] = np.average(
                    np.stack([p.betas.mean(0) for _, p, _ in predicted]),
                    axis=0,
                    weights=[n for _, _, n in predicted],
                )
                local_joints = np.zeros((frames, 17, 3), np.float64)
                offsets = np.zeros((frames, 3), np.float64)
                rotations = np.zeros((frames, 3, 3), np.float64)
                segment_ids = np.full(frames, -1, np.int64)
                for segment_id, (source, params, _) in enumerate(predicted):
                    dense_frames: NDArray[np.int64] = np.arange(
                        int(source[0]), int(source[-1]) + 1, dtype=np.int64
                    )
                    segment_ids[dense_frames] = segment_id
                    eligible = (
                        skeleton.valid[player, dense_frames].sum(-1)
                        >= placement_config.min_joints
                    )
                    dense_frames = dense_frames[eligible]
                    for offset in range(0, len(dense_frames), 32):
                        selected = dense_frames[offset : offset + 32]
                        dense_params = BodyParameters(
                            interpolate_rotations(params.body_pose, source, selected),
                            interpolate_rotations(
                                params.global_orient, source, selected
                            ),
                            np.repeat(betas[player, None], len(selected), axis=0),
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
                        if vertices is None:
                            vertices = np.zeros(
                                (players, frames, geometry.vertices.shape[1], 3),
                                np.float32,
                            )
                        vertices[player, selected] = canonical.vertices
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
                        poses[player, selected] = dense_params.body_pose
                weights = placement_weights(skeleton, grouped, player, placement_config)
                fitted = fit_supported_track(
                    local_joints,
                    np.asarray(skeleton.positions[player], np.float64),
                    weights,
                    segment_ids,
                    fps=raw.fps,
                    config=placement_config,
                )
                diagnostics["players"][str(int(identity))] = {
                    "scale": fitted.scale,
                    "scale_pairs": fitted.scale_pairs,
                    "intervals": fitted.intervals,
                    "weighted_joint_frames": int((weights > 0).sum()),
                    "rejection_counts": {
                        str(int(code)): int((fitted.reasons == code).sum())
                        for code in np.unique(fitted.reasons)
                    },
                }
                roots_reason[player] = fitted.reasons
                accepted_frames = np.flatnonzero(fitted.valid)
                if fitted.scale is None or vertices is None:
                    continue
                for offset in range(0, len(accepted_frames), 32):
                    selected = accepted_frames[offset : offset + 32]
                    canonical = CanonicalBody(
                        vertices[player, selected],
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
                        visible = skeleton.inliers[
                            player, camera_index, selected
                        ].copy()
                        torso = np.zeros_like(visible)
                        torso[:, [5, 6, 11, 12]] = True
                        visible &= torso
                        residual = np.linalg.norm(
                            projected - grouped.uv_px[player, camera_index, selected],
                            axis=-1,
                        )
                        error = np.where(
                            visible, np.where(front, residual, np.inf), 0
                        ).max(-1)
                        errors = np.maximum(errors, error)
                    root_valid[player, selected] = True
                    roots_reason[player, selected] = 0
                    heading_valid[player, selected] = placed.heading_valid
                    position[player, selected] = placed.position
                    yaw[player, selected] = placed.yaw
                    accepted = placed.heading_valid & (errors <= reprojection_px)
                    mesh_valid[player, selected] = accepted
                    mesh_error[player, selected] = np.where(
                        np.isfinite(errors), errors, np.finfo(np.float32).max
                    )
                    good = selected[accepted]
                    orient[player, good] = placed.global_orient[accepted]
                    vertices[player, good] = placed.vertices_local[accepted]
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
