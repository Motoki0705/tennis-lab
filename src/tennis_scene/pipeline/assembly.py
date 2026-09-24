"""Build one mask-complete scene without inventing geometry for missing data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.base.generate_dataset import resolve_court_keypoint_contract
from src.tasks.base.model_io import write_model_artifact_court_keypoint_contract
from src.tennis_scene.pipeline.components.ball_reconstruction import (
    BallReconstructionResult,
)
from src.tennis_scene.pipeline.components.camera_geometry import CameraGeometryResult
from src.tennis_scene.pipeline.components.court_kp import CourtKPResult
from src.tennis_scene.pipeline.components.player_reconstruction import (
    PlayerSkeleton,
    ReconstructedPlayers,
)
from src.tennis_scene.pipeline.model_io.observations import GroupedObservations
from src.tennis_scene.schema import (
    SceneResult,
    attach_scene_result_court_keypoint_provenance,
    validate_scene_result_arrays,
)
from src.utils.video import VideoInfo


def assemble_automatic_scene(
    *,
    video_paths: tuple[Path, ...],
    camera_ids: tuple[str, ...],
    info: VideoInfo,
    court: CourtKPResult,
    active_indices: tuple[int, ...],
    geometry: CameraGeometryResult | None,
    grouped_people: GroupedObservations | None,
    skeleton: PlayerSkeleton | None,
    players: ReconstructedPlayers | None,
    ball: BallReconstructionResult | None,
    metadata: dict[str, Any],
) -> SceneResult:
    views, frames = court.keypoints.shape[:2]
    identities = np.empty(0, np.int64) if grouped_people is None else grouped_people.identities
    count = len(identities)
    root = np.zeros((count, frames, 3), np.float32) if players is None else players.position
    yaw = np.zeros((count, frames), np.float32) if players is None else players.yaw
    root_valid = np.zeros((count, frames), bool) if players is None else players.root_valid
    heading_valid = np.zeros((count, frames), bool) if players is None else players.heading_valid
    mesh_valid = np.zeros((count, frames), bool) if players is None else players.smpl_valid
    human = np.zeros((count, views, frames, 17, 2), np.float32)
    human_vis = np.zeros(human.shape[:-1], np.float32)
    observed = np.zeros((count, frames), bool)
    if grouped_people is not None:
        human[:, list(active_indices)] = grouped_people.uv_px / np.asarray((info.width, info.height), np.float32)
        human_vis[:, list(active_indices)] = grouped_people.confidence * grouped_people.visibility
        observed = grouped_people.visibility.any(axis=(1, 3))
    points = np.zeros((count, frames, 17, 3), np.float32) if skeleton is None else skeleton.positions
    point_valid = np.zeros((count, frames, 17), bool) if skeleton is None else skeleton.valid
    canonical = np.zeros(points.shape, np.float32)
    if players is not None:
        from src.utils.geometry.matrices import rotation_matrix_z
        canonical = np.einsum("ptij,ptkj->ptki", rotation_matrix_z(-yaw), points - root[:, :, None]).astype(np.float32)
        canonical[~(point_valid & root_valid[..., None] & heading_valid[..., None])] = 0
    ball_uv = np.zeros((views, frames, 2), np.float32)
    ball_vis = np.zeros((views, frames), bool)
    ball_xyz = np.zeros((frames, 3), np.float32)
    ball_valid = np.zeros(frames, bool)
    ball_reasons = np.ones(frames, np.uint8)
    if ball is not None:
        ball_uv[list(active_indices)] = ball.uv_px / np.asarray((info.width, info.height), np.float32)
        ball_vis[list(active_indices)] = ball.visibility
        ball_xyz, ball_valid, ball_reasons = ball.trajectory.positions, ball.trajectory.valid, ball.trajectory.reasons
    body_metadata = {} if players is None else {k: v for k, v in players.diagnostics.items() if k != "raw_parameters"}
    meta = {
        **metadata,
        "scene_schema_version": 2,
        "representation": "triangulated_coco17_and_temporal_smpl_v1",
        "normalization": "image_width_height",
        "video_paths": [str(p) for p in video_paths],
        "camera_ids": list(camera_ids), "num_cameras": views, "sync_assumption": "preprocessed",
        "court_reference": None if geometry is None else geometry.document,
        "court_detection": court.diagnostics,
        "track_ids": identities.tolist(),
        "identity_scope": "one_model_input_clip",
        "body_placement": body_metadata,
        "ball_observation_contract": "single_detection_per_camera_frame",
        "validity_statistics": {
            "player_root_frames": root_valid.sum(-1).tolist(), "player_joint_frames": point_valid.any(-1).sum(-1).tolist(),
            "player_smpl_frames": mesh_valid.sum(-1).tolist(), "ball_3d_frames": int(ball_valid.sum()),
        },
    }
    scene = SceneResult(
        num_frames=frames, fps=info.fps, width=info.width, height=info.height,
        court_kp=(court.keypoints * (np.maximum(np.array([info.width, info.height], np.float32) - 1, 1) / np.array([info.width, info.height], np.float32))).astype(np.float32),
        court_vis=court.visibility.astype(np.float32), player_position=root, player_yaw=yaw,
        smpl_body_pose=None if players is None else players.body_pose,
        smpl_global_orient=None if players is None else players.global_orient,
        smpl_betas=None if players is None else players.betas,
        smpl_vertices_local=None if players is None else players.vertices_local,
        ball_uv=ball_uv, ball_vis=ball_vis, ball_3d=ball_xyz,
        human_kp_2d=human, human_kp_vis=human_vis,
        player_track_ids=identities.astype(np.int32), player_kp_3d=points, player_canonical_pose=canonical,
        player_observed=observed, player_valid=root_valid, player_heading_valid=heading_valid,
        player_kp_3d_vis=point_valid, player_smpl_valid=mesh_valid, ball_3d_valid=ball_valid,
        player_rejection_code=np.ones((count, frames), np.uint8) if players is None else players.root_reasons,
        player_kp_3d_rejection_code=np.ones((count, frames, 17), np.uint8) if skeleton is None else skeleton.reasons,
        ball_rejection_code=ball_reasons, metadata=meta,
    )
    contract = resolve_court_keypoint_contract("camera_view_v2")
    if geometry is not None:
        attach_scene_result_court_keypoint_provenance(scene, contract, geometry.selection.provenance)
    else:
        write_model_artifact_court_keypoint_contract(scene.metadata, contract)
    validate_scene_result_arrays(scene)
    return scene
