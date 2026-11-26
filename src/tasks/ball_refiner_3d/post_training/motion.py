"""SMPL-X articulation and trajectory features on the original frame grid."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from src.submodules.vendor.gvhmr.body_model.smplx_lite import batch_rigid_transform_v2
from src.tasks.ball_refiner_3d.model_io.adapters import normalization
from src.tennis_scene.schema import SceneResult
from src.tennis_scene.visualization.player_view.assets import load_body_kinematics
from src.utils.geometry.matrices import (
    SMPL_Y_UP_TO_COURT_Z_UP,
    axis_angle_to_rotation_matrix,
    rotation_matrix_z,
)
from src.utils.geometry.rotation_conversions import matrix_to_rotation_6d
from src.utils.schema.player import SMPLH_JOINT_IDX

PLAYER_SLOTS = 4
PLAYER_FEATURES = 161
COORDINATE_SCALE = normalization(3)[0]
VELOCITY_SCALE_MPS = 10.0
WRISTS = [SMPLH_JOINT_IDX[name] for name in ("left_wrist", "right_wrist")]


@dataclass(frozen=True, slots=True)
class BodySkeleton:
    rest: NDArray[np.float32]
    parents: NDArray[np.int32]

    @classmethod
    def load(cls, path: Path) -> BodySkeleton:
        assets = load_body_kinematics(path)
        return cls(assets["smplx_rest_joints"], assets["smplx_parents"])


@dataclass(frozen=True, slots=True)
class MotionFeatures:
    ball_m: NDArray[np.float32]
    ball_valid: NDArray[np.bool_]
    players: NDArray[np.float32]
    root_valid: NDArray[np.bool_]
    smpl_valid: NDArray[np.bool_]
    track_ids: tuple[int | None, ...]


def player_slots(scene: SceneResult) -> tuple[int | None, ...]:
    """Return source rows: negative-y then positive-y, ascending ID per side."""
    ids = scene.player_track_ids
    sides = scene.metadata["player_sides"]
    count = len(scene.player_position)
    if (
        ids is None
        or ids.shape != (count,)
        or ids.dtype.kind not in "iu"
        or (ids < 0).any()
        or len(set(ids.tolist())) != count
        or not isinstance(sides, list)
        or len(sides) != count
        or any(type(side) is not int or side not in (-1, 1) for side in sides)
    ):
        raise ValueError(
            "Player rows require unique stable IDs and explicit court sides"
        )
    slots: list[int | None] = []
    for side in (-1, 1):
        side_rows = sorted(
            (row for row, value in enumerate(sides) if value == side),
            key=lambda row: int(ids[row]),
        )
        if len(side_rows) > 2:
            raise ValueError(
                f"More than two players were assigned to court side {side}"
            )
        slots.extend([*side_rows, *([None] * (2 - len(side_rows)))])
    return tuple(slots)


def _require_placement_scale(scene: SceneResult, identity: int) -> float:
    scale = scene.metadata["body_placement"]["players"][str(identity)]["scale"]
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError(f"Player {identity} requires a recorded positive body scale")
    return float(scale)


def wrist_features(
    body_pose: NDArray[np.float32],
    compensation: NDArray[np.float32],
    yaw: NDArray[np.float32],
    scale: float,
    skeleton: BodySkeleton,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return parent-relative 6D pose and court-axis wrist offsets from joint 0."""
    count = len(body_pose)
    rotation = axis_angle_to_rotation_matrix(body_pose.reshape(count, 21, 3))
    rotation_tensor = torch.from_numpy(rotation)
    poses = matrix_to_rotation_6d(rotation_tensor).reshape(count, 126).numpy()
    root_rot = torch.eye(3).expand(count, 1, 3, 3)
    joints, _ = batch_rigid_transform_v2(
        torch.cat((root_rot, rotation_tensor), dim=1),
        torch.from_numpy(skeleton.rest),
        torch.from_numpy(skeleton.parents.astype(np.int64)),
    )
    joints_local = (joints[:, WRISTS] - joints[:, :1]).numpy()
    # SceneResult stores a renderer compensation rotation. Reverse it using the
    # renderer's axis conversion and yaw to recover the fitted court rotation.
    body_to_court = (
        rotation_matrix_z(yaw)
        @ SMPL_Y_UP_TO_COURT_Z_UP
        @ axis_angle_to_rotation_matrix(compensation).swapaxes(-1, -2)
    )
    offsets = scale * np.einsum("tij,twj->twi", body_to_court, joints_local)
    return poses, offsets


def _velocity(
    positions: NDArray[np.float32],
    valid: NDArray[np.bool_],
    fps: float,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    available = np.zeros_like(valid)
    available[1:] = valid[1:] & valid[:-1]
    velocity = np.zeros_like(positions)
    velocity[1:] = np.where(
        available[1:, None],
        np.diff(positions, axis=0) * (fps / VELOCITY_SCALE_MPS),
        0,
    )
    return velocity, available


def extract_motion(scene: SceneResult, skeleton: BodySkeleton) -> MotionFeatures:
    """Build four slots with independent flags and post-normalization zero filling."""
    ids = scene.player_track_ids
    roots, headings, bodies = (
        scene.player_valid,
        scene.player_heading_valid,
        scene.player_smpl_valid,
    )
    if any(
        value is None
        for value in (ids, roots, headings, bodies, scene.ball_3d, scene.ball_3d_valid)
    ):
        raise ValueError("Player motion requires a masked SceneResult")
    assert roots is not None and headings is not None and bodies is not None
    assert (
        ids is not None
        and scene.ball_3d is not None
        and scene.ball_3d_valid is not None
    )
    frames, ball_valid = scene.num_frames, scene.ball_3d_valid
    features = np.zeros((frames, PLAYER_SLOTS, PLAYER_FEATURES), np.float32)
    root_slots = np.zeros((frames, PLAYER_SLOTS), dtype=np.bool_)
    smpl_slots = np.zeros_like(root_slots)
    track_ids: list[int | None] = []
    ball = np.where(ball_valid[:, None], scene.ball_3d, 0)
    for slot, row in enumerate(player_slots(scene)):
        if row is None:
            track_ids.append(None)
            continue
        identity = int(ids[row])
        track_ids.append(identity)
        root_ok, heading_ok, smpl_ok = roots[row], headings[row], bodies[row]
        root_slots[:, slot], smpl_slots[:, slot] = root_ok, smpl_ok
        root, yaw = scene.player_position[row], scene.player_yaw[row]
        pose = np.zeros((frames, 126), np.float32)
        wrist = np.zeros((frames, 2, 3), np.float32)
        if smpl_ok.any():
            if scene.smpl_body_pose is None or scene.smpl_global_orient is None:
                raise ValueError(
                    f"Valid SMPL frames lack pose or placement for {identity}"
                )
            pose[smpl_ok], wrist[smpl_ok] = wrist_features(
                scene.smpl_body_pose[row, smpl_ok],
                scene.smpl_global_orient[row, smpl_ok],
                yaw[smpl_ok],
                _require_placement_scale(scene, identity),
                skeleton,
            )
        root_velocity, root_velocity_ok = _velocity(root, root_ok, scene.fps)
        wrist_m = wrist.reshape(frames, 6)
        wrist_velocity, wrist_velocity_ok = _velocity(wrist_m, smpl_ok, scene.fps)
        ball_from_root = ball - root
        ball_from_wrist = ball_from_root[:, None, :] - wrist
        flags = np.stack(
            (
                root_ok,
                heading_ok,
                smpl_ok,
                ball_valid,
                root_velocity_ok,
                wrist_velocity_ok,
            ),
            axis=-1,
        )
        features[:, slot] = np.concatenate(
            (
                pose,
                np.where(root_ok[:, None], root / COORDINATE_SCALE, 0),
                np.where(
                    heading_ok[:, None],
                    np.stack((np.sin(yaw), np.cos(yaw)), axis=-1),
                    0,
                ),
                root_velocity,
                wrist_m,
                wrist_velocity,
                np.where(
                    (ball_valid & root_ok)[:, None],
                    ball_from_root / COORDINATE_SCALE,
                    0,
                ),
                np.where(
                    (ball_valid & smpl_ok)[:, None, None],
                    ball_from_wrist / COORDINATE_SCALE,
                    0,
                ).reshape(frames, 6),
                flags,
            ),
            axis=-1,
        )
    features[~root_slots] = 0
    return MotionFeatures(
        ball, ball_valid, features, root_slots, smpl_slots, tuple(track_ids)
    )
