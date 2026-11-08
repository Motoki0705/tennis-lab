"""Head-mounted pinholes from placed SMPL vertices and parent-relative rotations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.visualization.blender.export.camera import PinholeCamera
from src.utils.geometry.matrices import axis_angle_to_rotation_matrix
from src.utils.schema.player import SMPLH_BODY_JOINT_NAMES, SMPLH_SKELETON

# Neutral SMPL/SMPL-X: +Y up, +Z face-forward, +X towards the subject's left.
# Columns are OpenCV right, down, forward expressed in that head frame.
OPENCV_TO_HEAD = np.diag([-1.0, -1.0, 1.0])


class CameraRejectionCode(IntEnum):
    VALID = 0
    MESH_INVALID = 1
    HEAD_POSE_NONFINITE = 2
    PLACEMENT_ROTATION_INVALID = 3
    HEAD_POSITION_NONFINITE = 4


@dataclass(frozen=True, slots=True)
class HeadCameraConfig:
    """Prototype offset in unit-scale SMPL head axes; clipping in court metres."""

    width: int = 1280
    height: int = 720
    horizontal_fov_degrees: float = 90.0
    face_offset_head_m: tuple[float, float, float] = (0.0, 0.08, 0.12)
    near_m: float = 0.05
    far_m: float = 100.0

    def __post_init__(self) -> None:
        if any(type(v) is not int or v <= 0 for v in (self.width, self.height)):
            raise ValueError("Camera width and height must be positive integers")
        if not np.isfinite(self.horizontal_fov_degrees) or not (
            0 < self.horizontal_fov_degrees < 180
        ):
            raise ValueError("horizontal_fov_degrees must be finite and in (0, 180)")
        offset = np.asarray(self.face_offset_head_m)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("face_offset_head_m must contain three finite values")
        if not np.isfinite([self.near_m, self.far_m]).all() or not (
            0 < self.near_m < self.far_m
        ):
            raise ValueError("Clipping requires 0 < near_m < far_m, both finite")

    def intrinsic_matrix(self) -> NDArray[np.float64]:
        focal = self.width / (2 * np.tan(np.deg2rad(self.horizontal_fov_degrees) / 2))
        return np.array(
            [[focal, 0, self.width / 2], [0, focal, self.height / 2], [0, 0, 1]],
            dtype=np.float64,
        )


def head_joint_chain(
    parents: NDArray, body_joint_names: Sequence[str], mesh_joint_names: Sequence[str]
) -> list[int]:
    """Validate the named body/mesh correspondence before choosing any joint row."""
    if tuple(body_joint_names) != SMPLH_BODY_JOINT_NAMES or tuple(mesh_joint_names) != (
        *SMPLH_BODY_JOINT_NAMES,
        "left_hand",
        "right_hand",
    ):
        raise ValueError("Bundle must declare the standard named SMPL body/mesh joints")
    if (
        parents.shape != (22,)
        or parents.dtype.kind not in "iu"
        or parents[0] != -1
        or {(int(parents[j]), j) for j in range(1, 22)} != set(SMPLH_SKELETON)
    ):
        raise ValueError("Bundle parents disagree with the named SMPL body skeleton")
    joint = body_joint_names.index("head")
    chain = []
    while joint != 0:
        chain.append(joint)
        joint = int(parents[joint])
    return chain[::-1]


@dataclass(frozen=True, slots=True)
class HeadCameraSeries:
    K: NDArray[np.float64]  # one constant (3,3) per player
    camera_to_court: NDArray[np.float64]  # (F,4,4), all NaN when invalid
    camera_valid: NDArray[np.bool_]
    camera_rejection_code: NDArray[np.uint8]
    head_position: NDArray[np.float64]  # regressed from placed mesh, (F,3)
    config: HeadCameraConfig

    def pinhole(self, frame_index: int) -> PinholeCamera:
        """Refuse missing cameras; never manufacture an origin or fixed viewpoint."""
        if not self.camera_valid[frame_index]:
            reason = CameraRejectionCode(self.camera_rejection_code[frame_index]).name
            raise ValueError(f"Camera sample {frame_index} is invalid: {reason}")
        transform = self.camera_to_court[frame_index]
        rotation = transform[:3, :3].T
        return PinholeCamera(
            self.K,
            rotation,
            -rotation @ transform[:3, 3],
            self.config.width,
            self.config.height,
        )


def build_head_camera_series(
    *,
    placed_vertices: NDArray,
    body_pose: NDArray,
    body_to_court_rotation: NDArray,
    mesh_valid: NDArray,
    scale: float | None,
    joint_regressor: NDArray,
    parents: NDArray,
    body_joint_names: Sequence[str],
    mesh_joint_names: Sequence[str],
    config: HeadCameraConfig,
) -> HeadCameraSeries:
    """Use mesh-regressed head position and rotational FK, without temporal edits.

    Placement/root centering and scale are already baked into the vertices.
    The saved decoded body rotation is composed with the articulated head chain;
    encoded global_orient is deliberately not an input. SMPL-X FK *positions*
    are not interchangeable with positions on the converted SMPL topology.
    """
    chain = head_joint_chain(parents, body_joint_names, mesh_joint_names)
    frames = len(body_pose)
    if (
        body_pose.shape != (frames, 63)
        or body_to_court_rotation.shape != (frames, 3, 3)
        or mesh_valid.shape != (frames,)
        or mesh_valid.dtype != np.bool_
        or placed_vertices.ndim != 3
        or placed_vertices.shape[0] != frames
        or placed_vertices.shape[2] != 3
        or frames == 0
    ):
        raise ValueError("Head-camera inputs disagree on frame/pose/mesh dimensions")
    if (
        joint_regressor.shape != (24, placed_vertices.shape[1])
        or not np.isfinite(joint_regressor).all()
        or not np.allclose(joint_regressor.sum(1), 1, atol=1e-4)
    ):
        raise ValueError(
            "SMPL joint regressor must be finite (24,V), rows summing to one"
        )
    if mesh_valid.any() and (
        scale is None or isinstance(scale, bool) or not np.isfinite(scale) or scale <= 0
    ):
        raise ValueError("Valid mesh requires a recorded positive placement scale")

    pose = body_pose.reshape(frames, 21, 3)[:, np.array(chain) - 1]
    reason: NDArray[np.uint8] = np.full(
        frames, CameraRejectionCode.VALID, dtype=np.uint8
    )
    reason[~mesh_valid] = CameraRejectionCode.MESH_INVALID
    reason[mesh_valid & ~np.isfinite(pose).all((1, 2))] = (
        CameraRejectionCode.HEAD_POSE_NONFINITE
    )
    # Invalid placement may include C1's zero-filled missing frames. Its mask
    # wins over numeric checks, and no rotation projection/repair is attempted.
    rotation_ok = np.isfinite(body_to_court_rotation).all((1, 2))
    candidates = np.flatnonzero(rotation_ok)
    rotation = np.asarray(body_to_court_rotation[candidates], dtype=np.float64)
    rotation_ok[candidates] &= (
        np.max(np.abs(rotation @ rotation.transpose(0, 2, 1) - np.eye(3)), axis=(1, 2))
        < 1e-5
    )
    rotation_ok[candidates] &= np.abs(np.linalg.det(rotation) - 1) < 1e-5
    reason[(reason == 0) & ~rotation_ok] = (
        CameraRejectionCode.PLACEMENT_ROTATION_INVALID
    )

    head = np.full((frames, 3), np.nan)
    head_row = mesh_joint_names.index("head")
    for start in range(0, frames, 32):
        stop = min(start + 32, frames)
        head[start:stop] = np.einsum(
            "v,fvc->fc",
            joint_regressor[head_row],
            placed_vertices[start:stop],
            dtype=np.float64,
        )
    reason[(reason == 0) & ~np.isfinite(head).all(1)] = (
        CameraRejectionCode.HEAD_POSITION_NONFINITE
    )
    valid = reason == CameraRejectionCode.VALID
    head[~valid] = np.nan
    transform = np.full((frames, 4, 4), np.nan)
    if valid.any():
        head_rotation = np.asarray(body_to_court_rotation[valid], dtype=np.float64)
        for step in range(len(chain)):
            head_rotation = head_rotation @ axis_angle_to_rotation_matrix(
                np.asarray(pose[valid, step], dtype=np.float64)
            )
        transform[valid] = np.eye(4)
        transform[valid, :3, :3] = head_rotation @ OPENCV_TO_HEAD
        assert scale is not None  # validated above; never infer a missing scale
        transform[valid, :3, 3] = head[valid] + (
            head_rotation @ (scale * np.asarray(config.face_offset_head_m))
        )
    return HeadCameraSeries(
        config.intrinsic_matrix(), transform, valid, reason, head, config
    )
