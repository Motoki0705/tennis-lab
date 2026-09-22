"""Place incam bodies using triangulated hips and a fixed calibrated rotation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation, Slerp

from src.utils.geometry.triangulation import PinholeCamera


@dataclass(frozen=True)
class PlacedBody:
    position: NDArray[np.float32]
    yaw: NDArray[np.float32]
    vertices_local: NDArray[np.float32]
    global_orient: NDArray[np.float32]
    heading_valid: NDArray[np.bool_]
    vertices_court: NDArray[np.float32]
    joints_court: NDArray[np.float32]


def interpolate_rotations(
    values: NDArray[np.float32], sample_frames: NDArray[np.int64], frames: NDArray[np.int64]
) -> NDArray[np.float32]:
    """SO(3) interpolation within one accepted segment, never extrapolation."""
    if values.ndim != 2 or values.shape[0] != len(sample_frames) or values.shape[1] % 3 or len(sample_frames) < 2:
        raise ValueError("Rotations require at least two samples of axis-angle triples")
    if (np.diff(sample_frames) <= 0).any() or (frames < sample_frames[0]).any() or (frames > sample_frames[-1]).any():
        raise ValueError("Rotation interpolation forbids extrapolation")
    result = np.empty((len(frames), values.shape[1]), np.float32)
    for column in range(0, values.shape[1], 3):
        result[:, column:column + 3] = Slerp(sample_frames, Rotation.from_rotvec(values[:, column:column + 3]))(frames).as_rotvec()
    return result


def place_incam_body(
    vertices: NDArray[np.float32],
    coco_joints: NDArray[np.float32],
    orient: NDArray[np.float32],
    triangulated_hips: NDArray[np.float32],
    camera: PinholeCamera,
    root_regressor: NDArray[np.float64],
) -> PlacedBody:
    """Preserve tilt and the exact renderer transform, including the hip/root offset."""
    count = len(vertices)
    if vertices.ndim != 3 or vertices.shape[-1] != 3 or coco_joints.shape != (count, 17, 3) or orient.shape != (count, 3) or triangulated_hips.shape != (count, 2, 3):
        raise ValueError("SMPL placement arrays disagree")
    regressor = root_regressor[0] if root_regressor.ndim == 2 else root_regressor
    if regressor.shape != (vertices.shape[1],):
        raise ValueError("SMPL root regressor must match reconstructed mesh")
    if not all(np.isfinite(x).all() for x in (vertices, coco_joints, orient, triangulated_hips, regressor)):
        raise ValueError("SMPL placement inputs must be finite")
    root = np.einsum("v,tvc->tc", regressor, vertices)
    hip = coco_joints[:, 11:13].mean(1)
    position = triangulated_hips.mean(1) + (root - hip) @ camera.rotation
    in_cam = Rotation.from_rotvec(orient).as_matrix()
    court_body = camera.rotation.T @ in_cam
    yaw = np.arctan2(court_body[:, 1, 0], court_body[:, 0, 0])
    heading_valid = np.linalg.norm(court_body[:, :2, 0], axis=-1) > 1e-6
    yaw[~heading_valid] = 0
    canonical = np.einsum("tji,tvj->tvi", in_cam, vertices - root[:, None])
    # This is the same Y-up -> Z-up matrix used by the scene renderer.
    from src.utils.geometry.matrices import SMPL_Y_UP_TO_COURT_Z_UP
    a = np.asarray(SMPL_Y_UP_TO_COURT_Z_UP, np.float64)
    rz = Rotation.from_euler("z", yaw[:, None]).as_matrix()
    encoded = court_body.transpose(0, 2, 1) @ rz @ a
    world = (vertices - root[:, None]) @ camera.rotation + position[:, None]
    joints = (coco_joints - root[:, None]) @ camera.rotation + position[:, None]
    return PlacedBody(
        position.astype(np.float32), yaw.astype(np.float32), canonical.astype(np.float32),
        Rotation.from_matrix(encoded).as_rotvec().astype(np.float32), heading_valid,
        world.astype(np.float32), joints.astype(np.float32),
    )
