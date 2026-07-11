"""Ground-plane homography from detected court keypoints.

The issue #634 clips carry **no calibrated cameras**, so a full 3D->2D
projection is impossible without fabricating intrinsics. What *is* well
defined from 2D court keypoints alone is the court-plane (z=0) homography:
world ``(x, y, 1)`` on the ground maps to image pixels. The 2D overlay uses
it to draw predicted player footprints and the ball's ground shadow
``(x, y, 0)`` — explicitly documented as a ground projection, never as a
projection of elevated points. Anything requiring true camera parameters
raises instead of guessing.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from src.utils.schema.court import court_keypoints_3d


class GroundProjectionError(RuntimeError):
    """Homography could not be established from the available court points."""


def default_court_kp_indices(num_court_kp: int) -> tuple[int, ...]:
    """Map scene court-keypoint rows to CourtKP20 indices.

    - ``K == 20``: full CourtKP20.
    - ``K == 14``: the real court detector's 14 ground keypoints
      (CourtKP20 indices 0..13; all z=0).

    Any other K needs an explicit mapping from the caller.
    """
    if num_court_kp == 20:
        return tuple(range(20))
    if num_court_kp == 14:
        return tuple(range(14))
    raise GroundProjectionError(
        f"No default CourtKP20 index mapping for K={num_court_kp}; pass "
        "court_kp_indices explicitly."
    )


def ground_homography_from_court(
    court_kp_uv: NDArray[np.float32],
    court_vis: NDArray[np.float32],
    *,
    width: int,
    height: int,
    court_kp_indices: tuple[int, ...] | None = None,
    min_points: int = 4,
    vis_threshold: float = 0.5,
) -> NDArray[np.float64]:
    """Estimate the world-ground -> pixel homography for one frame.

    Args:
        court_kp_uv: ``(K, 2)`` normalized UV court keypoints.
        court_vis: ``(K,)`` visibility scores.
        width / height: Image size in pixels (UV is denormalized with these).
        court_kp_indices: CourtKP20 index per row (default mapping for K in
            {14, 20}); rows mapping to elevated keypoints (z>0) are excluded.
        min_points: Minimum usable ground correspondences.

    Returns:
        ``(3, 3)`` homography ``H`` with ``pixel ~ H @ (x, y, 1)``.
    """
    court_kp_uv = np.asarray(court_kp_uv, dtype=np.float32)
    court_vis = np.asarray(court_vis, dtype=np.float32)
    if court_kp_uv.ndim != 2 or court_kp_uv.shape[1] != 2:
        raise GroundProjectionError(
            f"court_kp_uv must be (K, 2), got {court_kp_uv.shape}."
        )
    num_kp = court_kp_uv.shape[0]
    if court_vis.shape != (num_kp,):
        raise GroundProjectionError(
            f"court_vis must be (K,)={num_kp}, got {court_vis.shape}."
        )
    indices = (
        default_court_kp_indices(num_kp) if court_kp_indices is None else court_kp_indices
    )
    if len(indices) != num_kp:
        raise GroundProjectionError(
            f"court_kp_indices has {len(indices)} entries for K={num_kp} keypoints."
        )
    kp20: NDArray[np.float32] = court_keypoints_3d().numpy()
    if any(not 0 <= i < kp20.shape[0] for i in indices):
        raise GroundProjectionError(f"court_kp_indices out of CourtKP20 range: {indices}.")

    world_pts: list[NDArray[np.float32]] = []
    image_pts: list[NDArray[np.float32]] = []
    for row, kp20_idx in enumerate(indices):
        world = kp20[kp20_idx]
        if abs(float(world[2])) > 1e-6:
            continue  # elevated keypoint: not on the ground plane
        if float(court_vis[row]) < vis_threshold:
            continue
        uv = court_kp_uv[row]
        if not np.isfinite(uv).all():
            continue
        world_pts.append(world[:2])
        image_pts.append(np.array([uv[0] * width, uv[1] * height], dtype=np.float32))

    if len(world_pts) < min_points:
        raise GroundProjectionError(
            f"Only {len(world_pts)} usable ground correspondences "
            f"(need >= {min_points}); court keypoints too occluded for a homography."
        )
    homography, inlier_mask = cv2.findHomography(
        np.stack(world_pts), np.stack(image_pts), cv2.RANSAC, 5.0
    )
    if homography is None or inlier_mask is None or int(inlier_mask.sum()) < min_points:
        raise GroundProjectionError(
            "cv2.findHomography failed to produce a stable ground homography "
            f"({0 if inlier_mask is None else int(inlier_mask.sum())} inliers)."
        )
    return np.asarray(homography, dtype=np.float64)


def project_ground_points(
    homography: NDArray[np.float64],
    points_xy_m: NDArray[np.float32] | torch.Tensor,
) -> NDArray[np.float32]:
    """Project ground-plane points ``(M, 2)`` in meters to pixel coordinates."""
    pts = np.asarray(
        points_xy_m.detach().cpu().numpy()
        if isinstance(points_xy_m, torch.Tensor)
        else points_xy_m,
        dtype=np.float64,
    )
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise GroundProjectionError(f"points_xy_m must be (M, 2), got {pts.shape}.")
    homogeneous = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    projected = homogeneous @ homography.T
    w = projected[:, 2:3]
    if np.any(np.abs(w) < 1e-9):
        raise GroundProjectionError("Degenerate homography projection (w ~ 0).")
    return np.asarray(projected[:, :2] / w, dtype=np.float32)


__all__ = [
    "GroundProjectionError",
    "default_court_kp_indices",
    "ground_homography_from_court",
    "project_ground_points",
]
