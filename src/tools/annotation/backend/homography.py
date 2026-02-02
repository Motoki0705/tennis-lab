"""Court homography completion helpers."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from src.tools.annotation.backend.models import CourtFrameAnnotation, CourtKeypoint
from src.utils.geometry.court import court_keypoints_3d

_TOP_BASE_IDX: dict[int, int] = {
    16: 15,  # left_post_top -> left_post_base
    18: 17,  # right_post_top -> right_post_base
    19: 14,  # center_strap_top -> net_center
}


def _world_xy_for_index(kp_xyz: np.ndarray, idx: int) -> np.ndarray:
    if idx in _TOP_BASE_IDX:
        return kp_xyz[_TOP_BASE_IDX[idx], :2]
    return kp_xyz[idx, :2]


def _collect_manual_ground_points(
    ann: CourtFrameAnnotation,
    kp_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    world_pts: list[list[float]] = []
    image_pts: list[list[float]] = []
    for i, kp in enumerate(ann.keypoints):
        if kp.visibility == 0 or kp.source != "manual":
            continue
        if abs(float(kp_xyz[i, 2])) > 1e-6:
            continue
        world_pts.append([float(kp_xyz[i, 0]), float(kp_xyz[i, 1])])
        image_pts.append([float(kp.x_px), float(kp.y_px)])
    return np.array(world_pts, dtype=np.float32), np.array(image_pts, dtype=np.float32)


def _project_points(h: np.ndarray, xy: Iterable[float]) -> tuple[float, float]:
    pts = np.array([[list(xy)]], dtype=np.float32)
    projected = cv2.perspectiveTransform(pts, h)
    return float(projected[0, 0, 0]), float(projected[0, 0, 1])


def fill_court_keypoints_from_homography(
    ann: CourtFrameAnnotation,
) -> CourtFrameAnnotation:
    """Fill missing court keypoints using homography from manual ground points.

    Args:
        ann: Current court annotation (manual keypoints required).

    Returns:
        Updated annotation with homography-filled keypoints.

    Raises:
        ValueError: If not enough manual points or homography fails.

    """
    kp_xyz = court_keypoints_3d().numpy()
    world_pts, image_pts = _collect_manual_ground_points(ann, kp_xyz)
    if len(world_pts) < 4:
        raise ValueError("need >=4 manual ground keypoints for homography")

    h, _ = cv2.findHomography(world_pts, image_pts, method=0)
    if h is None:
        raise ValueError("failed to estimate homography")

    next_keypoints: list[CourtKeypoint] = []
    for i, kp in enumerate(ann.keypoints):
        should_update = kp.visibility == 0 or kp.source == "homography"
        if not should_update:
            next_keypoints.append(kp)
            continue

        if i in _TOP_BASE_IDX:
            base_idx = _TOP_BASE_IDX[i]
            base_kp = ann.keypoints[base_idx]
            if base_kp.visibility > 0:
                x_px, y_px = float(base_kp.x_px), float(base_kp.y_px)
            else:
                xy_world = _world_xy_for_index(kp_xyz, i)
                x_px, y_px = _project_points(h, xy_world)
        else:
            xy_world = _world_xy_for_index(kp_xyz, i)
            x_px, y_px = _project_points(h, xy_world)
        next_keypoints.append(
            CourtKeypoint(
                x_px=x_px,
                y_px=y_px,
                visibility=1,
                source="homography",
            )
        )

    return CourtFrameAnnotation(frame_idx=ann.frame_idx, keypoints=next_keypoints)
