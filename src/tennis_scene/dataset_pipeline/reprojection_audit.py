"""Image-support diagnostics for pseudo-label consistency, not measured accuracy."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.tennis_scene.dataset_pipeline.quality import project, summarize
from src.tennis_scene.schema import SceneResult


def pose_reprojection_diagnostics(
    scene: SceneResult, positive_weight: np.ndarray
) -> dict[str, Any]:
    """Summarize confident joint residuals and hip-center/root residuals by view/player.

    Image support partitions the joint diagnostic only; it never changes teacher
    weights or establishes whether a pseudo-label is correct. Camera fits follow
    the explicit reference camera_ids order, as do the observation view axes.
    """
    positions = np.asarray(scene.player_position)
    if positions.ndim != 3 or positions.shape[1:] != (scene.num_frames, 3):
        raise ValueError("player_position must have shape (P,T,3)")
    players, frames = positions.shape[:2]
    mask = np.asarray(positive_weight)
    if mask.shape != (players, frames) or not np.isfinite(mask).all():
        raise ValueError("positive_weight must be finite with shape (P,T)")
    if (mask < 0).any():
        raise ValueError("positive_weight must be nonnegative")
    mask = mask > 0
    if scene.width <= 0 or scene.height <= 0:
        raise ValueError("Image dimensions must be positive")
    reference = scene.metadata.get("reference", {})
    cameras = reference.get("camera_ids")
    fits = reference.get("camera_fits")
    if (
        not isinstance(cameras, list)
        or not cameras
        or any(not isinstance(camera, str) for camera in cameras)
        or len(set(cameras)) != len(cameras)
        or not isinstance(fits, list)
        or len(fits) != len(cameras)
    ):
        raise ValueError(
            "Explicit unique camera_ids and equally ordered camera_fits required"
        )
    views = len(cameras)
    if scene.court_kp.shape[:2] != (views, frames):
        raise ValueError("Court view/frame axes disagree with camera_ids")
    observations = scene.human_kp_2d
    confidence = scene.human_kp_vis
    joints = scene.player_kp_3d
    if observations is None or confidence is None or joints is None:
        raise ValueError("2D observations, confidence and 3D joints are required")
    if observations.shape != (players, views, frames, 17, 2):
        raise ValueError("human_kp_2d must have shape (P,V,T,17,2) in camera_ids order")
    if confidence.shape != (players, views, frames, 17):
        raise ValueError("human_kp_vis must have shape (P,V,T,17)")
    if joints.shape != (players, frames, 17, 3):
        raise ValueError("player_kp_3d must have shape (P,T,17,3)")
    size = np.asarray([scene.width, scene.height])
    per_camera = {}
    for view, (camera_id, camera) in enumerate(zip(cameras, fits, strict=True)):
        for name, shape in (("K", (3, 3)), ("R", (3, 3)), ("t", (3,))):
            value = np.asarray(camera.get(name), dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(
                    f"Invalid {camera_id} camera {name}: expected finite {shape}"
                )
        projected, in_front = project(joints, camera)
        observed = observations[:, view] * size
        residual = np.linalg.norm(projected - observed, axis=-1)
        valid = (
            (confidence[:, view] >= 0.3)
            & in_front
            & mask[..., None]
            & np.isfinite(residual)
        )
        inside = ((observed >= 0) & (observed < size)).all(-1) & (
            (projected >= 0) & (projected < size)
        ).all(-1)
        root_projected, root_in_front = project(positions, camera)
        hips = observed[..., [11, 12], :].mean(-2)
        root_residual = np.linalg.norm(root_projected - hips, axis=-1)
        root_valid = (
            (confidence[:, view][..., [11, 12]] >= 0.3).all(-1)
            & root_in_front
            & mask
            & np.isfinite(root_residual)
        )
        per_camera[camera_id] = {
            str(player): {
                "all_confident": summarize(residual[player][valid[player]]),
                "both_inside_image": summarize(
                    residual[player][valid[player] & inside[player]]
                ),
                "either_outside_image": summarize(
                    residual[player][valid[player] & ~inside[player]]
                ),
                "root_to_observed_hip_center_px": summarize(
                    root_residual[player][root_valid[player]]
                ),
            }
            for player in range(players)
        }
    return {
        "interpretation": "positive-weight pseudo-label consistency; image support is diagnostic only; NOT measured 3D accuracy",
        "per_camera": per_camera,
    }


def audit_reprojection_stages(
    raw: SceneResult, refined: SceneResult, positive_weight: np.ndarray
) -> dict[str, Any]:
    """Apply the same final training eligibility to both reconstruction stages."""
    if raw.metadata.get("reference", {}).get("camera_ids") != refined.metadata.get(
        "reference", {}
    ).get("camera_ids") or (raw.width, raw.height, raw.num_frames) != (
        refined.width,
        refined.height,
        refined.num_frames,
    ):
        raise ValueError("Raw/refined camera order or image/frame dimensions differ")
    return {
        "raw": pose_reprojection_diagnostics(raw, positive_weight),
        "refined": pose_reprojection_diagnostics(refined, positive_weight),
    }
