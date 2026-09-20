"""Observable consistency diagnostics, explicitly separate from 3D accuracy."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.tennis_scene.schema import SceneResult


def summarize(values: np.ndarray) -> dict[str, float | int | None]:
    valid = np.asarray(values)[np.isfinite(values)]
    if not valid.size:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(valid.size),
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "p95": float(np.percentile(valid, 95)),
        "max": float(np.max(valid)),
    }


def project(
    points: np.ndarray, camera: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Project court-space metres with the audited approximate pinhole camera."""
    rotation, translation, intrinsic = (
        np.asarray(camera[key], np.float64) for key in ("R", "t", "K")
    )
    local = points @ rotation.T + translation
    pixels = local @ intrinsic.T
    depth = local[..., 2]
    valid = np.isfinite(pixels).all(-1) & (depth > 1e-6)
    output = np.full(pixels.shape[:-1] + (2,), np.nan)
    np.divide(pixels[..., :2], pixels[..., 2:3], out=output, where=valid[..., None])
    return output, valid


def evaluate_reconstruction(
    scene: SceneResult, homographies: np.ndarray, half_turns: list[bool]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Report reprojection, foot-ground consistency, coverage and motion tails.

    Calibration uses learned planar court points. None of the 3D comparisons
    here is an independent ground-truth accuracy measurement.
    """
    if any(
        value is None
        for value in (
            scene.ball_uv,
            scene.ball_vis,
            scene.ball_3d,
            scene.human_kp_2d,
            scene.human_kp_vis,
            scene.player_kp_3d,
        )
    ):
        raise ValueError(
            "Full PLCS/BLCS/pose observations are required for quality evaluation"
        )
    assert (
        scene.ball_uv is not None
        and scene.ball_vis is not None
        and scene.ball_3d is not None
    )
    assert (
        scene.human_kp_2d is not None
        and scene.human_kp_vis is not None
        and scene.player_kp_3d is not None
    )
    fits = scene.metadata["reference"]["camera_fits"]
    cameras = scene.metadata["reference"]["camera_ids"]
    size = np.asarray([scene.width, scene.height])
    feet = scene.human_kp_2d[..., [15, 16], :].mean(-2) * size
    feet_valid = np.asarray(
        (scene.human_kp_vis[..., [15, 16]] >= 0.3).all(-1), dtype=bool
    )
    ground = np.full(feet.shape, np.nan)
    ball_error = np.full(scene.ball_vis.shape, np.nan)
    pose_error = np.full(scene.human_kp_vis.shape, np.nan)
    for view, (camera, matrix, turn) in enumerate(
        zip(fits, homographies, half_turns, strict=True)
    ):
        points = cv2.perspectiveTransform(
            feet[:, view].reshape(1, -1, 2).astype(np.float64), np.linalg.inv(matrix)
        ).reshape(feet[:, view].shape)
        if turn:
            points *= -1
        ground[:, view] = np.where(feet_valid[:, view, :, None], points, np.nan)
        projected, in_front = project(scene.ball_3d, camera)
        valid = scene.ball_vis[view] & in_front
        ball_error[view] = np.where(
            valid,
            np.linalg.norm(projected - scene.ball_uv[view] * size, axis=-1),
            np.nan,
        )
        projected, in_front = project(scene.player_kp_3d, camera)
        valid = (scene.human_kp_vis[:, view] >= 0.3) & in_front
        pose_error[:, view] = np.where(
            valid,
            np.linalg.norm(projected - scene.human_kp_2d[:, view] * size, axis=-1),
            np.nan,
        )
    # Hip/root XY is an approximate foot-ground anchor, not identical anatomy.
    foot_error = np.linalg.norm(ground - scene.player_position[:, None, :, :2], axis=-1)
    pair_errors = [
        np.linalg.norm(ground[:, a] - ground[:, b], axis=-1)
        for a in range(len(cameras))
        for b in range(a + 1, len(cameras))
    ]
    speed_player = (
        np.linalg.norm(np.diff(scene.player_position, axis=1), axis=-1) * scene.fps
    )
    speed_ball = np.linalg.norm(np.diff(scene.ball_3d, axis=0), axis=-1) * scene.fps
    report = {
        "schema_version": 1,
        "interpretation": "observable consistency of pseudo labels; NOT measured 3D accuracy",
        "calibration": "learned planar court, approximate pinhole, no measured lens distortion",
        "plcs": {
            "ground_foot_to_root_xy_m": summarize(foot_error),
            "pose_reprojection_px": summarize(pose_error),
            "foot_ground_cross_view_distance_m": summarize(
                np.stack(pair_errors) if pair_errors else np.empty(0)
            ),
            "speed_mps": summarize(speed_player),
        },
        "blcs": {
            "reprojection_px": summarize(ball_error),
            "speed_mps": summarize(speed_ball),
            "negative_height_fraction": float(np.mean(scene.ball_3d[:, 2] < -0.1)),
            "height_m": summarize(scene.ball_3d[:, 2]),
            "observed_in_two_views_fraction": float(
                np.mean(scene.ball_vis.sum(0) >= 2)
            ),
        },
        "per_camera": {
            camera: {
                "ball_reprojection_px": summarize(ball_error[view]),
                "pose_reprojection_px": summarize(pose_error[:, view]),
                "ground_foot_to_root_xy_m": summarize(foot_error[:, view]),
                "ball_coverage": float(scene.ball_vis[view].mean()),
                "pose_coverage_at_0.3": float(
                    (scene.human_kp_vis[:, view] >= 0.3).mean()
                ),
                "court_fit_rmse_px": fits[view]["rmse_px"],
            }
            for view, camera in enumerate(cameras)
        },
    }
    return report, {
        "ball_reprojection_px": ball_error,
        "pose_reprojection_px": pose_error,
        "ground_foot_to_root_xy_m": foot_error,
        "foot_ground_xy_m": ground,
        "player_speed_mps": speed_player,
        "ball_speed_mps": speed_ball,
    }
