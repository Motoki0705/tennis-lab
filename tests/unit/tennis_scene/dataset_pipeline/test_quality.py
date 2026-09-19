"""Known projections detect wrong 3D labels and exclude invisible observations."""

import numpy as np
import pytest

from src.tennis_scene.dataset_pipeline.quality import (
    evaluate_reconstruction,
    project,
    summarize,
)
from src.tennis_scene.schema import SceneResult


def test_projection_quality_is_zero_for_consistent_labels_and_detects_shift():
    frames = 4
    cameras = [
        {
            "K": [[500.0, 0, 320], [0, 500, 180], [0, 0, 1]],
            "R": np.eye(3).tolist(),
            "t": [offset, 0, 20.0],
            "rmse_px": 0.0,
        }
        for offset in (0.0, 2.0)
    ]
    homographies = np.array(
        [[[25, 0, 320 + 25 * c["t"][0]], [0, 25, 180], [0, 0, 1]] for c in cameras]
    )
    positions = np.broadcast_to(
        np.array([[[-2.0, 1.0, 1.0]], [[3.0, -4.0, 1.0]]], dtype=np.float32), (2, frames, 3)
    ).copy()
    joints = np.repeat(positions[:, :, None], 17, axis=2)
    joints[:, :, [15, 16], 2] = 0
    ball = np.broadcast_to(np.array([1.0, 2.0, 1.0], dtype=np.float32), (frames, 3)).copy()
    ball_uv = np.stack([project(ball, c)[0] / [640, 360] for c in cameras]).astype(np.float32)
    human_uv = np.stack([project(joints, c)[0] / [640, 360] for c in cameras], axis=1).astype(np.float32)
    ball_vis: np.ndarray = np.ones((2, frames), bool)
    ball_vis[0, 0] = False
    ball_uv[0, 0] = 100  # Excluded observation must never inflate an error.
    scene = SceneResult(
        frames,
        30.0,
        640,
        360,
        np.zeros((2, frames, 14, 2), dtype=np.float32),
        np.ones((2, frames, 14), dtype=np.float32),
        positions,
        np.zeros((2, frames), dtype=np.float32),
        ball_uv=ball_uv,
        ball_vis=ball_vis,
        ball_3d=ball,
        human_kp_2d=human_uv,
        human_kp_vis=np.ones((2, 2, frames, 17), dtype=np.float32),
        player_kp_3d=joints,
        metadata={
            "reference": {"camera_fits": cameras, "camera_ids": ["cam0", "cam1"]}
        },
    )
    report, _ = evaluate_reconstruction(scene, homographies, [False, False])
    assert report["blcs"]["reprojection_px"]["count"] == 7
    # Normalized observations are stored as float32; bound their pixel rounding.
    pixel_tolerance = float(np.finfo(np.float32).eps) * max(scene.width, scene.height)
    assert report["blcs"]["reprojection_px"]["max"] == pytest.approx(0, abs=pixel_tolerance)
    assert report["plcs"]["pose_reprojection_px"]["max"] == pytest.approx(0, abs=pixel_tolerance)
    assert report["plcs"]["ground_foot_to_root_xy_m"]["max"] == pytest.approx(
        0, abs=1e-6
    )
    assert scene.ball_3d is not None
    scene.ball_3d[:, 0] += 2
    shifted, _ = evaluate_reconstruction(scene, homographies, [False, False])
    assert shifted["blcs"]["reprojection_px"]["median"] > 40


def test_unavailable_measurements_are_explicitly_null():
    assert summarize(np.array([np.nan, np.inf])) == {
        "count": 0,
        "mean": None,
        "median": None,
        "p95": None,
        "max": None,
    }
