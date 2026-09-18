import numpy as np

from src.tennis_scene.dataset_pipeline.quality import project
from src.tennis_scene.dataset_pipeline.refinement import (
    RefinementSettings,
    check_label_coverage,
    refine_scene,
)
from src.tennis_scene.schema import SceneResult


def test_refinement_recovers_roots_and_never_trusts_unsupported_ball():
    frames = 5
    cameras = [
        {
            "R": np.eye(3).tolist(),
            "t": [offset, 0, 10],
            "K": [[500, 0, 320], [0, 500, 180], [0, 0, 1]],
        }
        for offset in (-3, 3)
    ]
    roots = np.broadcast_to(
        np.array([[[0, -2, 0.8]], [[0, 2, 0.9]]], np.float32), (2, frames, 3)
    ).copy()
    ball = np.broadcast_to([0.5, 0, 1.0], (frames, 3)).copy().astype(np.float32)
    joints = np.repeat(roots[:, :, None], 17, axis=2)
    scene = SceneResult(
        frames,
        1.0,
        640,
        360,
        np.zeros((2, frames, 14, 2)),
        np.ones((2, frames, 14)),
        roots + 3,
        np.zeros((2, frames)),
        ball_3d=ball + 3,
        ball_uv=np.stack([project(ball, c)[0] / [640, 360] for c in cameras]),
        ball_vis=np.ones((2, frames), bool),
        human_kp_2d=np.stack(
            [project(joints, c)[0] / [640, 360] for c in cameras], axis=1
        ),
        human_kp_vis=np.ones((2, 2, frames, 17)),
        player_kp_3d=joints + 3,
        metadata={"reference": {"camera_fits": cameras}},
    )
    scene.ball_vis[:, 1] = False
    settings = RefinementSettings(True, 2, 65, 2, 12, 0, 0.15, 0.5, 0.5, 0.5)
    evidence = refine_scene(scene, np.stack([np.eye(3), np.eye(3)]), settings)
    np.testing.assert_allclose(scene.player_position, roots, atol=1e-5)
    np.testing.assert_allclose(scene.player_kp_3d, joints, atol=1e-5)
    np.testing.assert_allclose(
        scene.ball_3d[[0, 2, 3, 4]], ball[[0, 2, 3, 4]], atol=1e-5
    )
    assert scene.metadata["label_quality"]["ball_weight"][1] == 0
    assert scene.metadata["label_quality"]["ball_source"][1] == 0
    assert scene.metadata["label_quality"]["is_ground_truth"] is False
    check_label_coverage(evidence, settings)
