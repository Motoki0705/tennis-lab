from dataclasses import asdict, replace
from unittest.mock import patch

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tennis_scene.dataset_pipeline.geometry import triangulate_ball
from src.tennis_scene.dataset_pipeline.quality import project
from src.tennis_scene.dataset_pipeline.refinement import (
    RefinementSettings,
    check_label_coverage,
    refine_scene,
)
from src.tennis_scene.schema import SceneResult
from src.utils.paths import PROJECT_ROOT


def _scene():
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
        np.zeros((2, frames, 14, 2), np.float32),
        np.ones((2, frames, 14), np.float32),
        roots + 3,
        np.zeros((2, frames), np.float32),
        ball_3d=ball + 3,
        ball_uv=np.stack([project(ball, c)[0] / [640, 360] for c in cameras]),
        ball_vis=np.ones((2, frames), bool),
        human_kp_2d=np.stack(
            [project(joints, c)[0] / [640, 360] for c in cameras], axis=1
        ),
        human_kp_vis=np.ones((2, 2, frames, 17), np.float32),
        player_kp_3d=joints + 3,
        metadata={"reference": {"camera_fits": cameras}},
    )
    assert scene.ball_vis is not None
    scene.ball_vis[:, 1] = False
    settings = RefinementSettings(True, 2, 65, 2, 12, 0, 0.15, 0.5, 0.5, 0.5)
    return scene, roots, joints, ball, settings


def test_refinement_recovers_roots_and_never_trusts_unsupported_ball():
    scene, roots, joints, ball, settings = _scene()
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


@pytest.mark.parametrize("policy", ["hips", "hips_and_shoulders"])
@pytest.mark.parametrize("joint", [5, 6, 11, 12])
@pytest.mark.parametrize("score", [0.299, 0.3])
def test_root_view_support_selects_cameras_without_changing_hip_center(
    policy, joint, score
):
    scene, roots, _, _, settings = _scene()
    # The shoulders are deliberately distinct from the hips: their positions must
    # never enter the root target, even when they gate camera participation.
    scene.human_kp_2d[..., 5:7, :] += 0.15
    scene.human_kp_2d[..., 11, 0] -= 0.01
    scene.human_kp_2d[..., 12, 0] += 0.01
    scene.human_kp_vis[0, 1, :, joint] = score
    settings = replace(settings, player_root_view_support=policy)
    with patch(
        "src.tennis_scene.dataset_pipeline.refinement.triangulate_ball",
        wraps=triangulate_ball,
    ) as triangulate:
        refine_scene(scene, np.stack([np.eye(3), np.eye(3)]), settings)
    root_uv, participating = triangulate.call_args_list[1].args[:2]
    np.testing.assert_allclose(
        root_uv,
        np.stack(
            [
                project(roots[0], c)[0] / [640, 360]
                for c in scene.metadata["reference"]["camera_fits"]
            ]
        ),
    )
    rejected = score < 0.3 and (joint in (11, 12) or policy == "hips_and_shoulders")
    assert participating[0].all()
    assert (participating[1] == (not rejected)).all()
    expected = roots[0] + 3 if rejected else roots[0]
    np.testing.assert_allclose(scene.player_position[0], expected, atol=1e-5)
    assert (
        scene.metadata["label_quality"]["player_source"][0]
        == [0 if rejected else 1] * 5
    )


@pytest.mark.parametrize("value", ["shoulders", "", None, 1, ["hips"]])
def test_unknown_root_support_policy_is_rejected(value):
    settings = _scene()[-1]
    raw = asdict(settings)
    raw["player_root_view_support"] = value
    with pytest.raises(ValueError, match="player_root_view_support"):
        RefinementSettings.from_config(OmegaConf.create(raw))


def test_omitted_root_support_policy_keeps_legacy_hips():
    raw = asdict(_scene()[-1])
    raw.pop("player_root_view_support")
    assert (
        RefinementSettings.from_config(OmegaConf.create(raw)).player_root_view_support
        == "hips"
    )


def test_meiji_v9_and_broadcast_v4_profiles_keep_separate_root_policies():
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs"), version_base="1.3"
    ):
        meiji = compose(config_name="build_slcs_dataset")
        broadcast = compose(config_name="build_broadcast_slcs_dataset")
        report = compose(config_name="report_slcs_dataset_quality")
        assembly = compose(config_name="assemble_slcs_dataset")
    assert (
        RefinementSettings.from_config(meiji.refinement).player_root_view_support
        == "hips_and_shoulders"
    )
    assert (
        RefinementSettings.from_config(broadcast.refinement).player_root_view_support
        == "hips"
    )
    # Omission also preserves the already-generated broadcast recipe identity.
    assert "player_root_view_support" not in broadcast.refinement
    assert broadcast.dataset_output_directory == "slcs/broadcast_rgb_v4"
    assert (
        meiji.dataset_output_directory
        == report.dataset_directory
        == "slcs/meiji_rgb_v9"
    )
    assert report.generation_directories == [meiji.output_dir]
    assert meiji.observation_directory.endswith("meiji_dino_vitpose/s42-005")
    assert meiji.court.crop_refinement_padding_px == 20.0
    assert "crop_refinement_padding_px" not in broadcast.court
    assert list(assembly.source_datasets) == [
        "slcs/meiji_rgb_v9",
        "slcs/broadcast_rgb_v4",
    ]
