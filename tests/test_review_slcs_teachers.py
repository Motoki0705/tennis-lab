from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.tasks.slcs.data.quality import QualityConfig
from src.tennis_scene.dataset_pipeline.teacher_review import (
    contact_sheet,
    label_masks,
    select_frames,
    trajectory_plot,
    validate_scene,
)
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.tennis_scene.schema import SceneResult


def fixture(tmp_path: Path) -> tuple[SceneResult, ClipManifest]:
    t = 9
    cameras = ("a", "b", "c")
    scene = SceneResult(
        num_frames=t,
        fps=30,
        width=64,
        height=48,
        court_kp=np.zeros((3, t, 20, 2), np.float32),
        court_vis=np.ones((3, t, 20), np.float32),
        player_position=np.zeros((1, t, 3), np.float32),
        player_yaw=np.zeros((1, t), np.float32),
        ball_3d=np.zeros((t, 3), np.float32),
        ball_uv=np.zeros((3, t, 2), np.float32),
        ball_vis=np.ones((3, t), bool),
        human_kp_2d=np.zeros((1, 3, t, 17, 2), np.float32),
        human_kp_vis=np.ones((1, 3, t, 17), np.float32),
        metadata={
            "reference": {
                "camera_ids": list(cameras),
                "camera_fits": [
                    {"R": np.eye(3).tolist(), "t": [0, 0, 1], "K": np.eye(3).tolist()}
                    for _ in cameras
                ],
            },
            "label_quality": {
                "schema_version": 1,
                "is_ground_truth": False,
                "ball_weight": [1, 0, 1, 1, 1, 1, 1, 1, 1],
                "player_weight": [[1] * t],
            },
        },
    )
    clip = ClipManifest(
        tmp_path,
        "test",
        "v/c",
        "v",
        "c",
        30,
        t,
        64,
        48,
        cameras,
        tuple(f"{c}.avi" for c in cameras),
        (),
    )
    return scene, clip


def test_selection_uses_same_positive_weight_mask_and_frame_axis(
    tmp_path: Path,
) -> None:
    raw, clip = fixture(tmp_path)
    refined = deepcopy(raw)
    assert raw.ball_3d is not None and refined.ball_3d is not None
    raw.ball_3d[1, 0] = 1000  # rejected by teacher mask
    raw.ball_3d[3, 0] = 100
    refined.ball_3d[5, 0] = 90
    refined.player_position[0, 7, 0] = 10
    masks = label_masks(refined, QualityConfig(0.3, 1, 1, 0.5))
    selected = select_frames(raw, refined, masks)
    assert "max_raw_ball_residual_on_positive_teacher_weight" in selected[3]
    assert "max_refined_ball_residual_on_positive_teacher_weight" in selected[5]
    assert "minimum_total_teacher_weight" in selected[1]
    assert "max_raw_to_refined_root_displacement" in selected[7]
    assert "max_refined_root_frame_step" in selected[7]
    validate_scene(refined, clip)


def test_validation_and_zero_weight(tmp_path: Path) -> None:
    scene, clip = fixture(tmp_path)
    masks = label_masks(scene, QualityConfig(0.3, 1, 1, 0.5))
    assert not masks["ball_label_valid"][1]
    scene.metadata["reference"]["camera_fits"][0]["t"] = [[0, 0, 1]]
    with pytest.raises(ValueError, match="reference camera t"):
        validate_scene(scene, clip)
    scene.metadata["reference"]["camera_fits"][0]["t"] = [0, 0, 1]
    scene.metadata["reference"]["camera_ids"].reverse()
    with pytest.raises(ValueError, match="camera order"):
        validate_scene(scene, clip)


def test_cpu_fixture_images(tmp_path: Path) -> None:
    scene, clip = fixture(tmp_path)
    for camera in clip.camera_ids:
        writer = cv2.VideoWriter(
            str(clip.media_path(camera, must_exist=False)),
            cv2.VideoWriter.fourcc(*"MJPG"),
            30,
            (64, 48),
        )
        assert writer.isOpened()
        try:
            for _ in range(scene.num_frames):
                writer.write(np.zeros((48, 64, 3), np.uint8))
        finally:
            writer.release()
    masks = label_masks(scene, QualityConfig(0.3, 1, 1, 0.5))
    contact_sheet(
        clip,
        scene,
        scene,
        masks,
        {0: ["test"], 1: ["unsupported"]},
        tmp_path / "sheet.png",
    )
    trajectory_plot(scene, scene, masks, tmp_path / "trajectory.png")
    for name in ("sheet.png", "trajectory.png"):
        image = cv2.imread(str(tmp_path / name))
        assert image is not None and image.shape[0] > 100 and image.shape[1] > 100
