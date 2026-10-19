"""Exercise scene archive -> source video -> Blender bundle without Blender."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.tennis_scene.archive import load_scene_result, save_scene_result
from src.tennis_scene.schema import SceneResult
from src.tennis_scene.visualization.blender.export.bundle import (
    BlenderBundleConfig,
    export_blender_bundle,
)
from src.tennis_scene.visualization.blender.export.pointcache import read_pc2
from src.utils.video.writer import save_video_rgb

pytestmark = pytest.mark.integration


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    frames, views = 7, 2
    width, height = 64, 48
    video = tmp_path / "cam1.mp4"
    rgb: np.ndarray = np.zeros((frames, height, width, 3), dtype=np.uint8)
    rgb[..., 1] = np.arange(frames, dtype=np.uint8)[:, None, None] * 30
    save_video_rgb(rgb, video, fps=30.0)
    K = np.array([[50.0, 0, 32.0], [0, 50.0, 24.0], [0, 0, 1.0]])
    R = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    t = np.array([0.0, 3.0, 20.0])
    body_valid = np.array([[False, True, True, False, True, False, False]])
    ball_valid = np.array([True, True, False, True, False, False, True])
    positions = np.tile([2.0, -3.0, 1.0], (1, frames, 1)).astype(np.float32)
    positions[~body_valid] = 0
    vertices = np.tile([[0, 0, 0], [1, 0, 0], [0, 1, 0]], (1, frames, 1, 1)).astype(
        np.float32
    )
    vertices[~body_valid] = 0
    ball = np.stack(
        [np.arange(frames) / 10, np.zeros(frames), np.ones(frames)], axis=1
    ).astype(np.float32)
    cam = ball @ R.T + t
    uv = (cam[:, :2] / cam[:, 2:] @ K[:2, :2].T + K[:2, 2]) / [width, height]
    ball[~ball_valid] = 0
    fits = [{"K": K.tolist(), "R": R.tolist(), "t": t.tolist()}] * views
    scene = SceneResult(
        num_frames=frames,
        fps=30.0,
        width=width,
        height=height,
        court_kp=np.zeros((views, frames, 14, 2), np.float32),
        court_vis=np.ones((views, frames, 14), np.float32),
        player_position=positions,
        player_yaw=np.zeros((1, frames), np.float32),
        smpl_body_pose=np.zeros((1, frames, 63), np.float32),
        smpl_global_orient=np.zeros((1, frames, 3), np.float32),
        smpl_vertices_local=vertices,
        player_kp_3d=np.zeros((1, frames, 17, 3), np.float32),
        human_kp_2d=np.zeros((1, views, frames, 17, 2), np.float32),
        human_kp_vis=np.zeros((1, views, frames, 17), np.float32),
        ball_3d=ball,
        ball_uv=np.tile(uv, (views, 1, 1)).astype(np.float32),
        ball_vis=np.ones((views, frames), bool),
        player_track_ids=np.array([7], np.int32),
        player_observed=body_valid.copy(),
        player_valid=body_valid.copy(),
        player_heading_valid=body_valid.copy(),
        player_smpl_valid=body_valid.copy(),
        player_kp_3d_vis=np.zeros((1, frames, 17), bool),
        ball_3d_valid=ball_valid,
        player_rejection_code=(~body_valid).astype(np.uint8),
        player_kp_3d_rejection_code=np.ones((1, frames, 17), np.uint8),
        ball_rejection_code=(~ball_valid).astype(np.uint8),
        metadata={
            "scene_schema_version": 2,
            "camera_ids": ["cam0", "cam1"],
            "video_paths": [str(video), str(video)],
            "court_reference": {"camera_ids": ["cam0", "cam1"], "camera_fits": fits},
        },
    )
    scene_path = tmp_path / "scene.npz"
    save_scene_result(scene, scene_path)
    faces = tmp_path / "faces.npz"
    regressor = tmp_path / "regressor.pt"
    np.savez(faces, f=np.array([[0, 1, 2]], np.int64))
    torch.save(torch.tensor([[1.0, 0, 0]]), regressor)
    return scene_path, faces, regressor


def _config() -> BlenderBundleConfig:
    return BlenderBundleConfig(
        camera_id="cam1",
        player_max_gap_frames=1,
        ball_max_gap_frames=1,
        max_ball_reprojection_median_px=8.0,
        sensor_width_mm=36.0,
        plate_jpeg_quality=95,
    )


def _export(inputs: tuple[Path, Path, Path], output: Path) -> Path:
    scene, faces, regressor = inputs
    return export_blender_bundle(
        scene_path=scene,
        smpl_faces_path=faces,
        smpl_joint_regressor_path=regressor,
        output_dir=output,
        config=_config(),
    )


def test_v2_bundle_preserves_timeline_geometry_and_explicit_gaps(
    inputs: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    source_bytes = inputs[0].read_bytes()
    output = tmp_path / "bundle"
    manifest = json.loads(_export(inputs, output).read_text())
    assert inputs[0].read_bytes() == source_bytes
    assert (manifest["num_frames"], manifest["fps"]) == (7, 30.0)
    assert manifest["source"]["camera_id"] == "cam1"
    assert manifest["camera"]["ball_reprojection_px"]["median"] < 1e-4
    assert len(list((output / "plate").glob("*.jpg"))) == 7
    assert (output / "plate/00000.jpg").is_file()
    assert (output / "plate/00006.jpg").is_file()
    assert manifest["players"][0]["filled_frames"] == [3]
    assert manifest["players"][0]["invalid_frames"] == [0, 5, 6]
    assert manifest["ball"]["filled_frames"] == [2]
    assert manifest["ball"]["invalid_frames"] == [4, 5]
    vertices, start, rate = read_pc2(output / "player_0.pc2")
    assert (start, rate) == (0.0, 1.0)
    np.testing.assert_allclose(vertices[3], [[2, -3, 1], [3, -3, 1], [2, -3, 2]])
    assert (vertices[[0, 5, 6]] == 0).all()
    np.testing.assert_array_equal(
        np.load(output / "player_valid.npy"),
        [[False, True, True, True, True, False, False]],
    )
    balls = np.load(output / "ball_positions.npy")
    np.testing.assert_allclose(balls[2], [0.2, 0, 1])
    assert np.isnan(balls[[4, 5]]).all()
    with pytest.raises(FileExistsError):
        _export(inputs, output)


def test_rejects_camera_mismatch_before_publishing(
    inputs: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    scene = load_scene_result(inputs[0])
    assert scene.ball_uv is not None
    scene.ball_uv[1, :, 0] += 0.5
    save_scene_result(scene, inputs[0])
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="reprojection error"):
        _export(inputs, output)
    assert not output.exists()


def test_rejects_video_fps_mismatch_and_removes_partial_bundle(
    inputs: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    scene = replace(load_scene_result(inputs[0]), fps=60.0)
    save_scene_result(scene, inputs[0])
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="Video FPS"):
        _export(inputs, output)
    assert not output.exists()


@pytest.mark.parametrize("video_frames", [6, 8])
def test_rejects_video_length_mismatch_and_removes_partial_bundle(
    inputs: tuple[Path, Path, Path], tmp_path: Path, video_frames: int
) -> None:
    scene = load_scene_result(inputs[0])
    video = Path(scene.metadata["video_paths"][1])
    save_video_rgb(np.zeros((video_frames, 48, 64, 3), dtype=np.uint8), video, fps=30.0)
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="scene has 7"):
        _export(inputs, output)
    assert not output.exists()


def test_rejects_legacy_scene_without_inferring_masks(
    inputs: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    from src.tennis_scene.schema import SCENE_MASK_FIELDS, SCENE_REASON_FIELDS

    scene = load_scene_result(inputs[0])
    scene.metadata.pop("scene_schema_version")
    for name in (*SCENE_MASK_FIELDS, *SCENE_REASON_FIELDS):
        setattr(scene, name, None)
    save_scene_result(scene, inputs[0])
    with pytest.raises(ValueError, match="requires SceneResult v2"):
        _export(inputs, tmp_path / "bundle")
