from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.tennis_scene.io import SceneResult
from src.tennis_scene.rendering.debug_visualization import (
    DebugVisualizationConfig,
    save_intermediate_visualizations,
)


def _write_test_video(path: Path, *, width: int, height: int, num_frames: int) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (width, height),
    )
    assert writer.isOpened()
    try:
        for frame_idx in range(num_frames):
            frame = np.full((height, width, 3), 32, dtype=np.uint8)
            cv2.rectangle(frame, (10, 10), (width - 10, height - 10), (60, 120, 60), -1)
            cv2.putText(
                frame,
                f"F{frame_idx}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()


def _assert_video(path: Path) -> None:
    assert path.exists()
    assert path.stat().st_size > 0
    cap = cv2.VideoCapture(str(path))
    try:
        assert cap.isOpened()
        ret, frame = cap.read()
        assert ret
        assert frame is not None
        assert frame.std() > 0
    finally:
        cap.release()


def _make_scene(*, width: int, height: int, num_frames: int) -> SceneResult:
    court_kp = np.stack(
        [
            np.linspace(0.1, 0.9, 14, dtype=np.float32),
            np.linspace(0.15, 0.85, 14, dtype=np.float32),
        ],
        axis=-1,
    )
    human_kp = np.zeros((2, num_frames, 17, 2), dtype=np.float32)
    for player_idx in range(2):
        for frame_idx in range(num_frames):
            human_kp[player_idx, frame_idx, :, 0] = 0.25 + 0.35 * player_idx
            human_kp[player_idx, frame_idx, :, 1] = np.linspace(
                0.2,
                0.8,
                17,
                dtype=np.float32,
            )
            human_kp[player_idx, frame_idx, :, 0] += frame_idx * 0.02

    player_position = np.zeros((2, num_frames, 3), dtype=np.float32)
    player_position[0, :, 0] = np.linspace(-2.0, -1.0, num_frames, dtype=np.float32)
    player_position[0, :, 1] = -4.0
    player_position[1, :, 0] = np.linspace(2.0, 1.0, num_frames, dtype=np.float32)
    player_position[1, :, 1] = 4.0

    return SceneResult(
        num_frames=num_frames,
        fps=5.0,
        width=width,
        height=height,
        court_kp=court_kp,
        court_vis=np.ones(14, dtype=np.float32),
        player_position=player_position,
        player_yaw=np.zeros((2, num_frames), dtype=np.float32),
        smpl_body_pose=np.zeros((2, num_frames, 63), dtype=np.float32),
        smpl_global_orient=np.zeros((2, num_frames, 3), dtype=np.float32),
        smpl_betas=np.zeros((2, 10), dtype=np.float32),
        ball_uv=np.array(
            [[0.2, 0.3], [0.4, 0.4], [0.6, 0.5]],
            dtype=np.float32,
        ),
        ball_visibility=np.array([True, False, True], dtype=np.bool_),
        ball_3d=np.array(
            [[-1.0, -2.0, 1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=np.float32,
        ),
        human_kp_2d=human_kp,
        human_kp_vis=np.ones((2, num_frames, 17), dtype=np.float32),
        player_track_ids=np.array([10, 11], dtype=np.int32),
    )


def test_save_intermediate_visualizations_writes_videos_and_manifest(tmp_path: Path) -> None:
    width = 160
    height = 96
    num_frames = 3
    video_path = tmp_path / "input.mp4"
    _write_test_video(video_path, width=width, height=height, num_frames=num_frames)
    scene = _make_scene(width=width, height=height, num_frames=num_frames)

    manifest = save_intermediate_visualizations(
        scene,
        video_path,
        DebugVisualizationConfig(output_dir=tmp_path / "debug", fps=5.0),
    )

    expected_names = {
        "court_kp_overlay",
        "ball_2d_overlay",
        "blcs_input_overlay",
        "human_kp_overlay",
        "plcs_court_view",
    }
    assert set(manifest.saved) == expected_names
    assert manifest.skipped == {}
    for path in manifest.saved.values():
        _assert_video(path)

    with manifest.manifest_path.open("r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    assert set(manifest_data["saved"]) == expected_names
    assert manifest_data["skipped"] == {}
    assert manifest_data["num_frames"] == num_frames


def test_save_intermediate_visualizations_records_skipped_outputs(tmp_path: Path) -> None:
    width = 160
    height = 96
    num_frames = 3
    video_path = tmp_path / "input.mp4"
    _write_test_video(video_path, width=width, height=height, num_frames=num_frames)
    scene = _make_scene(width=width, height=height, num_frames=num_frames)
    scene.ball_uv = None
    scene.human_kp_2d = None

    manifest = save_intermediate_visualizations(
        scene,
        video_path,
        DebugVisualizationConfig(output_dir=tmp_path / "debug", fps=5.0),
    )

    assert "court_kp_overlay" in manifest.saved
    assert "plcs_court_view" in manifest.saved
    assert manifest.skipped["ball_2d_overlay"] == "scene.ball_uv is missing"
    assert manifest.skipped["blcs_input_overlay"] == "scene.ball_uv or scene.court_kp is missing"
    assert manifest.skipped["human_kp_overlay"] == "scene.human_kp_2d is missing"