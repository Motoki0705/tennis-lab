"""Verify the rendered trail cannot connect opposite sides of missing evidence."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from src.tennis_scene.scripts.visualize_tasks import _render_ball_detection


def test_ball_overlay_breaks_trail_at_missing_source_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.tennis_scene.schema import SceneResult

    frames: list[np.ndarray] = [np.zeros((100, 100, 3), np.uint8) for _ in range(5)]
    scene = SceneResult(num_frames=5, fps=30., width=100, height=100,
                        court_kp=np.zeros((1, 5, 14, 2), np.float32), court_vis=np.zeros((1, 5, 14), np.float32),
                        player_position=np.zeros((0, 5, 3), np.float32), player_yaw=np.zeros((0, 5), np.float32),
                        ball_uv=np.array([[[.1, .1], [.2, .2], [.9, .9], [.4, .4], [.5, .5]]], np.float32),
                        ball_vis=np.array([[True, True, False, True, True]]))
    segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
    class Writer:
        def write(self, image: Any) -> None:
            pass
        def release(self) -> None:
            pass
    monkeypatch.setattr("src.tennis_scene.scripts.visualize_tasks._open_writer", lambda *args: Writer())
    monkeypatch.setattr(cv2, "line", lambda image, a, b, *args: segments.append((a, b)))
    _render_ball_detection(frames, scene, tmp_path / "ball.mp4", fps=30., frame_range=range(5), trail_length=5)
    assert segments == [((10, 10), (20, 20))] * 3
