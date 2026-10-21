"""Verify same-frame composition and reject incompatible source time bases."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from src.tennis_scene.visualization.rendering.multiview import (
    MultiviewSceneRenderer,
    sampled_frame_indices,
)
from src.tennis_scene.visualization.rendering.tennis_scene_renderer import (
    TennisSceneRenderer,
    TennisSceneStyle,
)
from src.utils.rendering.camera_view import CameraController


@pytest.fixture
def multiview_scene(tiny_scene):
    tiny_scene.width, tiny_scene.height = 32, 18
    tiny_scene.metadata.update(scene_schema_version=2, camera_ids=["cam0", "cam1", "cam2"])
    tiny_scene.player_valid = np.ones((1, 5), bool)
    tiny_scene.player_smpl_valid = np.ones((1, 5), bool)
    tiny_scene.player_kp_3d = np.zeros((1, 5, 17, 3), np.float32)
    tiny_scene.player_kp_3d_vis = np.ones((1, 5, 17), bool)
    tiny_scene.ball_3d_valid = np.ones(5, bool)
    return tiny_scene


def _video(path: Path, *, fps=30., frames=5, size=(32, 18), camera=0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), fps, size)
    assert writer.isOpened()
    try:
        for frame in range(frames):
            # Camera identity in blue, time in red, well clear of the label.
            image = np.zeros((size[1], size[0], 3), np.uint8)
            image[:] = [40 + camera * 60, 0, frame * 40]
            writer.write(image)
    finally:
        writer.release()
    return path


@pytest.fixture
def videos(tmp_path):
    return {f"cam{c}": _video(tmp_path / f"cam{c}.avi", camera=c) for c in range(3)}


@pytest.fixture
def renderer(smpl_renderer_assets):
    faces, regressor = smpl_renderer_assets
    return TennisSceneRenderer(
        TennisSceneStyle(player_representation="smpl", theme="dark", show_hud=False, show_minimap=False),
        smpl_faces_path=faces, smpl_joint_regressor_path=regressor,
        camera=CameraController("corner"),
    )


def test_same_source_frame_in_all_four_panels(multiview_scene, renderer, videos):
    # Deliberately reverse mapping insertion order: scene camera IDs own order.
    draw = Mock(wraps=renderer.render_frame_3d)
    renderer.render_frame_3d = draw
    with MultiviewSceneRenderer(multiview_scene, renderer, dict(reversed(list(videos.items()))), width=480) as view:
        first, later = view.render(0), view.render(3)
        assert first.shape == later.shape == (350, 480, 3)
        for row in range(3):
            y = 48 + row * 90 + 65
            # RGB image: red encodes frame 3; blue encodes the camera row.
            np.testing.assert_allclose(later[y, 100], [120, 0, 40 + row * 60], atol=6)
            assert later[y, 100, 0] > first[y, 100, 0] + 100
        assert [call.args[1] for call in draw.call_args_list] == [0, 3]
        assert np.any(first[48:318, 160:] != later[48:318, 160:])
    assert all(reader._cap is None for reader in view.readers)
    with pytest.raises(RuntimeError, match="closed"):
        view.render(0)


@pytest.mark.parametrize("changes", [{"fps": 25.}, {"frames": 4}, {"size": (64, 36)}])
def test_reject_video_timebase_or_dimensions(multiview_scene, renderer, videos, tmp_path, changes):
    videos["cam1"] = _video(tmp_path / "incompatible.avi", **changes)
    with pytest.raises(ValueError, match="cam1.*differ from scene"):
        MultiviewSceneRenderer(multiview_scene, renderer, videos)


def test_reject_ambiguous_camera_binding(multiview_scene, renderer, videos):
    videos["wrong_camera"] = videos.pop("cam0")
    with pytest.raises(ValueError, match="camera_ids"):
        MultiviewSceneRenderer(multiview_scene, renderer, videos)


def test_no_silent_video_decode_fallback(multiview_scene, renderer, videos, monkeypatch):
    with MultiviewSceneRenderer(multiview_scene, renderer, videos) as view:
        monkeypatch.setattr(view.readers[1], "read", Mock(side_effect=RuntimeError("decode failed")))
        with pytest.raises(RuntimeError, match="decode failed"):
            view.render(2)


def test_missing_mesh_and_invalid_source_frame_are_rejected(multiview_scene, renderer, videos):
    with MultiviewSceneRenderer(multiview_scene, renderer, videos) as view:
        for index in (-1, 5):
            with pytest.raises(ValueError, match="out of range"):
                view.render(index)
    multiview_scene.player_smpl_valid[:] = False
    with pytest.raises(ValueError, match="valid SMPL/ball"):
        MultiviewSceneRenderer(multiview_scene, renderer, videos)


def test_sampling_uses_source_indices_without_stretching_clip():
    assert sampled_frame_indices(1010, start=0, stop=1010, step=6) == tuple(range(0, 1010, 6))
    assert sampled_frame_indices(10, start=2, stop=9, step=3) == (2, 5, 8)


@pytest.mark.parametrize("start,stop,step", [(-1, 5, 1), (0, 6, 1), (3, 3, 1), (0, 5, 0)])
def test_invalid_sampling_is_rejected(start, stop, step):
    with pytest.raises(ValueError, match="Require"):
        sampled_frame_indices(5, start=start, stop=stop, step=step)
