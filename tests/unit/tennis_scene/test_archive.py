"""Tests for the canonical tennis-scene schema and NPZ archive boundary."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

import src.tennis_scene as tennis_scene_package
from src.tennis_scene.archive import load_scene_result, save_scene_result
from src.tennis_scene.schema import SceneResult


def _scene() -> SceneResult:
    return SceneResult(
        num_frames=3,
        fps=29.97,
        width=640,
        height=360,
        court_kp=np.arange(2 * 3 * 4 * 2, dtype=np.float32).reshape(2, 3, 4, 2),
        court_vis=np.ones((2, 3, 4), dtype=np.float32),
        player_position=np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3),
        player_yaw=np.arange(2 * 3, dtype=np.float32).reshape(2, 3),
        smpl_body_pose=np.zeros((2, 3, 63), dtype=np.float32),
        smpl_global_orient=np.zeros((2, 3, 3), dtype=np.float32),
        smpl_betas=np.zeros((2, 10), dtype=np.float32),
        smpl_vertices_local=np.zeros((2, 3, 5, 3), dtype=np.float32),
        ball_uv=np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2),
        ball_vis=np.array([[True, False, True], [True, True, False]]),
        ball_3d=np.arange(3 * 3, dtype=np.float32).reshape(3, 3),
        human_kp_2d=np.zeros((2, 2, 3, 17, 2), dtype=np.float32),
        human_kp_vis=np.ones((2, 2, 3, 17), dtype=np.float32),
        player_track_ids=np.array([4, 8], dtype=np.int32),
        player_kp_3d=np.zeros((2, 3, 17, 3), dtype=np.float32),
        metadata={"camera_ids": ["near", "far"], "source": "fixture"},
    )


def test_archive_round_trip_preserves_every_array_and_metadata(tmp_path: Path) -> None:
    expected = _scene()
    path = tmp_path / "scene.npz"

    save_scene_result(expected, path)
    actual = load_scene_result(path)

    assert actual.num_frames == expected.num_frames
    assert actual.fps == expected.fps
    assert actual.width == expected.width
    assert actual.height == expected.height
    assert actual.metadata == expected.metadata
    for field in (
        "court_kp",
        "court_vis",
        "player_position",
        "player_yaw",
        "smpl_body_pose",
        "smpl_global_orient",
        "smpl_betas",
        "smpl_vertices_local",
        "ball_uv",
        "ball_vis",
        "ball_3d",
        "human_kp_2d",
        "human_kp_vis",
        "player_track_ids",
        "player_kp_3d",
    ):
        actual_array = getattr(actual, field)
        expected_array = getattr(expected, field)
        assert actual_array.dtype == expected_array.dtype
        np.testing.assert_array_equal(actual_array, expected_array)
    assert path.with_suffix(".metadata.json").is_file()


def test_archive_preserves_absent_optional_arrays(tmp_path: Path) -> None:
    expected = _scene()
    expected.smpl_vertices_local = None
    expected.ball_uv = None
    expected.ball_vis = None
    expected.ball_3d = None
    expected.human_kp_2d = None
    expected.human_kp_vis = None
    expected.player_track_ids = None
    expected.player_kp_3d = None
    path = tmp_path / "minimal.npz"

    save_scene_result(expected, path)
    actual = load_scene_result(path)

    assert actual.smpl_vertices_local is None
    assert actual.ball_uv is None
    assert actual.ball_vis is None
    assert actual.ball_3d is None
    assert actual.human_kp_2d is None
    assert actual.human_kp_vis is None
    assert actual.player_track_ids is None
    assert actual.player_kp_3d is None


def test_load_rejects_missing_metadata_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "scene.npz"
    np.savez_compressed(path, ignored=np.array([1]))

    with pytest.raises(FileNotFoundError, match="metadata sidecar"):
        load_scene_result(path)


def test_load_rejects_non_object_metadata(tmp_path: Path) -> None:
    path = tmp_path / "scene.npz"
    save_scene_result(_scene(), path)
    path.with_suffix(".metadata.json").write_text(
        json.dumps(["not", "an", "object"]), encoding="utf-8"
    )

    with pytest.raises(TypeError, match="must be a JSON object"):
        load_scene_result(path)


def test_save_rejects_non_npz_path_without_writing(tmp_path: Path) -> None:
    path = tmp_path / "scene.data"

    with pytest.raises(ValueError, match="must use the .npz suffix"):
        save_scene_result(_scene(), path)

    assert not path.exists()
    assert not (tmp_path / "scene.data.npz").exists()


def test_save_rejects_non_object_metadata_without_writing(tmp_path: Path) -> None:
    scene = _scene()
    scene.metadata = ["not", "an", "object"]  # type: ignore[assignment]
    path = tmp_path / "scene.npz"

    with pytest.raises(TypeError, match="metadata must be a dictionary"):
        save_scene_result(scene, path)

    assert not path.exists()
    assert not path.with_suffix(".metadata.json").exists()


def test_save_rejects_non_json_metadata_without_writing(tmp_path: Path) -> None:
    scene = _scene()
    scene.metadata = {"invalid": object()}
    path = tmp_path / "scene.npz"

    with pytest.raises(TypeError, match="not JSON serializable"):
        save_scene_result(scene, path)

    assert not path.exists()
    assert not path.with_suffix(".metadata.json").exists()


def test_removed_tennis_scene_convenience_paths_fail_explicitly() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.tennis_scene.io")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.tennis_scene.utils")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.tennis_scene.utils.transforms")
    assert not hasattr(tennis_scene_package, "SceneResult")
    assert not hasattr(SceneResult, "load")
    assert not hasattr(SceneResult, "save")


def test_shared_scene_directory_io_remains_available() -> None:
    scene_io = importlib.import_module("src.utils.data.scene_io")

    assert callable(scene_io.load_scene_payload)
