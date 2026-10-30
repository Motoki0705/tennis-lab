"""Tests for the canonical tennis-scene schema and NPZ archive boundary."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

import src.tennis_scene as tennis_scene_package
from src.tennis_scene.archive import load_scene_result, save_scene_result
from src.tennis_scene.schema import (
    SCENE_MASK_FIELDS,
    SCENE_REASON_FIELDS,
    SceneResult,
)


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
        gvhmr_aligned_player_position=np.full((2, 3, 3), 10.0, dtype=np.float32),
        gvhmr_aligned_player_yaw=np.full((2, 3), 0.5, dtype=np.float32),
        gvhmr_aligned_smpl_global_orient=np.full((2, 3, 3), 1.0, dtype=np.float32),
        gvhmr_aligned_smpl_vertices_local=np.full((2, 3, 5, 3), 2.0, dtype=np.float32),
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
        "gvhmr_aligned_player_position",
        "gvhmr_aligned_player_yaw",
        "gvhmr_aligned_smpl_global_orient",
        "gvhmr_aligned_smpl_vertices_local",
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


def test_ball_archive_contract_has_camera_time_streams_but_no_ball_ids(
    tmp_path: Path,
) -> None:
    expected = _scene()
    expected.ball_uv = np.array(
        [
            [[0.1, 0.2], [0.0, 0.0], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    path = tmp_path / "ball-streams.npz"

    save_scene_result(expected, path)
    actual = load_scene_result(path)

    assert actual.ball_uv is not None
    assert actual.ball_vis is not None
    assert actual.ball_uv.shape == (2, 3, 2)
    assert actual.ball_vis.shape == (2, 3)
    assert actual.ball_uv.dtype == np.float32
    assert actual.ball_vis.dtype == np.bool_
    assert actual.player_track_ids is not None
    assert not hasattr(actual, "ball_track_ids")
    with np.load(path, allow_pickle=False) as archive:
        assert archive["ball_uv"].shape == (2, 3, 2)
        assert archive["ball_vis"].shape == (2, 3)
        assert "player_track_ids" in archive.files
        assert "ball_track_ids" not in archive.files


def test_archive_preserves_absent_optional_arrays(tmp_path: Path) -> None:
    expected = _scene()
    expected.smpl_vertices_local = None
    expected.gvhmr_aligned_player_position = None
    expected.gvhmr_aligned_player_yaw = None
    expected.gvhmr_aligned_smpl_global_orient = None
    expected.gvhmr_aligned_smpl_vertices_local = None
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
    assert actual.gvhmr_aligned_player_position is None
    assert actual.gvhmr_aligned_player_yaw is None
    assert actual.gvhmr_aligned_smpl_global_orient is None
    assert actual.gvhmr_aligned_smpl_vertices_local is None
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


def test_v2_round_trip_preserves_validity_masks_and_reasons(tmp_path: Path, v2_scene: SceneResult) -> None:
    expected = v2_scene
    save_scene_result(expected, tmp_path / "scene.npz")
    actual = load_scene_result(tmp_path / "scene.npz")
    assert actual.schema_version == 2
    for name in (*SCENE_MASK_FIELDS, *SCENE_REASON_FIELDS):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
        assert getattr(actual, name).dtype == getattr(expected, name).dtype
    # Identical zero coordinates represent a valid origin at frame 0 and
    # missing geometry at frame 1. Only the stored masks/reasons distinguish them.
    for geometry, mask, reason in (
        ("player_position", "player_valid", "player_rejection_code"),
        ("player_kp_3d", "player_kp_3d_vis", "player_kp_3d_rejection_code"),
        ("ball_3d", "ball_3d_valid", "ball_rejection_code"),
    ):
        np.testing.assert_array_equal(getattr(actual, geometry), getattr(expected, geometry))
        coordinates = getattr(actual, geometry)
        valid = getattr(actual, mask)
        codes = getattr(actual, reason)
        if geometry != "ball_3d":
            coordinates, valid, codes = coordinates[0], valid[0], codes[0]
        assert not coordinates[:2].any()
        assert valid[0].all() and not valid[1].any()
        assert (codes[0] == 0).all() and (codes[1] != 0).all()


def test_v2_save_rejects_inconsistent_validity(tmp_path: Path, v2_scene: SceneResult) -> None:
    v2_scene.player_valid = None
    path = tmp_path / "scene.npz"
    with pytest.raises(ValueError, match="player_valid must be boolean"):
        save_scene_result(v2_scene, path)
    assert not path.exists()
    assert not path.with_suffix(".metadata.json").exists()


def test_v2_load_rejects_archive_without_a_mask(tmp_path: Path, v2_scene: SceneResult) -> None:
    path = tmp_path / "scene.npz"
    save_scene_result(v2_scene, path)
    with np.load(path) as archive:
        arrays = {name: archive[name] for name in archive.files if name != "ball_3d_valid"}
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="v2 archive requires ball_3d_valid"):
        load_scene_result(path)


def test_v1_scene_rejects_validity_fields(tmp_path: Path) -> None:
    scene = _scene()
    scene.ball_3d_valid = np.ones(3, bool)
    with pytest.raises(ValueError, match="v1 scenes have no reconstruction validity"):
        save_scene_result(scene, tmp_path / "scene.npz")
