from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_generator(
    module: str,
    output_dir: Path,
    overrides: list[str],
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        module,
        f"run.output_dir={output_dir}",
        *overrides,
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(
            f"{module} failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def _read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _read_split_file(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _assert_root_file_set(output_dir: Path, expected_files: set[str]) -> None:
    actual_files = {path.name for path in output_dir.iterdir()}
    assert actual_files == expected_files


def _assert_split_contract(output_dir: Path, meta_json: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    split_info = _read_json(output_dir / "split_info.json")
    train_split = _read_split_file(output_dir / "train.txt")
    val_split = _read_split_file(output_dir / "val.txt")
    test_split = _read_split_file(output_dir / "test.txt")

    split_union = train_split + val_split + test_split
    meta_scene_files = [scene["file"] for scene in meta_json["scenes"]]

    assert split_info["n_scenes"] == {
        "train": len(train_split),
        "val": len(val_split),
        "test": len(test_split),
    }
    assert set(meta_scene_files) == set(split_union)
    assert len(meta_scene_files) == len(split_union)

    return split_union, split_info


def _assert_json_camera_record_contract(scene_record: dict[str, Any]) -> None:
    assert set(scene_record) >= {
        "file",
        "scene_id",
        "num_frames",
        "num_cameras_sampled",
        "num_cameras",
        "cameras",
    }
    assert isinstance(scene_record["cameras"], list)
    assert len(scene_record["cameras"]) == scene_record["num_cameras"]


def _load_npz(path: Path) -> np.lib.npyio.NpzFile:
    return np.load(path, allow_pickle=True)


def _assert_plcs_scene_contract(path: Path) -> None:
    with _load_npz(path) as scene_npz:
        num_cameras = int(scene_npz["num_cameras"])
        expected_keys = {
            "meta",
            "position",
            "rotation",
            "canonical_pose_3d",
            "num_cameras",
        }
        for camera_index in range(num_cameras):
            prefix = f"cam_{camera_index}_"
            expected_keys.update(
                {
                    f"{prefix}params",
                    f"{prefix}human_kp_uv",
                    f"{prefix}human_kp_visible",
                    f"{prefix}human_visibility_ratio",
                    f"{prefix}court_kp_uv",
                    f"{prefix}court_kp_visible",
                    f"{prefix}court_visibility_count",
                }
            )
        assert set(scene_npz.files) == expected_keys

        meta = json.loads(str(scene_npz["meta"]))
        expected_meta_keys = {
            "scene_id",
            "motion_source",
            "motion_category",
            "gender",
            "fps",
            "num_frames",
            "initial_position",
            "initial_yaw",
            "num_cameras_sampled",
            "num_cameras",
        }
        assert set(meta) == expected_meta_keys

        position = scene_npz["position"]
        rotation = scene_npz["rotation"]
        canonical_pose_3d = scene_npz["canonical_pose_3d"]

        num_frames = int(position.shape[0])
        assert position.shape == (num_frames, 3)
        assert position.dtype == np.float32
        assert rotation.shape == (num_frames, 2)
        assert rotation.dtype == np.float32
        assert canonical_pose_3d.shape[0] == num_frames
        assert canonical_pose_3d.dtype == np.float32
        assert num_cameras >= 1
        assert meta["num_frames"] == num_frames
        assert meta["num_cameras"] == num_cameras

        for camera_index in range(num_cameras):
            human_kp_uv = scene_npz[f"cam_{camera_index}_human_kp_uv"]
            human_kp_visible = scene_npz[f"cam_{camera_index}_human_kp_visible"]
            court_kp_uv = scene_npz[f"cam_{camera_index}_court_kp_uv"]
            court_kp_visible = scene_npz[f"cam_{camera_index}_court_kp_visible"]
            json.loads(str(scene_npz[f"cam_{camera_index}_params"]))

            assert human_kp_uv.shape == (num_frames, 17, 2)
            assert human_kp_uv.dtype == np.float32
            assert human_kp_visible.shape == (num_frames, 17)
            assert human_kp_visible.dtype == np.bool_
            # The generated PLCS NPZ stores static court keypoints per frame so the
            # loader can slice a time window without special-casing court features.
            assert court_kp_uv.shape == (num_frames, 20, 2)
            assert court_kp_uv.dtype == np.float32
            assert court_kp_visible.shape == (num_frames, 20)
            assert court_kp_visible.dtype == np.bool_


def _assert_blcs_scene_contract(path: Path) -> None:
    with _load_npz(path) as scene_npz:
        num_cameras = int(scene_npz["num_cameras"])
        expected_keys = {
            "meta",
            "ball_pos_world",
            "ball_pos_norm",
            "ball_vel_world",
            "num_cameras",
            "rally_length",
            "end_reason",
        }
        for camera_index in range(num_cameras):
            prefix = f"cam_{camera_index}_"
            expected_keys.update(
                {
                    f"{prefix}params",
                    f"{prefix}ball_uv",
                    f"{prefix}ball_visible",
                    f"{prefix}ball_visibility_ratio",
                    f"{prefix}court_kp_uv",
                    f"{prefix}court_kp_visible",
                    f"{prefix}court_visibility_count",
                }
            )
        assert set(scene_npz.files) == expected_keys

        meta = json.loads(str(scene_npz["meta"]))
        expected_meta_keys = {
            "scene_id",
            "initial_from_cell",
            "initial_from_side",
            "rally_length",
            "end_reason",
            "winner_side",
            "shots",
            "fps_out",
            "sim_fps",
            "num_frames",
            "num_cameras_sampled",
            "num_cameras",
            "physics_config",
            "court_config",
        }
        assert set(meta) == expected_meta_keys

        ball_pos_world = scene_npz["ball_pos_world"]
        ball_pos_norm = scene_npz["ball_pos_norm"]
        ball_vel_world = scene_npz["ball_vel_world"]
        rally_length = int(scene_npz["rally_length"])
        end_reason = str(scene_npz["end_reason"])

        num_frames = int(ball_pos_world.shape[0])
        assert ball_pos_world.shape == (num_frames, 3)
        assert ball_pos_world.dtype == np.float32
        assert ball_pos_norm.shape == (num_frames, 3)
        assert ball_pos_norm.dtype == np.float32
        assert ball_vel_world.shape == (num_frames, 3)
        assert ball_vel_world.dtype == np.float32
        assert num_cameras >= 1
        assert meta["num_frames"] == num_frames
        assert meta["num_cameras"] == num_cameras
        assert meta["rally_length"] == rally_length
        assert meta["end_reason"] == end_reason
        assert meta["shots"]

        for camera_index in range(num_cameras):
            ball_uv = scene_npz[f"cam_{camera_index}_ball_uv"]
            ball_visible = scene_npz[f"cam_{camera_index}_ball_visible"]
            court_kp_uv = scene_npz[f"cam_{camera_index}_court_kp_uv"]
            court_kp_visible = scene_npz[f"cam_{camera_index}_court_kp_visible"]
            json.loads(str(scene_npz[f"cam_{camera_index}_params"]))

            assert ball_uv.shape == (num_frames, 2)
            assert ball_uv.dtype == np.float32
            assert ball_visible.shape == (num_frames,)
            assert ball_visible.dtype == np.bool_
            assert court_kp_uv.shape == (20, 2)
            assert court_kp_uv.dtype == np.float32
            assert court_kp_visible.shape == (20,)
            assert court_kp_visible.dtype == np.bool_


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.local_data
def test_plcs_generator_output_contract(tmp_path: Path) -> None:
    smplh_model_dir = REPO_ROOT / "data/smplx/smplh"
    motion_root = REPO_ROOT / "data/ACCAD"
    if not smplh_model_dir.exists() or not motion_root.exists():
        pytest.skip("PLCS generation test requires local SMPL-H and ACCAD data.")

    output_dir = tmp_path / "plcs"
    _run_generator(
        "src.tasks.plcs.scripts.generate_dataset",
        output_dir,
        [
            "simulation.num_scenes=1",
            "run.device=cpu",
        ],
    )

    _assert_root_file_set(
        output_dir,
        {
            "config.yaml",
            "dataset_info.json",
            "meta.json",
            "scenes",
            "scenes_meta.json",
            "split_info.json",
            "stats.json",
            "test.txt",
            "train.txt",
            "val.txt",
        },
    )

    meta_json = _read_json(output_dir / "meta.json")
    dataset_info = _read_json(output_dir / "dataset_info.json")
    stats_json = _read_json(output_dir / "stats.json")
    scenes_meta = _read_json(output_dir / "scenes_meta.json")
    scene_paths = sorted((output_dir / "scenes").glob("scene_*.npz"))
    split_union, _ = _assert_split_contract(output_dir, meta_json)

    assert meta_json["stats"]["total_scenes"] == len(scene_paths)
    assert dataset_info["total_scenes"] == meta_json["stats"]["total_scenes"]
    assert stats_json["successful_scenes"] == meta_json["stats"]["total_scenes"]
    assert stats_json["failed_scenes"] == 0
    assert len(scenes_meta) == meta_json["stats"]["total_scenes"]
    assert {path.name for path in scene_paths} == set(split_union)

    for scene_record in meta_json["scenes"]:
        _assert_json_camera_record_contract(scene_record)

    assert {scene_meta["scene_id"] for scene_meta in scenes_meta} == {
        scene_record["scene_id"] for scene_record in meta_json["scenes"]
    }

    for scene_path in scene_paths:
        _assert_plcs_scene_contract(scene_path)


@pytest.mark.integration
@pytest.mark.slow
def test_blcs_generator_output_contract(tmp_path: Path) -> None:
    output_dir = tmp_path / "blcs"
    _run_generator(
        "src.tasks.blcs.scripts.generate_dataset",
        output_dir,
        [
            "generator.num_scenes=2",
            "run.device=cpu",
            "run.num_workers=2",
        ],
    )

    _assert_root_file_set(
        output_dir,
        {
            "config.yaml",
            "dataset_info.json",
            "meta.json",
            "scenes",
            "split_info.json",
            "test.txt",
            "train.txt",
            "val.txt",
        },
    )

    meta_json = _read_json(output_dir / "meta.json")
    dataset_info = _read_json(output_dir / "dataset_info.json")
    scene_paths = sorted((output_dir / "scenes").glob("scene_*.npz"))
    split_union, _ = _assert_split_contract(output_dir, meta_json)

    assert meta_json["stats"]["total_scenes"] == len(scene_paths)
    assert dataset_info["total_scenes"] == meta_json["stats"]["total_scenes"]
    assert {path.name for path in scene_paths} == set(split_union)

    for scene_record in meta_json["scenes"]:
        _assert_json_camera_record_contract(scene_record)

    for scene_path in scene_paths:
        _assert_blcs_scene_contract(scene_path)
