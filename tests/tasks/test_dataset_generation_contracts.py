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
    completed = _run_generator_raw(module, output_dir, overrides)
    if completed.returncode != 0:
        pytest.fail(
            f"{module} failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def _run_generator_raw(
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


def _load_npy(scene_dir: Path, key: str) -> np.ndarray:
    """Load a single .npy array from a scene directory."""
    return np.load(scene_dir / f"{key}.npy")


def _assert_plcs_scene_contract(scene_dir: Path) -> None:
    """Verify the npy + json directory layout for a PLCS scene."""
    assert scene_dir.is_dir()

    # Load scalars and meta
    scalars = _read_json(scene_dir / "scalars.json")
    meta = _read_json(scene_dir / "meta.json")
    num_cameras = int(scalars["num_cameras"])

    # Check expected npy files
    expected_npy_keys = {
        "position",
        "rotation",
        "canonical_pose_3d",
    }
    for camera_index in range(num_cameras):
        prefix = f"cam_{camera_index}_"
        expected_npy_keys.update(
            {
                f"{prefix}human_kp_uv",
                f"{prefix}human_kp_visible",
                f"{prefix}human_visibility_ratio",
                f"{prefix}court_kp_uv",
                f"{prefix}court_kp_visible",
                f"{prefix}court_visibility_count",
            }
        )
    actual_npy_files = {p.stem for p in scene_dir.glob("*.npy")}
    assert actual_npy_files == expected_npy_keys

    # Check expected json files
    assert (scene_dir / "meta.json").exists()
    assert (scene_dir / "scalars.json").exists()

    # Check scalars contain camera params
    for camera_index in range(num_cameras):
        assert f"cam_{camera_index}_params" in scalars

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

    position = _load_npy(scene_dir, "position")
    rotation = _load_npy(scene_dir, "rotation")
    canonical_pose_3d = _load_npy(scene_dir, "canonical_pose_3d")

    num_frames = int(position.shape[0])
    assert position.shape == (num_frames, 3)
    assert position.dtype == np.float32
    assert rotation.shape == (num_frames, 2)
    assert rotation.dtype == np.float32
    assert canonical_pose_3d.shape[0] == num_frames
    assert canonical_pose_3d.dtype == np.float32
    assert num_cameras == 6
    assert meta["num_frames"] == num_frames
    assert meta["num_cameras"] == num_cameras
    assert meta["num_cameras_sampled"] == 6

    for camera_index in range(num_cameras):
        human_kp_uv = _load_npy(scene_dir, f"cam_{camera_index}_human_kp_uv")
        human_kp_visible = _load_npy(scene_dir, f"cam_{camera_index}_human_kp_visible")
        court_kp_uv = _load_npy(scene_dir, f"cam_{camera_index}_court_kp_uv")
        court_kp_visible = _load_npy(scene_dir, f"cam_{camera_index}_court_kp_visible")
        # Validate camera params are valid JSON
        json.loads(json.dumps(scalars[f"cam_{camera_index}_params"]))

        assert human_kp_uv.shape == (num_frames, 17, 2)
        assert human_kp_uv.dtype == np.float32
        assert human_kp_visible.shape == (num_frames, 17)
        assert human_kp_visible.dtype == np.bool_
        # The generated PLCS scene stores static court keypoints per frame so the
        # loader can slice a time window without special-casing court features.
        assert court_kp_uv.shape == (num_frames, 20, 2)
        assert court_kp_uv.dtype == np.float32
        assert court_kp_visible.shape == (num_frames, 20)
        assert court_kp_visible.dtype == np.bool_


def _assert_blcs_scene_contract(scene_dir: Path) -> None:
    """Verify the npy + json directory layout for a BLCS scene."""
    assert scene_dir.is_dir()

    # Load scalars and meta
    scalars = _read_json(scene_dir / "scalars.json")
    meta = _read_json(scene_dir / "meta.json")
    num_cameras = int(scalars["num_cameras"])

    # Check expected npy files
    expected_npy_keys = {
        "ball_pos_world",
        "ball_pos_norm",
        "ball_vel_world",
    }
    for camera_index in range(num_cameras):
        prefix = f"cam_{camera_index}_"
        expected_npy_keys.update(
            {
                f"{prefix}ball_uv",
                f"{prefix}ball_visible",
                f"{prefix}ball_visibility_ratio",
                f"{prefix}court_kp_uv",
                f"{prefix}court_kp_visible",
                f"{prefix}court_visibility_count",
            }
        )
    actual_npy_files = {p.stem for p in scene_dir.glob("*.npy")}
    assert actual_npy_files == expected_npy_keys

    # Check expected json files
    assert (scene_dir / "meta.json").exists()
    assert (scene_dir / "scalars.json").exists()

    # Check scalars contain expected keys
    assert "num_cameras" in scalars
    assert "rally_length" in scalars
    assert "end_reason" in scalars
    for camera_index in range(num_cameras):
        assert f"cam_{camera_index}_params" in scalars

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

    ball_pos_world = _load_npy(scene_dir, "ball_pos_world")
    ball_pos_norm = _load_npy(scene_dir, "ball_pos_norm")
    ball_vel_world = _load_npy(scene_dir, "ball_vel_world")
    rally_length = int(scalars["rally_length"])
    end_reason = str(scalars["end_reason"])

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
    assert meta["num_cameras_sampled"] == 6
    assert meta["rally_length"] == rally_length
    assert meta["end_reason"] == end_reason
    assert meta["shots"]

    for camera_index in range(num_cameras):
        ball_uv = _load_npy(scene_dir, f"cam_{camera_index}_ball_uv")
        ball_visible = _load_npy(scene_dir, f"cam_{camera_index}_ball_visible")
        court_kp_uv = _load_npy(scene_dir, f"cam_{camera_index}_court_kp_uv")
        court_kp_visible = _load_npy(scene_dir, f"cam_{camera_index}_court_kp_visible")
        # Validate camera params are valid JSON
        json.loads(json.dumps(scalars[f"cam_{camera_index}_params"]))

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
@pytest.mark.parametrize(
    ("num_scenes", "num_workers"),
    [(1, 1), (2, 2)],
    ids=["single_worker", "multi_worker"],
)
def test_plcs_generator_output_contract(
    tmp_path: Path,
    num_scenes: int,
    num_workers: int,
) -> None:
    smplh_model_dir = REPO_ROOT / "data/smplx/smplh"
    motion_root = REPO_ROOT / "data/ACCAD"
    if not smplh_model_dir.exists() or not motion_root.exists():
        pytest.skip("PLCS generation test requires local SMPL-H and ACCAD data.")

    output_dir = tmp_path / "plcs"
    _run_generator(
        "src.tasks.plcs.scripts.generate_dataset",
        output_dir,
        [
            f"simulation.num_scenes={num_scenes}",
            "run.device=cpu",
            f"run.num_workers={num_workers}",
        ],
    )

    _assert_root_file_set(
        output_dir,
        {
            "config.yaml",
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
    stats_json = _read_json(output_dir / "stats.json")
    scenes_meta = _read_json(output_dir / "scenes_meta.json")
    scene_dirs = sorted(
        p for p in (output_dir / "scenes").iterdir() if p.is_dir() and p.name.startswith("scene_")
    )
    split_union, _ = _assert_split_contract(output_dir, meta_json)

    assert meta_json["stats"]["total_scenes"] == len(scene_dirs)
    assert stats_json["successful_scenes"] == meta_json["stats"]["total_scenes"]
    assert stats_json["failed_scenes"] == 0
    assert len(scenes_meta) == meta_json["stats"]["total_scenes"]
    assert {p.name for p in scene_dirs} == set(split_union)

    for scene_record in meta_json["scenes"]:
        _assert_json_camera_record_contract(scene_record)
        assert scene_record["num_cameras"] == 6
        assert scene_record["num_cameras_sampled"] == 6

    assert {scene_meta["scene_id"] for scene_meta in scenes_meta} == {
        scene_record["scene_id"] for scene_record in meta_json["scenes"]
    }

    for scene_dir in scene_dirs:
        _assert_plcs_scene_contract(scene_dir)


def test_plcs_parallel_generator_requires_cpu(tmp_path: Path) -> None:
    output_dir = tmp_path / "plcs"
    completed = _run_generator_raw(
        "src.tasks.plcs.scripts.generate_dataset",
        output_dir,
        [
            "simulation.num_scenes=2",
            "run.device=cuda",
            "run.num_workers=2",
        ],
    )

    assert completed.returncode != 0
    combined_output = f"{completed.stdout}\n{completed.stderr}"
    assert "Parallel PLCS dataset generation requires run.device=cpu" in combined_output


def test_plcs_generator_requires_positive_num_workers(tmp_path: Path) -> None:
    output_dir = tmp_path / "plcs"
    completed = _run_generator_raw(
        "src.tasks.plcs.scripts.generate_dataset",
        output_dir,
        [
            "simulation.num_scenes=1",
            "run.device=cpu",
            "run.num_workers=0",
        ],
    )

    assert completed.returncode != 0
    combined_output = f"{completed.stdout}\n{completed.stderr}"
    assert "Parallel PLCS scene generation requires num_workers >= 1" in combined_output


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize(
    ("num_scenes", "num_workers"),
    [(1, 1), (2, 2)],
    ids=["single_worker", "multi_worker"],
)
def test_blcs_generator_output_contract(
    tmp_path: Path,
    num_scenes: int,
    num_workers: int,
) -> None:
    output_dir = tmp_path / "blcs"
    _run_generator(
        "src.tasks.blcs.scripts.generate_dataset",
        output_dir,
        [
            f"generator.num_scenes={num_scenes}",
            "run.device=cpu",
            f"run.num_workers={num_workers}",
        ],
    )

    _assert_root_file_set(
        output_dir,
        {
            "config.yaml",
            "meta.json",
            "scenes",
            "split_info.json",
            "test.txt",
            "train.txt",
            "val.txt",
        },
    )

    meta_json = _read_json(output_dir / "meta.json")
    scene_dirs = sorted(
        p for p in (output_dir / "scenes").iterdir() if p.is_dir() and p.name.startswith("scene_")
    )
    split_union, _ = _assert_split_contract(output_dir, meta_json)

    assert meta_json["stats"]["total_scenes"] == len(scene_dirs)
    assert {p.name for p in scene_dirs} == set(split_union)

    for scene_record in meta_json["scenes"]:
        _assert_json_camera_record_contract(scene_record)
        assert scene_record["num_cameras"] == 6
        assert scene_record["num_cameras_sampled"] == 6

    for scene_dir in scene_dirs:
        _assert_blcs_scene_contract(scene_dir)


def test_blcs_generator_requires_positive_num_workers(tmp_path: Path) -> None:
    output_dir = tmp_path / "blcs"
    completed = _run_generator_raw(
        "src.tasks.blcs.scripts.generate_dataset",
        output_dir,
        [
            "generator.num_scenes=1",
            "run.device=cpu",
            "run.num_workers=0",
        ],
    )

    assert completed.returncode != 0
    combined_output = f"{completed.stdout}\n{completed.stderr}"
    assert "Parallel BLCS scene generation requires num_workers >= 1" in combined_output
