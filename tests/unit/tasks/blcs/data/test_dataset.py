"""Persisted-scene boundary tests for the standard BLCS dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from src.tasks.blcs.data.dataset import BallTrajectoryDataset

_CONFIG_DIR = Path("src/tasks/blcs/configs").resolve()


def test_dataset_zeroes_invisible_ball_and_court_coordinates(tmp_path: Path) -> None:
    scene = tmp_path / "scenes" / "scene_000000"
    scene.mkdir(parents=True)
    (scene / "meta.json").write_text(json.dumps({"num_frames": 2}), encoding="utf-8")
    (scene / "scalars.json").write_text(
        json.dumps(
            {
                "num_cameras": 1,
                "cam_0_params": {
                    "R": np.eye(3).tolist(),
                    "C": [0.0, 0.0, 1.0],
                    "f": 1.0,
                    "cx": 0.5,
                    "cy": 0.5,
                    "w": 1,
                    "h": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "test.txt").write_text("scene_000000\n", encoding="utf-8")

    np.save(
        scene / "cam_0_ball_uv.npy",
        np.asarray([[0.25, 0.75], [-3.0, 4.0]], dtype=np.float32),
    )
    np.save(
        scene / "cam_0_ball_vis.npy",
        np.asarray([True, False], dtype=np.bool_),
    )
    court_uv: np.ndarray = np.zeros((20, 2), dtype=np.float32)
    court_uv[1] = np.asarray([float("nan"), float("inf")], dtype=np.float32)
    court_vis: np.ndarray = np.ones(20, dtype=np.bool_)
    court_vis[1] = False
    np.save(scene / "cam_0_court_kp_uv.npy", court_uv)
    np.save(scene / "cam_0_court_kp_vis.npy", court_vis)
    np.save(scene / "ball_pos_norm.npy", np.zeros((2, 3), dtype=np.float32))
    np.save(scene / "ball_vel_world.npy", np.zeros((2, 3), dtype=np.float32))

    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="train")
    config.data.seq_len_range = [2, 2]
    config.data.num_views_range = [1, 1]
    config.data.camera_mode = "first"

    sample = BallTrajectoryDataset(
        scene_dir=tmp_path,
        split_file="test.txt",
        config=config,
        augment=False,
    )[0]

    torch.testing.assert_close(sample["ball_uv"][0, 0], torch.tensor([0.25, 0.75]))
    torch.testing.assert_close(sample["ball_uv"][0, 1], torch.zeros(2))
    torch.testing.assert_close(sample["court_kp"][0, :, 1], torch.zeros(2, 2))
