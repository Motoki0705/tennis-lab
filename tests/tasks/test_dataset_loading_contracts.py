from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.tasks.ball_detection.data.tracknet_datamodule import TrackNetDataModule
from src.tasks.blcs.data.dataset import BallTrajectoryDataset
from src.tasks.court_detection.data.court_kp_dataset import CourtKPDataset
from src.tasks.court_detection.data.court_line_dataset import CourtLineDataset
from src.tasks.court_detection.data.court_seg_dataset import CourtSegDataset
from src.tasks.plcs.data.dataset import SceneDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.local_data


def _require_paths(*paths: Path) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        pytest.skip(f"local dataset assets are missing: {joined}")


def _assert_tensor(
    value: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    assert isinstance(value, torch.Tensor)
    assert tuple(value.shape) == shape
    assert value.dtype == dtype


def test_plcs_scene_dataset_getitem_contract() -> None:
    scene_dir = REPO_ROOT / "data/plcs/scenes"
    split_file = REPO_ROOT / "data/plcs/train.txt"
    _require_paths(scene_dir, split_file)

    dataset = SceneDataset(
        scene_dir=scene_dir,
        split_file=split_file,
        config={
            "data": {
                "seq_len_range": [64, 64],
                "num_views_range": [1, 1],
                "camera_mode": "first",
                "num_court_kp": 20,
                "augmentation": {
                    "keypoint_noise_std": 0.0,
                    "visibility_drop_prob": 0.0,
                },
            }
        },
        augment=False,
    )

    assert len(dataset) > 0

    sample = dataset[0]

    assert set(sample) == {
        "human_kp",
        "court_kp",
        "human_vis",
        "court_vis",
        "human_mask",
        "position",
        "rotation",
        "human_kp_3d",
    }
    _assert_tensor(sample["human_kp"], shape=(1, 64, 17, 2), dtype=torch.float32)
    _assert_tensor(sample["court_kp"], shape=(1, 64, 20, 2), dtype=torch.float32)
    _assert_tensor(sample["human_vis"], shape=(1, 64, 17), dtype=torch.float32)
    _assert_tensor(sample["court_vis"], shape=(1, 64, 20), dtype=torch.float32)
    _assert_tensor(sample["human_mask"], shape=(1, 64), dtype=torch.float32)
    _assert_tensor(sample["position"], shape=(64, 3), dtype=torch.float32)
    _assert_tensor(sample["rotation"], shape=(64, 2), dtype=torch.float32)
    _assert_tensor(sample["human_kp_3d"], shape=(64, 17, 3), dtype=torch.float32)

    assert sample["human_kp"].shape[:2] == sample["human_mask"].shape
    assert sample["human_kp"].shape[1] == sample["position"].shape[0]
    assert sample["human_kp"].shape[1] == sample["rotation"].shape[0]
    assert sample["human_kp"].shape[1] == sample["human_kp_3d"].shape[0]


def test_blcs_ball_trajectory_dataset_getitem_contract() -> None:
    scene_dir = REPO_ROOT / "data/blcs/scenes"
    split_file = REPO_ROOT / "data/blcs/train.txt"
    _require_paths(scene_dir, split_file)

    dataset = BallTrajectoryDataset(
        scene_dir=scene_dir,
        split_file=split_file,
        config={
            "data": {
                "seq_len_range": [32, 32],
                "num_views_range": [2, 2],
                "camera_mode": "first",
                "num_court_kp": 20,
                "augmentation": {
                    "uv_noise_std": 0.0,
                    "visibility_drop_prob": 0.0,
                    "scale_range": [1.0, 1.0],
                },
            }
        },
        augment=False,
    )

    assert len(dataset) > 0

    sample = dataset[0]

    assert set(sample) == {
        "ball_uv",
        "ball_vis",
        "ball_mask",
        "court_kp",
        "court_vis",
        "position_3d",
        "velocity_3d",
        "seq_len",
        "camera_R",
        "camera_C",
        "camera_f",
        "camera_cx",
        "camera_cy",
        "camera_w",
        "camera_h",
    }
    _assert_tensor(sample["ball_uv"], shape=(2, 32, 2), dtype=torch.float32)
    _assert_tensor(sample["ball_vis"], shape=(2, 32), dtype=torch.float32)
    _assert_tensor(sample["ball_mask"], shape=(2, 32), dtype=torch.float32)
    _assert_tensor(sample["court_kp"], shape=(2, 32, 20, 2), dtype=torch.float32)
    _assert_tensor(sample["court_vis"], shape=(2, 32, 20), dtype=torch.float32)
    _assert_tensor(sample["position_3d"], shape=(32, 3), dtype=torch.float32)
    _assert_tensor(sample["velocity_3d"], shape=(32, 3), dtype=torch.float32)
    _assert_tensor(sample["seq_len"], shape=(), dtype=torch.int64)
    _assert_tensor(sample["camera_R"], shape=(2, 3, 3), dtype=torch.float32)
    _assert_tensor(sample["camera_C"], shape=(2, 3), dtype=torch.float32)
    _assert_tensor(sample["camera_f"], shape=(2,), dtype=torch.float32)
    _assert_tensor(sample["camera_cx"], shape=(2,), dtype=torch.float32)
    _assert_tensor(sample["camera_cy"], shape=(2,), dtype=torch.float32)
    _assert_tensor(sample["camera_w"], shape=(2,), dtype=torch.float32)
    _assert_tensor(sample["camera_h"], shape=(2,), dtype=torch.float32)

    seq_len = sample["seq_len"].item()
    assert seq_len == 32
    assert sample["ball_uv"].shape[:2] == sample["ball_mask"].shape
    assert sample["ball_uv"].shape[:2] == sample["ball_vis"].shape
    assert sample["ball_uv"].shape[1] == sample["position_3d"].shape[0]
    assert sample["ball_uv"].shape[1] == sample["velocity_3d"].shape[0]


def test_ball_detection_dataset_getitem_contract() -> None:
    data_dir = REPO_ROOT / "data/tennis"
    split_file = REPO_ROOT / "src/tasks/ball_detection/configs/data/splits/train.txt"
    _require_paths(data_dir, split_file)

    config = {
        "data": {
            "data_dir": str(data_dir),
            "sample_stride": 1,
            "image_size": [288, 512],
            "heatmap_size": [144, 256],
            "sigma_ratio": 0.0066,
        },
        "model": {"num_frames": 8},
    }
    datamodule = TrackNetDataModule(config)
    dataset = datamodule.create_dataset(
        split_name="train",
        split_file=split_file,
        augmentation=None,
    )

    assert len(dataset) > 0

    sample = dataset[0]

    assert set(sample) == {
        "images",
        "heatmaps",
        "coords",
        "visibility",
        "original_size",
        "heatmap_size",
    }
    _assert_tensor(sample["images"], shape=(8, 3, 288, 512), dtype=torch.float32)
    _assert_tensor(sample["heatmaps"], shape=(8, 144, 256), dtype=torch.float32)
    _assert_tensor(sample["coords"], shape=(8, 8, 2), dtype=torch.float32)
    _assert_tensor(sample["visibility"], shape=(8, 8), dtype=torch.float32)
    _assert_tensor(sample["original_size"], shape=(2,), dtype=torch.float32)
    _assert_tensor(sample["heatmap_size"], shape=(2,), dtype=torch.float32)

    assert sample["images"].shape[0] == 8
    assert sample["images"].shape[0] == sample["heatmaps"].shape[0]
    assert sample["images"].shape[0] == sample["coords"].shape[0]
    assert sample["images"].shape[0] == sample["visibility"].shape[0]


def test_court_kp_dataset_getitem_contract() -> None:
    root = REPO_ROOT / "data/court"
    _require_paths(root, root / "data_train.json", root / "images")

    dataset = CourtKPDataset(
        root=root,
        split="train",
        is_train=False,
        config={"val_short_side": 640, "sigma_ratio": 0.01},
    )

    assert len(dataset) > 0

    sample = dataset[0]

    assert set(sample) == {"image", "heatmap", "keypoints", "image_id"}
    _assert_tensor(sample["image"], shape=(3, 640, 1136), dtype=torch.float32)
    _assert_tensor(sample["heatmap"], shape=(14, 640, 1136), dtype=torch.float32)
    _assert_tensor(sample["keypoints"], shape=(14, 2), dtype=torch.float32)
    assert isinstance(sample["image_id"], str)
    assert sample["image_id"]

    assert sample["image"].shape[-2:] == sample["heatmap"].shape[-2:]
    assert sample["heatmap"].shape[0] == sample["keypoints"].shape[0]


def test_court_line_dataset_getitem_contract() -> None:
    root = REPO_ROOT / "data/court"
    _require_paths(root, root / "data_train.json", root / "images", root / "line_masks")

    dataset = CourtLineDataset(
        root=root,
        split="train",
        is_train=False,
        config={"val_short_side": 640},
    )

    assert len(dataset) > 0

    sample = dataset[0]

    assert set(sample) == {"image", "mask", "image_id"}
    _assert_tensor(sample["image"], shape=(3, 640, 1136), dtype=torch.float32)
    _assert_tensor(sample["mask"], shape=(1, 640, 1136), dtype=torch.float32)
    assert isinstance(sample["image_id"], str)
    assert sample["image_id"]

    assert sample["image"].shape[-2:] == sample["mask"].shape[-2:]


def test_court_seg_dataset_getitem_contract() -> None:
    root = REPO_ROOT / "data/court"
    _require_paths(root, root / "data_train.json", root / "images", root / "masks")

    dataset = CourtSegDataset(
        root=root,
        split="train",
        is_train=False,
        config={"val_short_side": 640},
    )

    assert len(dataset) > 0

    sample = dataset[0]

    assert set(sample) == {"image", "mask", "image_id"}
    _assert_tensor(sample["image"], shape=(3, 640, 1136), dtype=torch.float32)
    _assert_tensor(sample["mask"], shape=(640, 1136), dtype=torch.int64)
    assert isinstance(sample["image_id"], str)
    assert sample["image_id"]

    assert sample["image"].shape[-2:] == sample["mask"].shape
