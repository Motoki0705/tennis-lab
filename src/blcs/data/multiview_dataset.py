"""Multi-view dataset for BLCS training from pre-generated scene files.

This module provides datasets that return ball trajectory observations from
multiple cameras simultaneously, enabling multi-camera fusion for 3D trajectory
estimation.
"""

from __future__ import annotations

import json
import random as rng
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.base.data.augmentation import add_gaussian_noise, random_visibility_dropout
from src.blcs.data.types import BLCSMultiViewSample

if TYPE_CHECKING:
    from omegaconf import DictConfig


class MultiViewBallTrajectoryDataset(Dataset):
    """Dataset for multi-view ball trajectory estimation.

    Unlike BallTrajectoryDataset which returns single-camera samples, this
    dataset returns observations from multiple cameras for the same trajectory,
    enabling multi-camera triangulation and fusion approaches.
    """

    def __init__(
        self,
        scene_dir: str | Path | None = None,
        split_file: str | Path | None = None,
        config: DictConfig | None = None,
        augment: bool = True,
        num_views: int = 2,
        min_cameras: int = 2,
    ) -> None:
        """Initialize the multi-view dataset.

        Args:
            scene_dir: Directory containing pre-generated scenes.
            split_file: Path to split file (train.txt, val.txt, test.txt).
            config: Configuration dictionary.
            augment: Apply data augmentation.
            num_views: Number of camera views to return per sample.
            min_cameras: Minimum cameras required in a scene.

        """
        super().__init__()

        self.config = config or {}
        self.augment = augment
        self.num_views = num_views
        self.min_cameras = min_cameras

        data_cfg = self.config.get("data", {})
        self.min_seq_len = int(data_cfg.get("min_seq_len", 15))
        self.max_seq_len = int(data_cfg.get("max_seq_len", 120))

        # Augmentation parameters
        aug_cfg = data_cfg.get("augmentation", {})
        self.uv_noise_std = aug_cfg.get("uv_noise_std", 0.005)
        self.vis_drop_prob = aug_cfg.get("visibility_drop_prob", 0.1)

        # Load scene list
        self.scene_dir = Path(scene_dir) if scene_dir else None
        self.scenes: list[Path] = []

        if self.scene_dir:
            scenes_subdir = self.scene_dir / "scenes"
            if scenes_subdir.exists():
                self.scenes_base = scenes_subdir
            else:
                self.scenes_base = self.scene_dir

            if split_file:
                self.scenes = self._load_split_file(split_file)
            else:
                self.scenes = sorted(self.scenes_base.glob("*.npz"))

        # Filter scenes with enough cameras
        self._filter_scenes_by_camera_count()

    def _load_split_file(self, split_file: str | Path) -> list[Path]:
        """Load scene list from split file."""
        split_path = Path(split_file)
        # If path exists as-is, use it directly
        if not split_path.exists() and not split_path.is_absolute():
            if self.scene_dir is None:
                raise ValueError("scene_dir must be set to use relative split_file")
            # Only prepend scene_dir for simple filenames like "train.txt"
            if split_path.parent == Path("."):
                split_path = self.scene_dir / split_file

        scenes = []
        with open(split_path) as f:
            for line in f:
                filename = line.strip()
                if filename:
                    scenes.append(self.scenes_base / filename)
        return scenes

    def _filter_scenes_by_camera_count(self) -> None:
        """Filter scenes that have at least min_cameras."""
        valid_scenes = []
        for scene_path in self.scenes:
            try:
                data = np.load(scene_path, allow_pickle=True)
                num_cameras = int(data["num_cameras"])
                if num_cameras >= self.min_cameras:
                    valid_scenes.append(scene_path)
            except Exception:
                continue

        print(
            f"MultiViewBallTrajectoryDataset: {len(valid_scenes)}/{len(self.scenes)} "
            f"scenes have >= {self.min_cameras} cameras"
        )
        self.scenes = valid_scenes

    def _load_scene_multiview(self, path: Path) -> dict[str, Tensor | list]:
        """Load a single scene with multiple camera views."""
        data = np.load(path, allow_pickle=True)

        # Parse metadata
        meta = json.loads(str(data["meta"]))
        num_cameras = int(data["num_cameras"])

        # Select random subset of cameras
        selected_cams = rng.sample(range(num_cameras), min(self.num_views, num_cameras))

        # Collect data from each camera
        ball_uv_list: list[Tensor] = []
        ball_vis_list: list[Tensor] = []
        court_kp_list: list[Tensor] = []
        court_vis_list: list[Tensor] = []
        camera_params_list: list[dict] = []

        for cam_idx in selected_cams:
            prefix = f"cam_{cam_idx}_"

            ball_uv = torch.from_numpy(data[f"{prefix}ball_uv"]).float()
            ball_vis = torch.from_numpy(data[f"{prefix}ball_visible"]).float()
            court_kp = torch.from_numpy(data[f"{prefix}court_kp_uv"]).float()
            court_vis = torch.from_numpy(data[f"{prefix}court_kp_visible"]).float()

            # Load camera params if available
            params_key = f"{prefix}params"
            if params_key in data:
                params_raw = data[params_key].item()
                if isinstance(params_raw, (bytes, bytearray)):
                    params_raw = params_raw.decode("utf-8")
                cam_params = (
                    json.loads(params_raw)
                    if isinstance(params_raw, str)
                    else params_raw
                )
            else:
                cam_params = {}

            ball_uv_list.append(ball_uv)
            ball_vis_list.append(ball_vis)
            court_kp_list.append(court_kp)
            court_vis_list.append(court_vis)
            camera_params_list.append(cam_params)

        # Stack into tensors: (N_cam, ...)
        return {
            "ball_uv": torch.stack(ball_uv_list, dim=0),  # (N_cam, T, 2)
            "ball_vis": torch.stack(ball_vis_list, dim=0),  # (N_cam, T)
            "court_kp": torch.stack(court_kp_list, dim=0),  # (N_cam, 20, 2)
            "court_vis": torch.stack(court_vis_list, dim=0),  # (N_cam, 20)
            "camera_params": camera_params_list,
            "num_views": torch.tensor(len(selected_cams)),
            "position_3d": torch.from_numpy(data["ball_pos_norm"]).float(),
            "velocity_3d": torch.from_numpy(data["ball_vel_world"]).float(),
            "seq_len": torch.tensor(meta["num_frames"]),
            "meta": meta,
        }

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.scenes)

    def __getitem__(self, idx: int) -> BLCSMultiViewSample:
        """Get a multi-view sample.

        Returns:
            Multi-view sample dictionary containing observations from
            multiple cameras for the same ball trajectory.

        """
        sample = self._load_scene_multiview(self.scenes[idx])

        # Apply augmentation
        if self.augment:
            sample = self._apply_augmentation(sample)

        return {
            "ball_uv": sample["ball_uv"],  # (N_cam, T, 2)
            "ball_mask": sample["ball_vis"],  # (N_cam, T)
            "court_kp": sample["court_kp"],  # (N_cam, 20, 2)
            "court_vis": sample["court_vis"],  # (N_cam, 20)
            "camera_params": sample["camera_params"],
            "num_views": sample["num_views"],
            "position_3d": sample["position_3d"],  # (T, 3)
            "velocity_3d": sample["velocity_3d"],  # (T, 3)
            "seq_len": sample["seq_len"],
        }

    def _apply_augmentation(
        self, sample: dict[str, Tensor | list]
    ) -> dict[str, Tensor | list]:
        """Apply data augmentation to a multi-view sample."""
        sample = {
            k: v.clone() if isinstance(v, Tensor) else v for k, v in sample.items()
        }

        # Add Gaussian noise to UV coordinates (per camera)
        ball_uv = sample["ball_uv"]
        if isinstance(ball_uv, Tensor):
            sample["ball_uv"] = add_gaussian_noise(ball_uv, self.uv_noise_std).clamp(
                0, 1
            )

        court_kp = sample["court_kp"]
        if isinstance(court_kp, Tensor):
            sample["court_kp"] = add_gaussian_noise(court_kp, self.uv_noise_std).clamp(
                0, 1
            )

        # Random visibility dropout for ball
        ball_vis = sample["ball_vis"]
        if isinstance(ball_vis, Tensor):
            sample["ball_vis"] = random_visibility_dropout(ball_vis, self.vis_drop_prob)

        return sample


def collate_multiview_trajectories(
    batch: list[BLCSMultiViewSample],
) -> BLCSMultiViewSample:
    """Collate function for multi-view trajectory batches.

    Handles variable sequence lengths and number of views by padding.

    Args:
        batch: List of multi-view samples.

    Returns:
        Collated batch with padded tensors.

    """
    max_views = max(sample["num_views"].item() for sample in batch)
    max_seq_len = max(sample["seq_len"].item() for sample in batch)

    ball_uv_batch = []
    ball_mask_batch = []
    court_kp_batch = []
    court_vis_batch = []
    position_3d_batch = []
    velocity_3d_batch = []
    seq_len_batch = []
    num_views_batch = []

    for sample in batch:
        n_views = sample["num_views"].item()
        seq_len = sample["seq_len"].item()
        pad_views = max_views - n_views
        pad_seq = max_seq_len - seq_len

        # Pad views dimension
        ball_uv = sample["ball_uv"]  # (N, T, 2)
        ball_mask = sample["ball_mask"]  # (N, T)
        court_kp = sample["court_kp"]  # (N, 20, 2)
        court_vis = sample["court_vis"]  # (N, 20)
        position_3d = sample["position_3d"]  # (T, 3)
        velocity_3d = sample["velocity_3d"]  # (T, 3)

        # Pad sequence dimension first
        if pad_seq > 0:
            ball_uv = torch.cat([ball_uv, torch.zeros(n_views, pad_seq, 2)], dim=1)
            ball_mask = torch.cat([ball_mask, torch.zeros(n_views, pad_seq)], dim=1)
            position_3d = torch.cat([position_3d, torch.zeros(pad_seq, 3)], dim=0)
            velocity_3d = torch.cat([velocity_3d, torch.zeros(pad_seq, 3)], dim=0)

        # Pad views dimension
        if pad_views > 0:
            ball_uv = torch.cat(
                [ball_uv, torch.zeros(pad_views, max_seq_len, 2)], dim=0
            )
            ball_mask = torch.cat(
                [ball_mask, torch.zeros(pad_views, max_seq_len)], dim=0
            )
            court_kp = torch.cat([court_kp, torch.zeros(pad_views, 20, 2)], dim=0)
            court_vis = torch.cat([court_vis, torch.zeros(pad_views, 20)], dim=0)

        ball_uv_batch.append(ball_uv)
        ball_mask_batch.append(ball_mask)
        court_kp_batch.append(court_kp)
        court_vis_batch.append(court_vis)
        position_3d_batch.append(position_3d)
        velocity_3d_batch.append(velocity_3d)
        seq_len_batch.append(sample["seq_len"])
        num_views_batch.append(sample["num_views"])

    return {
        "ball_uv": torch.stack(ball_uv_batch, dim=0),  # (B, N_max, T_max, 2)
        "ball_mask": torch.stack(ball_mask_batch, dim=0),  # (B, N_max, T_max)
        "court_kp": torch.stack(court_kp_batch, dim=0),  # (B, N_max, 20, 2)
        "court_vis": torch.stack(court_vis_batch, dim=0),  # (B, N_max, 20)
        "camera_params": [s["camera_params"] for s in batch],
        "num_views": torch.stack(num_views_batch, dim=0),  # (B,)
        "position_3d": torch.stack(position_3d_batch, dim=0),  # (B, T_max, 3)
        "velocity_3d": torch.stack(velocity_3d_batch, dim=0),  # (B, T_max, 3)
        "seq_len": torch.stack(seq_len_batch, dim=0),  # (B,)
    }
