"""Multi-view dataset for PLCS training from pre-generated scene files.

This module provides datasets that return observations from multiple cameras
simultaneously for the same frame/sequence, enabling multi-camera fusion models.
"""

from __future__ import annotations

import random as rng
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.base.data.augmentation import augment_keypoints
from src.plcs.data.types import PLCSMultiViewBatch
from src.plcs.generate_dataset.io.scene_loader import load_scene

if TYPE_CHECKING:
    from omegaconf import DictConfig


class MultiViewSceneDataset(Dataset[PLCSMultiViewBatch]):
    """Dataset for multi-view PLCS training from pre-generated scene files.

    Unlike SceneDataset which returns single-camera samples, this dataset
    returns observations from multiple cameras for the same frame, enabling
    multi-camera fusion approaches.
    """

    def __init__(
        self,
        scene_dir: str | Path,
        config: DictConfig | None = None,
        augment: bool = True,
        num_views: int = 2,
        min_cameras: int = 2,
    ) -> None:
        """Initialize the multi-view scene dataset.

        Args:
            scene_dir: Directory containing scene NPZ files.
            config: Configuration dictionary.
            augment: Whether to apply data augmentation.
            num_views: Number of camera views to return per sample.
            min_cameras: Minimum cameras required in a scene (skip otherwise).

        """
        self.scene_dir = Path(scene_dir)
        self.config = config or {}
        self.augment = augment
        self.num_views = num_views
        self.min_cameras = min_cameras

        # Augmentation parameters
        data_cfg = self.config.get("data", {})
        self.kp_noise_std = data_cfg.get("keypoint_noise_std", 0.01)
        self.visibility_drop_prob = data_cfg.get("visibility_drop_prob", 0.05)

        # Index all scene files
        scenes_subdir = self.scene_dir / "scenes"
        self.scene_files = sorted(scenes_subdir.glob("scene_*.npz"))
        if not self.scene_files:
            raise ValueError(f"No scene files found in {scenes_subdir}")

        print(f"MultiViewSceneDataset: found {len(self.scene_files)} scene files")

        # Build index: (scene_idx, frame_idx)
        self._build_index()

    def _build_index(self) -> None:
        """Build sample index from scene files.

        Only includes scenes with at least min_cameras cameras.
        """
        self.index: list[tuple[int, int]] = []
        self.scenes: list = []

        for _scene_idx, scene_file in enumerate(self.scene_files):
            scene = load_scene(scene_file)
            num_cameras = len(scene["cameras"])

            # Skip scenes without enough cameras
            if num_cameras < self.min_cameras:
                continue

            self.scenes.append(scene)
            actual_scene_idx = len(self.scenes) - 1

            num_frames = scene["meta"]["num_frames"]
            for frame_idx in range(num_frames):
                self.index.append((actual_scene_idx, frame_idx))

        print(
            f"MultiViewSceneDataset: indexed {len(self.index)} samples "
            f"from {len(self.scenes)} scenes (min_cameras={self.min_cameras})"
        )

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.index)

    def __getitem__(self, idx: int) -> PLCSMultiViewBatch:
        """Get a multi-view sample by index.

        Args:
            idx: Sample index.

        Returns:
            Multi-view sample dictionary with observations from multiple cameras.

        """
        scene_idx, frame_idx = self.index[idx]
        scene = self.scenes[scene_idx]
        num_cameras = len(scene["cameras"])

        # Select random subset of cameras
        selected_cams = rng.sample(range(num_cameras), min(self.num_views, num_cameras))

        # Collect data from each camera
        human_kp_list: list[Tensor] = []
        court_kp_list: list[Tensor] = []
        human_vis_list: list[Tensor] = []
        court_vis_list: list[Tensor] = []
        camera_params_list: list[dict] = []

        for cam_idx in selected_cams:
            cam = scene["cameras"][cam_idx]

            # Get keypoints
            human_kp = torch.from_numpy(cam["human_kp_uv"][frame_idx].copy())
            court_kp = torch.from_numpy(cam["court_kp_uv"][frame_idx].copy())
            human_vis = torch.from_numpy(cam["human_kp_visible"][frame_idx].copy())
            court_vis = torch.from_numpy(cam["court_kp_visible"][frame_idx].copy())

            # Apply augmentation
            if self.augment:
                human_kp, human_vis = augment_keypoints(
                    human_kp, human_vis, self.kp_noise_std, self.visibility_drop_prob
                )
                court_kp, court_vis = augment_keypoints(
                    court_kp, court_vis, self.kp_noise_std, self.visibility_drop_prob
                )

            # Apply visibility mask
            human_kp_masked = human_kp.clone()
            human_kp_masked[~human_vis] = 0.0
            court_kp_masked = court_kp.clone()
            court_kp_masked[~court_vis] = 0.0

            human_kp_list.append(human_kp_masked)
            court_kp_list.append(court_kp_masked)
            human_vis_list.append(human_vis)
            court_vis_list.append(court_vis)
            camera_params_list.append(cam.get("params", {}))

        # Stack into tensors: (N_cam, ...)
        human_kp_stacked = torch.stack(human_kp_list, dim=0)  # (N_cam, 17, 2)
        court_kp_stacked = torch.stack(court_kp_list, dim=0)  # (N_cam, 20, 2)
        human_vis_stacked = torch.stack(human_vis_list, dim=0)  # (N_cam, 17)
        court_vis_stacked = torch.stack(court_vis_list, dim=0)  # (N_cam, 20)

        # Get targets (same for all cameras)
        position = torch.from_numpy(scene["position"][frame_idx].copy())
        rotation = torch.from_numpy(scene["rotation"][frame_idx].copy())

        return {
            "human_kp": human_kp_stacked.float(),
            "court_kp": court_kp_stacked.float(),
            "human_vis": human_vis_stacked.float(),
            "court_vis": court_vis_stacked.float(),
            "camera_params": camera_params_list,
            "num_views": torch.tensor(len(selected_cams)),
            "position": position.float(),
            "rotation": rotation.float(),
        }


def collate_multiview(
    batch: list[PLCSMultiViewBatch],
) -> PLCSMultiViewBatch:
    """Collate function for multi-view batches.

    Handles variable number of views by padding to max views in batch.

    Args:
        batch: List of multi-view samples.

    Returns:
        Collated batch with padded tensors.

    """
    max_views = max(sample["num_views"].item() for sample in batch)

    human_kp_batch = []
    court_kp_batch = []
    human_vis_batch = []
    court_vis_batch = []
    position_batch = []
    rotation_batch = []
    num_views_batch = []

    for sample in batch:
        n_views = sample["num_views"].item()
        pad_views = max_views - n_views

        # Pad with zeros if needed
        if pad_views > 0:
            human_kp = torch.cat(
                [
                    sample["human_kp"],
                    torch.zeros(pad_views, 17, 2),
                ],
                dim=0,
            )
            court_kp = torch.cat(
                [
                    sample["court_kp"],
                    torch.zeros(pad_views, 20, 2),
                ],
                dim=0,
            )
            human_vis = torch.cat(
                [
                    sample["human_vis"],
                    torch.zeros(pad_views, 17),
                ],
                dim=0,
            )
            court_vis = torch.cat(
                [
                    sample["court_vis"],
                    torch.zeros(pad_views, 20),
                ],
                dim=0,
            )
        else:
            human_kp = sample["human_kp"]
            court_kp = sample["court_kp"]
            human_vis = sample["human_vis"]
            court_vis = sample["court_vis"]

        human_kp_batch.append(human_kp)
        court_kp_batch.append(court_kp)
        human_vis_batch.append(human_vis)
        court_vis_batch.append(court_vis)
        position_batch.append(sample["position"])
        rotation_batch.append(sample["rotation"])
        num_views_batch.append(sample["num_views"])

    return {
        "human_kp": torch.stack(human_kp_batch, dim=0),  # (B, N_max, 17, 2)
        "court_kp": torch.stack(court_kp_batch, dim=0),  # (B, N_max, 20, 2)
        "human_vis": torch.stack(human_vis_batch, dim=0),  # (B, N_max, 17)
        "court_vis": torch.stack(court_vis_batch, dim=0),  # (B, N_max, 20)
        "camera_params": [s["camera_params"] for s in batch],  # List of lists
        "num_views": torch.stack(num_views_batch, dim=0),  # (B,)
        "position": torch.stack(position_batch, dim=0),  # (B, 3)
        "rotation": torch.stack(rotation_batch, dim=0),  # (B, 2)
    }
