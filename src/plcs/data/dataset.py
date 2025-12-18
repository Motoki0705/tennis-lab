"""PyTorch Dataset for PLCS training from pre-generated scene files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from omegaconf import DictConfig


class SceneDataset(Dataset[dict[str, Tensor]]):
    """Dataset for PLCS training from pre-generated scene files.

    This dataset loads pre-generated scene NPZ files and provides
    frame-level samples for training.
    """

    def __init__(
        self,
        scene_dir: str | Path,
        config: DictConfig | None = None,
        augment: bool = True,
        camera_mode: str = "random",  # "random", "all", or specific index
    ) -> None:
        """Initialize the scene dataset.

        Args:
            scene_dir: Directory containing scene NPZ files.
            config: Configuration dictionary.
            augment: Whether to apply data augmentation.
            camera_mode: How to select cameras ("random", "all", or camera index).

        """
        from pathlib import Path

        self.scene_dir = Path(scene_dir)
        self.config = config or {}
        self.augment = augment
        self.camera_mode = camera_mode

        # Augmentation parameters
        data_cfg = self.config.get("data", {})
        self.kp_noise_std = data_cfg.get("keypoint_noise_std", 0.01)
        self.visibility_drop_prob = data_cfg.get("visibility_drop_prob", 0.05)

        # Index all scene files
        scenes_subdir = self.scene_dir / "scenes"
        self.scene_files = sorted(scenes_subdir.glob("scene_*.npz"))
        if not self.scene_files:
            raise ValueError(f"No scene files found in {scenes_subdir}")

        print(f"SceneDataset: found {len(self.scene_files)} scene files")

        # Build index: (scene_idx, frame_idx, camera_idx)
        self._build_index()

    def _build_index(self) -> None:
        """Build sample index from scene files."""
        from src.plcs.generate_dataset.scene_generator import SceneGenerator

        self.index: list[tuple[int, int, int]] = []
        self.scenes: list = []

        for scene_idx, scene_file in enumerate(self.scene_files):
            scene = SceneGenerator.load_scene(scene_file)
            self.scenes.append(scene)

            num_frames = scene.meta["num_frames"]
            num_cameras = len(scene.cameras)

            if self.camera_mode == "all":
                # All cameras, all frames
                for frame_idx in range(num_frames):
                    for cam_idx in range(num_cameras):
                        self.index.append((scene_idx, frame_idx, cam_idx))
            elif self.camera_mode == "random":
                # Random camera per frame (selected at getitem)
                for frame_idx in range(num_frames):
                    self.index.append((scene_idx, frame_idx, -1))  # -1 = random
            else:
                # Specific camera
                cam_idx = int(self.camera_mode)
                if cam_idx < num_cameras:
                    for frame_idx in range(num_frames):
                        self.index.append((scene_idx, frame_idx, cam_idx))

        print(f"SceneDataset: indexed {len(self.index)} samples")

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get a sample by index.

        Args:
            idx: Sample index.

        Returns:
            dict: Sample dictionary with input features and targets.

        """
        import random as rng

        scene_idx, frame_idx, cam_idx = self.index[idx]
        scene = self.scenes[scene_idx]

        # Select camera
        if cam_idx < 0:
            cam_idx = rng.randint(0, len(scene.cameras) - 1)

        cam = scene.cameras[cam_idx]

        # Get keypoints
        human_kp = torch.from_numpy(cam.human_kp_uv[frame_idx].copy())  # (17, 2)
        court_kp = torch.from_numpy(cam.court_kp_uv[frame_idx].copy())  # (20, 2)
        human_vis = torch.from_numpy(cam.human_kp_visible[frame_idx].copy())
        court_vis = torch.from_numpy(cam.court_kp_visible[frame_idx].copy())

        # Get targets
        position = torch.from_numpy(scene.position[frame_idx].copy())
        rotation = torch.from_numpy(scene.rotation[frame_idx].copy())

        # Apply augmentation
        if self.augment:
            human_kp, human_vis = self._augment_keypoints(human_kp, human_vis)
            court_kp, court_vis = self._augment_keypoints(court_kp, court_vis)

        # Apply visibility mask
        human_kp_masked = human_kp.clone()
        human_kp_masked[~human_vis] = 0.0

        court_kp_masked = court_kp.clone()
        court_kp_masked[~court_vis] = 0.0

        return {
            "human_kp": human_kp_masked.flatten().float(),  # (34,)
            "court_kp": court_kp_masked.flatten().float(),  # (40,)
            "human_vis": human_vis.float(),  # (17,)
            "court_vis": court_vis.float(),  # (20,)
            "position": position.float(),  # (3,)
            "rotation": rotation.float(),  # (2,)
        }

    def _augment_keypoints(
        self,
        keypoints: Tensor,
        visibility: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply augmentation to keypoints.

        Args:
            keypoints: Keypoint coordinates, shape (N, 2).
            visibility: Visibility mask, shape (N,).

        Returns:
            tuple: Augmented keypoints and visibility.

        """
        # Add Gaussian noise
        if self.kp_noise_std > 0:
            noise = torch.randn_like(keypoints) * self.kp_noise_std
            keypoints = keypoints + noise

        # Random visibility dropout
        if self.visibility_drop_prob > 0:
            drop_mask = torch.rand(visibility.shape) < self.visibility_drop_prob
            visibility = visibility & ~drop_mask

        return keypoints, visibility
