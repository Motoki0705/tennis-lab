"""Sequence dataset for PLCS.

Provides fixed-length frame sequences for training sequential PLCS models.
"""

from __future__ import annotations

import random as rng
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.base.data.augmentation import augment_keypoints
from src.plcs.data.types import PLCSSequenceBatch
from src.plcs.generate_dataset.io.scene_loader import load_scene

if TYPE_CHECKING:
    from omegaconf import DictConfig


class SceneSequenceDataset(Dataset[PLCSSequenceBatch]):
    """Dataset that returns fixed-length temporal clips from PLCS scenes.

    Each sample corresponds to a contiguous window of frames from a single
    scene and camera.
    """

    def __init__(
        self,
        scene_dir: str | Path,
        config: DictConfig | None = None,
        augment: bool = True,
        camera_mode: str = "random",  # "random", "all", or specific index
    ) -> None:
        super().__init__()

        self.scene_dir = Path(scene_dir)
        self.config = config or {}
        self.augment = augment
        self.camera_mode = camera_mode

        data_cfg = self.config.get("data", {})
        self.seq_len: int = int(data_cfg.get("seq_len", 16))
        self.seq_stride: int = int(data_cfg.get("seq_stride", self.seq_len))
        self.kp_noise_std: float = float(data_cfg.get("keypoint_noise_std", 0.01))
        self.visibility_drop_prob: float = float(
            data_cfg.get("visibility_drop_prob", 0.05)
        )

        # Index all scene files
        scenes_subdir = self.scene_dir / "scenes"
        self.scene_files = sorted(scenes_subdir.glob("scene_*.npz"))
        if not self.scene_files:
            raise ValueError(f"No scene files found in {scenes_subdir}")

        print(f"SceneSequenceDataset: found {len(self.scene_files)} scene files")

        # Build index: (scene_idx, cam_idx, start_frame)
        self._build_index()

    def _build_index(self) -> None:
        """Build sample index from scene files.

        Index entries are (scene_idx, cam_idx, start_frame).
        """
        self.index: list[tuple[int, int, int]] = []
        self.scenes: list = []

        for scene_idx, scene_file in enumerate(self.scene_files):
            scene = load_scene(scene_file)
            self.scenes.append(scene)

            num_frames = scene["meta"]["num_frames"]
            num_cameras = len(scene["cameras"])

            if num_frames < self.seq_len:
                # Skip scenes shorter than the desired sequence length
                continue

            max_start = num_frames - self.seq_len

            if self.camera_mode == "all":
                # All cameras, sliding window over frames
                for start in range(0, max_start + 1, self.seq_stride):
                    for cam_idx in range(num_cameras):
                        self.index.append((scene_idx, cam_idx, start))
            elif self.camera_mode == "random":
                # Camera is selected randomly at __getitem__ time
                for start in range(0, max_start + 1, self.seq_stride):
                    self.index.append((scene_idx, -1, start))  # -1 = random
            else:
                # Specific camera index
                cam_idx = int(self.camera_mode)
                if cam_idx < num_cameras:
                    for start in range(0, max_start + 1, self.seq_stride):
                        self.index.append((scene_idx, cam_idx, start))

        print(
            "SceneSequenceDataset: indexed "
            f"{len(self.index)} sequences (seq_len={self.seq_len})"
        )

    def __len__(self) -> int:
        """Return the number of sequence samples."""
        return len(self.index)

    def __getitem__(self, idx: int) -> PLCSSequenceBatch:
        """Get a sequence sample by index.

        Returns a dictionary containing:
            - human_kp: (T, 17, 2)
            - court_kp: (1, 20, 2) - aggregated over time (court is time-invariant)
            - human_vis: (T, 17)
            - court_vis: (1, 20) - aggregated over time
            - position: (T, 3)
            - rotation: (T, 2)
        """
        scene_idx, cam_idx, start = self.index[idx]
        scene = self.scenes[scene_idx]

        # Select camera
        if cam_idx < 0:
            cam_idx = rng.randint(0, len(scene["cameras"]) - 1)

        cam = scene["cameras"][cam_idx]
        end = start + self.seq_len

        # Keypoints and visibility
        human_kp = torch.from_numpy(cam["human_kp_uv"][start:end].copy())
        court_kp = torch.from_numpy(cam["court_kp_uv"][start:end].copy())
        human_vis = torch.from_numpy(cam["human_kp_visible"][start:end].copy())
        court_vis = torch.from_numpy(cam["court_kp_visible"][start:end].copy())

        # Targets
        position = torch.from_numpy(scene["position"][start:end].copy())
        rotation = torch.from_numpy(scene["rotation"][start:end].copy())

        # Apply augmentation to human keypoints
        if self.augment:
            human_kp, human_vis = augment_keypoints(
                human_kp, human_vis, self.kp_noise_std, self.visibility_drop_prob
            )
            court_kp, court_vis = augment_keypoints(
                court_kp, court_vis, self.kp_noise_std, self.visibility_drop_prob
            )

        # Aggregate court keypoints over time (court is time-invariant)
        court_kp_agg, court_vis_agg = self._aggregate_court_keypoints(
            court_kp, court_vis
        )  # (1, 20, 2), (1, 20)

        # Apply visibility mask (zero-out invisible keypoints)
        human_kp_masked = human_kp * human_vis.unsqueeze(-1)
        court_kp_masked = court_kp_agg * court_vis_agg.unsqueeze(-1)

        return {
            "human_kp": human_kp_masked.float(),
            "court_kp": court_kp_masked.float(),
            "human_vis": human_vis.float(),
            "court_vis": court_vis_agg.float(),
            "position": position.float(),
            "rotation": rotation.float(),
        }

    def _aggregate_court_keypoints(
        self,
        court_kp: Tensor,
        court_vis: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Aggregate court keypoints over time using visibility-weighted mean.

        Court keypoints are time-invariant (the court doesn't move), so we
        aggregate them into a single representation.

        Args:
            court_kp: Court keypoints, shape (T, 20, 2).
            court_vis: Court visibility, shape (T, 20).

        Returns:
            tuple: Aggregated court_kp (1, 20, 2) and court_vis (1, 20).

        """
        # Visibility-weighted mean over time
        vis_weight = court_vis.unsqueeze(-1).float()  # (T, 20, 1)
        vis_sum = vis_weight.sum(dim=0, keepdim=True).clamp(min=1e-8)  # (1, 20, 1)
        court_kp_agg = (court_kp * vis_weight).sum(
            dim=0, keepdim=True
        ) / vis_sum  # (1, 20, 2)

        # Aggregated visibility: keypoint is visible if visible in any frame
        court_vis_agg = (court_vis.sum(dim=0, keepdim=True) > 0).float()  # (1, 20)

        return court_kp_agg, court_vis_agg
