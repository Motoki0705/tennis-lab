"""PyTorch Dataset for PLCS training from pre-generated scene files."""

from __future__ import annotations

import random as rng
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from src.base.data.augmentation import augment_keypoints
from src.plcs.data.index_cache import (
    compute_config_hash,
    compute_scene_files_hash,
    get_index_cache_path,
    load_cached_index,
    save_cached_index,
)
from src.plcs.data.scene_cache import (
    extract_scene_meta_parallel,
    get_scene_cache,
)
from src.plcs.data.types import PLCSFrameBatch

if TYPE_CHECKING:
    from omegaconf import DictConfig


class SceneDataset(Dataset[PLCSFrameBatch]):
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
        cache_maxsize: int = 128,  # LRU cache size for scenes
        parallel_workers: int = 8,  # Threads for metadata extraction
    ) -> None:
        """Initialize the scene dataset.

        Args:
            scene_dir: Directory containing scene NPZ files.
            config: Configuration dictionary.
            augment: Whether to apply data augmentation.
            camera_mode: How to select cameras ("random", "all", or camera index).
            cache_maxsize: Maximum number of scenes to keep in LRU cache.
            parallel_workers: Number of threads for parallel metadata extraction.

        """
        self.scene_dir = Path(scene_dir)
        self.config = config or {}
        self.augment = augment
        self.camera_mode = camera_mode
        self.parallel_workers = parallel_workers

        # Augmentation parameters
        data_cfg = self.config.get("data", {})
        self.kp_noise_std = data_cfg.get("keypoint_noise_std", 0.01)
        self.visibility_drop_prob = data_cfg.get("visibility_drop_prob", 0.05)

        # Get shared scene cache (lazy loading with LRU)
        self._scene_cache = get_scene_cache(maxsize=cache_maxsize)

        # Scene paths (no longer loading all scenes into memory)
        self.scene_paths: list[Path] = []

        # Index all scene files
        scenes_subdir = self.scene_dir / "scenes"
        self.scene_files = sorted(scenes_subdir.glob("scene_*.npz"))
        if not self.scene_files:
            raise ValueError(f"No scene files found in {scenes_subdir}")

        print(f"SceneDataset: found {len(self.scene_files)} scene files")

        # Build index: (scene_idx, frame_idx, camera_idx)
        self._build_index()

    def _build_index(self) -> None:
        """Build sample index from scene files with parallel metadata extraction."""
        # Check for cached index
        config_hash = compute_config_hash(
            {"camera_mode": self.camera_mode},
            ["camera_mode"],
        )
        scene_hash = compute_scene_files_hash(self.scene_files)
        cache_path = get_index_cache_path(self.scene_dir, "SceneDataset", config_hash)

        cached = load_cached_index(cache_path, scene_hash)
        if cached is not None:
            self.index = cached.index
            self.scene_paths = [Path(m["path"]) for m in cached.scene_metas]
            print(f"SceneDataset: loaded cached index ({len(self.index)} samples)")
            return

        # Extract metadata in parallel (no full scene loading!)
        print(
            f"SceneDataset: extracting metadata from "
            f"{len(self.scene_files)} scenes..."
        )
        scene_metas = extract_scene_meta_parallel(
            self.scene_files,
            max_workers=self.parallel_workers,
        )

        # Build index from metadata only
        self.index = []
        self.scene_paths = []

        for meta in scene_metas:
            self.scene_paths.append(meta.scene_path)
            actual_idx = len(self.scene_paths) - 1

            if self.camera_mode == "all":
                # All cameras, all frames
                for frame_idx in range(meta.num_frames):
                    for cam_idx in range(meta.num_cameras):
                        self.index.append((actual_idx, frame_idx, cam_idx))
            elif self.camera_mode == "random":
                # Random camera per frame (selected at getitem)
                for frame_idx in range(meta.num_frames):
                    self.index.append((actual_idx, frame_idx, -1))  # -1 = random
            else:
                # Specific camera
                cam_idx = int(self.camera_mode)
                if cam_idx < meta.num_cameras:
                    for frame_idx in range(meta.num_frames):
                        self.index.append((actual_idx, frame_idx, cam_idx))

        # Save to cache
        scene_metas_serializable = [
            {
                "path": str(m.scene_path),
                "num_frames": m.num_frames,
                "num_cameras": m.num_cameras,
            }
            for m in scene_metas
        ]
        save_cached_index(
            cache_path, self.index, scene_metas_serializable, config_hash, scene_hash
        )

        print(f"SceneDataset: indexed {len(self.index)} samples")

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.index)

    def __getitem__(self, idx: int) -> PLCSFrameBatch:
        """Get a sample by index.

        Args:
            idx: Sample index.

        Returns:
            Sample dictionary with input features and targets.

        """
        scene_idx, frame_idx, cam_idx = self.index[idx]
        # Load scene on-demand via LRU cache
        scene = self._scene_cache.get(self.scene_paths[scene_idx])

        # Select camera
        if cam_idx < 0:
            cam_idx = rng.randint(0, len(scene["cameras"]) - 1)

        cam = scene["cameras"][cam_idx]

        # Get keypoints
        human_kp = torch.from_numpy(cam["human_kp_uv"][frame_idx].copy())  # (17, 2)
        court_kp = torch.from_numpy(cam["court_kp_uv"][frame_idx].copy())  # (20, 2)
        human_vis = torch.from_numpy(cam["human_kp_visible"][frame_idx].copy())
        court_vis = torch.from_numpy(cam["court_kp_visible"][frame_idx].copy())

        # Get targets
        position = torch.from_numpy(scene["position"][frame_idx].copy())
        rotation = torch.from_numpy(scene["rotation"][frame_idx].copy())

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

        return {
            "human_kp": human_kp_masked.flatten().float(),  # (34,)
            "court_kp": court_kp_masked.flatten().float(),  # (40,)
            "human_vis": human_vis.float(),  # (17,)
            "court_vis": court_vis.float(),  # (20,)
            "position": position.float(),  # (3,)
            "rotation": rotation.float(),  # (2,)
        }
