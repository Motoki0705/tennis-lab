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
from src.plcs.data.types import (
    PLCSMultiViewBatch,
    PLCSMultiViewBatchCollated,
    PLCSMultiViewSequenceBatch,
    PLCSMultiViewSequenceBatchCollated,
)

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
        cache_maxsize: int = 128,  # LRU cache size for scenes
        parallel_workers: int = 8,  # Threads for metadata extraction
    ) -> None:
        """Initialize the multi-view scene dataset.

        Args:
            scene_dir: Directory containing scene NPZ files.
            config: Configuration dictionary.
            augment: Whether to apply data augmentation.
            num_views: Number of camera views to return per sample.
            min_cameras: Minimum cameras required in a scene (skip otherwise).
            cache_maxsize: Maximum number of scenes to keep in LRU cache.
            parallel_workers: Number of threads for parallel metadata extraction.

        """
        self.scene_dir = Path(scene_dir)
        self.config = config or {}
        self.augment = augment
        self.num_views = num_views
        self.min_cameras = min_cameras
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

        print(f"MultiViewSceneDataset: found {len(self.scene_files)} scene files")

        # Build index: (scene_idx, frame_idx)
        self._build_index()

    def _build_index(self) -> None:
        """Build sample index with parallel metadata extraction.

        Only includes scenes with at least min_cameras cameras.
        """
        # Check for cached index
        config_hash = compute_config_hash(
            {"min_cameras": self.min_cameras},
            ["min_cameras"],
        )
        scene_hash = compute_scene_files_hash(self.scene_files)
        cache_path = get_index_cache_path(
            self.scene_dir, "MultiViewSceneDataset", config_hash
        )

        cached = load_cached_index(cache_path, scene_hash)
        if cached is not None:
            self.index = cached.index
            self.scene_paths = [Path(m["path"]) for m in cached.scene_metas]
            print(
                f"MultiViewSceneDataset: loaded cached index "
                f"({len(self.index)} samples from {len(self.scene_paths)} scenes)"
            )
            return

        # Extract metadata in parallel (no full scene loading!)
        print(
            f"MultiViewSceneDataset: extracting metadata from "
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
            # Skip scenes without enough cameras
            if meta.num_cameras < self.min_cameras:
                continue

            self.scene_paths.append(meta.scene_path)
            actual_scene_idx = len(self.scene_paths) - 1

            for frame_idx in range(meta.num_frames):
                self.index.append((actual_scene_idx, frame_idx))

        # Save to cache
        scene_metas_serializable = [
            {
                "path": str(m.scene_path),
                "num_frames": m.num_frames,
                "num_cameras": m.num_cameras,
            }
            for m in scene_metas
            if m.num_cameras >= self.min_cameras
        ]
        save_cached_index(
            cache_path, self.index, scene_metas_serializable, config_hash, scene_hash
        )

        print(
            f"MultiViewSceneDataset: indexed {len(self.index)} samples "
            f"from {len(self.scene_paths)} scenes (min_cameras={self.min_cameras})"
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
        # Load scene on-demand via LRU cache
        scene = self._scene_cache.get(self.scene_paths[scene_idx])
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
) -> PLCSMultiViewBatchCollated:
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


class MultiViewSequenceDataset(Dataset[PLCSMultiViewSequenceBatch]):
    """Dataset for multi-view sequential PLCS training.

    Returns observations from multiple cameras over a temporal sequence,
    enabling multi-camera sequential fusion models.

    Supports dynamic range-based sampling for views and sequence length:
        - num_views_range: [min, max] - randomly sample view count per sample
        - seq_len_range: [min, max] - randomly sample sequence length per sample
    """

    def __init__(
        self,
        scene_dir: str | Path,
        config: DictConfig | None = None,
        augment: bool = True,
        num_views: int = 2,
        min_cameras: int = 2,
        cache_maxsize: int = 128,  # LRU cache size for scenes
        parallel_workers: int = 8,  # Threads for metadata extraction
    ) -> None:
        """Initialize the multi-view sequence dataset.

        Args:
            scene_dir: Directory containing scene NPZ files.
            config: Configuration dictionary with data.seq_len, etc.
            augment: Whether to apply data augmentation.
            num_views: Number of camera views to return per sample.
            min_cameras: Minimum cameras required in a scene.
            cache_maxsize: Maximum number of scenes to keep in LRU cache.
            parallel_workers: Number of threads for parallel metadata extraction.

        """
        self.scene_dir = Path(scene_dir)
        self.config = config or {}
        self.augment = augment
        self.num_views = num_views
        self.min_cameras = min_cameras
        self.parallel_workers = parallel_workers

        # Sequence parameters
        data_cfg = self.config.get("data", {})
        self.seq_len: int = int(data_cfg.get("seq_len", 16))
        self.seq_stride: int = int(data_cfg.get("seq_stride", self.seq_len))
        self.kp_noise_std = data_cfg.get("keypoint_noise_std", 0.01)
        self.visibility_drop_prob = data_cfg.get("visibility_drop_prob", 0.05)

        # Range sampling (optional)
        # Format: [min, max] inclusive
        self.num_views_range: tuple[int, int] | None = None
        self.seq_len_range: tuple[int, int] | None = None

        if "num_views_range" in data_cfg:
            r = data_cfg["num_views_range"]
            self.num_views_range = (int(r[0]), int(r[1]))

        if "seq_len_range" in data_cfg:
            r = data_cfg["seq_len_range"]
            self.seq_len_range = (int(r[0]), int(r[1]))

        # Get shared scene cache (lazy loading with LRU)
        self._scene_cache = get_scene_cache(maxsize=cache_maxsize)

        # Scene paths (no longer loading all scenes into memory)
        self.scene_paths: list[Path] = []

        # Index all scene files
        scenes_subdir = self.scene_dir / "scenes"
        self.scene_files = sorted(scenes_subdir.glob("scene_*.npz"))
        if not self.scene_files:
            raise ValueError(f"No scene files found in {scenes_subdir}")

        print(f"MultiViewSequenceDataset: found {len(self.scene_files)} scene files")

        # Build index
        self._build_index()

    def _build_index(self) -> None:
        """Build sample index with parallel metadata extraction."""
        # Determine minimum seq_len for indexing
        min_seq_for_index = self.seq_len
        if self.seq_len_range is not None:
            min_seq_for_index = self.seq_len_range[0]

        # Check for cached index
        config_hash = compute_config_hash(
            {
                "min_cameras": self.min_cameras,
                "min_seq_for_index": min_seq_for_index,
                "seq_stride": self.seq_stride,
            },
            ["min_cameras", "min_seq_for_index", "seq_stride"],
        )
        scene_hash = compute_scene_files_hash(self.scene_files)
        cache_path = get_index_cache_path(
            self.scene_dir, "MultiViewSequenceDataset", config_hash
        )

        cached = load_cached_index(cache_path, scene_hash)
        if cached is not None:
            self.index = cached.index
            self.scene_paths = [Path(m["path"]) for m in cached.scene_metas]
            print(
                f"MultiViewSequenceDataset: loaded cached index "
                f"({len(self.index)} samples from {len(self.scene_paths)} scenes)"
            )
            return

        # Extract metadata in parallel (no full scene loading!)
        print(
            f"MultiViewSequenceDataset: extracting metadata from "
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
            if meta.num_cameras < self.min_cameras:
                continue
            if meta.num_frames < min_seq_for_index:
                continue

            self.scene_paths.append(meta.scene_path)
            actual_scene_idx = len(self.scene_paths) - 1

            max_start = meta.num_frames - min_seq_for_index
            for start in range(0, max_start + 1, self.seq_stride):
                self.index.append((actual_scene_idx, start))

        # Save to cache
        scene_metas_serializable = [
            {
                "path": str(m.scene_path),
                "num_frames": m.num_frames,
                "num_cameras": m.num_cameras,
            }
            for m in scene_metas
            if m.num_cameras >= self.min_cameras and m.num_frames >= min_seq_for_index
        ]
        save_cached_index(
            cache_path, self.index, scene_metas_serializable, config_hash, scene_hash
        )

        print(
            f"MultiViewSequenceDataset: indexed {len(self.index)} samples "
            f"from {len(self.scene_paths)} scenes"
        )

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.index)

    def __getitem__(self, idx: int) -> PLCSMultiViewSequenceBatch:
        """Get a multi-view sequence sample by index.

        Returns:
            Multi-view sequence sample with shape (N_cam, T, ...).
            When range sampling is enabled, N_cam and T may vary per sample.

        """
        scene_idx, start = self.index[idx]
        # Load scene on-demand via LRU cache
        scene = self._scene_cache.get(self.scene_paths[scene_idx])
        num_cameras = len(scene["cameras"])
        num_frames = scene["meta"]["num_frames"]

        # Determine actual seq_len for this sample
        if self.seq_len_range is not None:
            min_seq, max_seq = self.seq_len_range
            max_possible = min(max_seq, num_frames - start)
            actual_seq_len = rng.randint(min_seq, max_possible)
        else:
            actual_seq_len = self.seq_len

        end = start + actual_seq_len

        # Determine actual num_views for this sample
        if self.num_views_range is not None:
            min_views, max_views = self.num_views_range
            max_possible_views = min(max_views, num_cameras)
            actual_num_views = rng.randint(min_views, max_possible_views)
        else:
            actual_num_views = min(self.num_views, num_cameras)

        # Select random subset of cameras
        selected_cams = rng.sample(range(num_cameras), actual_num_views)

        # Collect data from each camera
        human_kp_list: list[Tensor] = []
        court_kp_list: list[Tensor] = []
        human_vis_list: list[Tensor] = []
        court_vis_list: list[Tensor] = []
        camera_params_list: list[dict] = []

        for cam_idx in selected_cams:
            cam = scene["cameras"][cam_idx]

            # Get sequence of keypoints: (T, K, 2)
            human_kp = torch.from_numpy(cam["human_kp_uv"][start:end].copy())
            court_kp = torch.from_numpy(cam["court_kp_uv"][start:end].copy())
            human_vis = torch.from_numpy(cam["human_kp_visible"][start:end].copy())
            court_vis = torch.from_numpy(cam["court_kp_visible"][start:end].copy())

            # Apply augmentation
            if self.augment:
                human_kp, human_vis = augment_keypoints(
                    human_kp, human_vis, self.kp_noise_std, self.visibility_drop_prob
                )
                court_kp, court_vis = augment_keypoints(
                    court_kp, court_vis, self.kp_noise_std, self.visibility_drop_prob
                )

            # Apply visibility mask
            human_kp_masked = human_kp * human_vis.unsqueeze(-1)
            court_kp_masked = court_kp * court_vis.unsqueeze(-1)

            human_kp_list.append(human_kp_masked)  # (T, 17, 2)
            court_kp_list.append(court_kp_masked)  # (T, 20, 2)
            human_vis_list.append(human_vis)  # (T, 17)
            court_vis_list.append(court_vis)  # (T, 20)
            camera_params_list.append(cam.get("params", {}))

        # Stack into tensors: (N_cam, T, ...)
        human_kp_stacked = torch.stack(human_kp_list, dim=0)  # (N_cam, T, 17, 2)
        court_kp_stacked = torch.stack(court_kp_list, dim=0)  # (N_cam, T, 20, 2)
        human_vis_stacked = torch.stack(human_vis_list, dim=0)  # (N_cam, T, 17)
        court_vis_stacked = torch.stack(court_vis_list, dim=0)  # (N_cam, T, 20)

        # Create masks (all True since no padding yet at sample level)
        view_mask = torch.ones(actual_num_views, dtype=torch.bool)
        seq_mask = torch.ones(actual_seq_len, dtype=torch.bool)

        # Get targets (same for all cameras)
        position = torch.from_numpy(scene["position"][start:end].copy())  # (T, 3)
        rotation = torch.from_numpy(scene["rotation"][start:end].copy())  # (T, 2)

        return {
            "human_kp": human_kp_stacked.float(),
            "court_kp": court_kp_stacked.float(),
            "human_vis": human_vis_stacked.float(),
            "court_vis": court_vis_stacked.float(),
            "camera_params": camera_params_list,
            "num_views": torch.tensor(actual_num_views),
            "seq_len": torch.tensor(actual_seq_len),
            "view_mask": view_mask,
            "seq_mask": seq_mask,
            "position": position.float(),
            "rotation": rotation.float(),
        }


def collate_multiview_sequence(
    batch: list[PLCSMultiViewSequenceBatch],
) -> PLCSMultiViewSequenceBatchCollated:
    """Collate function for multi-view sequence batches.

    Handles variable number of views and sequence lengths by padding to
    max values in the batch. Provides view_mask and seq_mask to indicate
    valid (non-padded) positions.

    Args:
        batch: List of multi-view sequence samples.

    Returns:
        Collated batch with padded tensors and masks:
            - view_mask: (B, N_max) True for valid views
            - seq_mask: (B, T_max) True for valid frames

    """
    max_views = max(sample["num_views"].item() for sample in batch)
    max_seq_len = max(sample["seq_len"].item() for sample in batch)

    human_kp_batch = []
    court_kp_batch = []
    human_vis_batch = []
    court_vis_batch = []
    position_batch = []
    rotation_batch = []
    num_views_batch = []
    seq_len_batch = []
    view_mask_batch = []
    seq_mask_batch = []

    for sample in batch:
        n_views = sample["num_views"].item()
        s_len = sample["seq_len"].item()
        pad_views = max_views - n_views
        pad_seq = max_seq_len - s_len

        human_kp = sample["human_kp"]  # (N, T, 17, 2)
        court_kp = sample["court_kp"]  # (N, T, 20, 2)
        human_vis = sample["human_vis"]  # (N, T, 17)
        court_vis = sample["court_vis"]  # (N, T, 20)
        position = sample["position"]  # (T, 3)
        rotation = sample["rotation"]  # (T, 2)

        # Pad sequence dimension first (dim=1 for kp, dim=0 for targets)
        if pad_seq > 0:
            human_kp = torch.cat(
                [human_kp, torch.zeros(n_views, pad_seq, 17, 2)], dim=1
            )
            court_kp = torch.cat(
                [court_kp, torch.zeros(n_views, pad_seq, 20, 2)], dim=1
            )
            human_vis = torch.cat(
                [human_vis, torch.zeros(n_views, pad_seq, 17)], dim=1
            )
            court_vis = torch.cat(
                [court_vis, torch.zeros(n_views, pad_seq, 20)], dim=1
            )
            position = torch.cat([position, torch.zeros(pad_seq, 3)], dim=0)
            rotation = torch.cat([rotation, torch.zeros(pad_seq, 2)], dim=0)

        # Pad view dimension (dim=0)
        if pad_views > 0:
            human_kp = torch.cat(
                [human_kp, torch.zeros(pad_views, max_seq_len, 17, 2)], dim=0
            )
            court_kp = torch.cat(
                [court_kp, torch.zeros(pad_views, max_seq_len, 20, 2)], dim=0
            )
            human_vis = torch.cat(
                [human_vis, torch.zeros(pad_views, max_seq_len, 17)], dim=0
            )
            court_vis = torch.cat(
                [court_vis, torch.zeros(pad_views, max_seq_len, 20)], dim=0
            )

        # Create masks
        view_mask = torch.zeros(max_views, dtype=torch.bool)
        view_mask[:n_views] = True
        seq_mask = torch.zeros(max_seq_len, dtype=torch.bool)
        seq_mask[:s_len] = True

        human_kp_batch.append(human_kp)
        court_kp_batch.append(court_kp)
        human_vis_batch.append(human_vis)
        court_vis_batch.append(court_vis)
        position_batch.append(position)
        rotation_batch.append(rotation)
        num_views_batch.append(sample["num_views"])
        seq_len_batch.append(sample["seq_len"])
        view_mask_batch.append(view_mask)
        seq_mask_batch.append(seq_mask)

    return {
        "human_kp": torch.stack(human_kp_batch, dim=0),  # (B, N_max, T_max, 17, 2)
        "court_kp": torch.stack(court_kp_batch, dim=0),  # (B, N_max, T_max, 20, 2)
        "human_vis": torch.stack(human_vis_batch, dim=0),  # (B, N_max, T_max, 17)
        "court_vis": torch.stack(court_vis_batch, dim=0),  # (B, N_max, T_max, 20)
        "camera_params": [s["camera_params"] for s in batch],
        "num_views": torch.stack(num_views_batch, dim=0),  # (B,)
        "seq_len": torch.stack(seq_len_batch, dim=0),  # (B,)
        "view_mask": torch.stack(view_mask_batch, dim=0),  # (B, N_max)
        "seq_mask": torch.stack(seq_mask_batch, dim=0),  # (B, T_max)
        "position": torch.stack(position_batch, dim=0),  # (B, T_max, 3)
        "rotation": torch.stack(rotation_batch, dim=0),  # (B, T_max, 2)
    }
