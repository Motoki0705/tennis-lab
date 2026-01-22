"""Sequence dataset for PLCS.

Provides fixed-length frame sequences for training sequential PLCS models.
"""

from __future__ import annotations

import random as rng
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from src.base.data.augmentation import augment_keypoints
from src.common.data.index_cache import (
    compute_config_hash,
    compute_scene_files_hash,
    get_index_cache_path,
    load_cached_index,
    save_cached_index,
)
from src.common.data.scene_cache import (
    extract_scene_meta_parallel,
    get_scene_cache,
)
from src.plcs.generate_dataset.io.scene_loader import load_scene
from src.plcs.data.types import PLCSSequenceBatch

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
        cache_maxsize: int = 128,  # LRU cache size for scenes
        parallel_workers: int = 8,  # Threads for metadata extraction
    ) -> None:
        super().__init__()

        self.scene_dir = Path(scene_dir)
        self.config = config or {}
        self.augment = augment
        self.camera_mode = camera_mode
        self.parallel_workers = parallel_workers

        data_cfg = self.config.get("data", {})
        self.seq_len: int = int(data_cfg.get("seq_len", 16))
        self.seq_stride: int = int(data_cfg.get("seq_stride", self.seq_len))
        self.kp_noise_std: float = float(data_cfg.get("keypoint_noise_std", 0.01))
        self.visibility_drop_prob: float = float(
            data_cfg.get("visibility_drop_prob", 0.05)
        )

        # Get shared scene cache (lazy loading with LRU)
        self._scene_cache = get_scene_cache(load_fn=load_scene, maxsize=cache_maxsize)

        # Scene paths (no longer loading all scenes into memory)
        self.scene_paths: list[Path] = []

        # Index all scene files
        scenes_subdir = self.scene_dir / "scenes"
        self.scene_files = sorted(scenes_subdir.glob("scene_*.npz"))
        if not self.scene_files:
            raise ValueError(f"No scene files found in {scenes_subdir}")

        print(f"SceneSequenceDataset: found {len(self.scene_files)} scene files")

        # Build index: (scene_idx, cam_idx, start_frame)
        self._build_index()

    def _build_index(self) -> None:
        """Build sample index with parallel metadata extraction.

        Index entries are (scene_idx, cam_idx, start_frame).
        """
        # Check for cached index
        config_hash = compute_config_hash(
            {
                "camera_mode": self.camera_mode,
                "seq_len": self.seq_len,
                "seq_stride": self.seq_stride,
            },
            ["camera_mode", "seq_len", "seq_stride"],
        )
        scene_hash = compute_scene_files_hash(self.scene_files)
        cache_path = get_index_cache_path(
            self.scene_dir, "SceneSequenceDataset", config_hash
        )

        cached = load_cached_index(cache_path, scene_hash)
        if cached is not None:
            self.index = cached.index
            self.scene_paths = [Path(m["path"]) for m in cached.scene_metas]
            print(
                f"SceneSequenceDataset: loaded cached index "
                f"({len(self.index)} sequences)"
            )
            return

        # Extract metadata in parallel (no full scene loading!)
        print(
            f"SceneSequenceDataset: extracting metadata from "
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
            if meta.num_frames < self.seq_len:
                # Skip scenes shorter than the desired sequence length
                continue

            self.scene_paths.append(meta.scene_path)
            actual_idx = len(self.scene_paths) - 1

            max_start = meta.num_frames - self.seq_len

            if self.camera_mode == "all":
                # All cameras, sliding window over frames
                for start in range(0, max_start + 1, self.seq_stride):
                    for cam_idx in range(meta.num_cameras):
                        self.index.append((actual_idx, cam_idx, start))
            elif self.camera_mode == "random":
                # Camera is selected randomly at __getitem__ time
                for start in range(0, max_start + 1, self.seq_stride):
                    self.index.append((actual_idx, -1, start))  # -1 = random
            else:
                # Specific camera index
                cam_idx = int(self.camera_mode)
                if cam_idx < meta.num_cameras:
                    for start in range(0, max_start + 1, self.seq_stride):
                        self.index.append((actual_idx, cam_idx, start))

        # Save to cache
        scene_metas_serializable = [
            {
                "path": str(m.scene_path),
                "num_frames": m.num_frames,
                "num_cameras": m.num_cameras,
            }
            for m in scene_metas
            if m.num_frames >= self.seq_len
        ]
        save_cached_index(
            cache_path, self.index, scene_metas_serializable, config_hash, scene_hash
        )

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
            - court_kp: (T, 20, 2) - full sequence (not aggregated)
            - human_vis: (T, 17)
            - court_vis: (T, 20)
            - position: (T, 3)
            - rotation: (T, 2)
        """
        scene_idx, cam_idx, start = self.index[idx]
        # Load scene on-demand via LRU cache
        scene = self._scene_cache.get(self.scene_paths[scene_idx])

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

        # Apply visibility mask (zero-out invisible keypoints)
        human_kp_masked = human_kp * human_vis.unsqueeze(-1)
        court_kp_masked = court_kp * court_vis.unsqueeze(-1)

        return {
            "human_kp": human_kp_masked.float(),
            "court_kp": court_kp_masked.float(),
            "human_vis": human_vis.float(),
            "court_vis": court_vis.float(),
            "position": position.float(),
            "rotation": rotation.float(),
        }


if __name__ == "__main__":
    import json
    from tempfile import TemporaryDirectory

    import numpy as np

    with TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        scenes_dir = base / "scenes"
        scenes_dir.mkdir(parents=True, exist_ok=True)
        scene_path = scenes_dir / "scene_000.npz"
        np.savez(
            scene_path,
            meta=json.dumps({"num_frames": 2}),
            num_cameras=np.array(1),
            position=np.zeros((2, 3), dtype=np.float32),
            rotation=np.zeros((2, 2), dtype=np.float32),
            canonical_pose_3d=np.zeros((17, 3), dtype=np.float32),
            cam_0_params=json.dumps({"fx": 1.0}),
            cam_0_human_kp_uv=np.zeros((2, 17, 2), dtype=np.float32),
            cam_0_human_kp_visible=np.ones((2, 17), dtype=np.bool_),
            cam_0_human_visibility_ratio=np.array(1.0),
            cam_0_court_kp_uv=np.zeros((2, 20, 2), dtype=np.float32),
            cam_0_court_kp_visible=np.ones((2, 20), dtype=np.bool_),
            cam_0_court_visibility_count=np.array(20.0),
        )
        cfg = {"data": {"seq_len": 1, "seq_stride": 1}}
        ds = SceneSequenceDataset(scene_dir=base, config=cfg, augment=False)
        sample = ds[0]
        assert sample["human_kp"].shape[-2:] == (17, 2)
        assert sample["position"].shape[-1] == 3
        print("plcs.data.sequence_dataset smoke ok")
