"""Base dataset for NPZ scene loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import numpy as np
from torch.utils.data import Dataset

from src.common.data.camera_selection import select_camera
from src.common.data.npz_meta import decode_meta, get_num_frames
from src.common.data.scene_cache import SceneCache, get_scene_cache, load_npz_scene
from src.common.data.scene_paths import resolve_scene_files

SampleT = TypeVar("SampleT")


@dataclass(frozen=True)
class SceneDatasetConfig:
    """Configuration shared by NPZ scene datasets."""

    scene_dir: Path
    split: str | None = None
    split_file: Path | None = None
    min_seq_len: int = 0
    max_seq_len: int = 256
    cache_max_scenes: int = 128
    camera_mode: str | int = "random"
    crop_mode: Literal["random", "center"] = "random"


@dataclass(frozen=True)
class NPZScene:
    """Loaded NPZ scene data."""

    path: Path
    data: dict[str, Any]
    meta: dict[str, Any]
    num_frames: int
    num_cameras: int
    camera_idx: int


class NPZSceneDatasetBase(Dataset, Generic[SampleT]):
    """Base dataset for NPZ scene files."""

    def __init__(
        self,
        *,
        config: SceneDatasetConfig,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.rng = rng
        self.scene_dir = config.scene_dir
        self.scenes = resolve_scene_files(
            self.scene_dir, split=config.split, split_file=config.split_file
        )
        if not self.scenes:
            raise RuntimeError(f"No scenes found under {self.scene_dir}")
        self._scene_cache: SceneCache | None = (
            get_scene_cache(load_fn=load_npz_scene, maxsize=config.cache_max_scenes)
            if config.cache_max_scenes > 0
            else None
        )

    def __len__(self) -> int:
        return len(self.scenes)

    def _infer_fallback_len(self, data: dict[str, Any]) -> int:
        for key in ("ball_pos_world", "ball_pos_norm", "ball_uv"):
            if key in data:
                return int(data[key].shape[0])
        for value in data.values():
            if hasattr(value, "shape") and len(value.shape) >= 1:
                return int(value.shape[0])
        return 0

    def _load_scene(self, path: Path) -> NPZScene:
        data = (
            self._scene_cache.get(path)
            if self._scene_cache is not None
            else load_npz_scene(path)
        )
        meta = decode_meta(data.get("meta", {}))
        fallback_T = self._infer_fallback_len(data)
        num_frames = get_num_frames(meta, fallback_T)
        num_cameras = int(data.get("num_cameras", 0))
        camera_idx = (
            select_camera(self.config.camera_mode, num_cameras, self.rng)
            if num_cameras > 0
            else 0
        )
        return NPZScene(
            path=path,
            data=data,
            meta=meta,
            num_frames=num_frames,
            num_cameras=num_cameras,
            camera_idx=camera_idx,
        )

    def build_sample(self, scene: NPZScene) -> SampleT:
        """Build a sample dictionary from a loaded NPZScene."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> SampleT:
        scene = self._load_scene(self.scenes[idx])
        return self.build_sample(scene)
