"""Scene loading with LRU cache and metadata extraction utilities.

This module provides optimized scene loading for PLCS datasets:
- Parallel metadata extraction for fast index building
- LRU-cached scene loading for memory efficiency
- Global singleton cache for sharing across datasets
"""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.plcs.generate_dataset.io.scene_loader import load_scene


@dataclass(frozen=True)
class SceneMeta:
    """Lightweight metadata for index building (no full data loaded)."""

    scene_idx: int
    scene_path: Path
    num_frames: int
    num_cameras: int


def extract_scene_meta(scene_path: Path, scene_idx: int) -> SceneMeta:
    """Extract only metadata from NPZ without loading full arrays.

    Uses np.load with mmap_mode='r' to avoid loading large arrays into memory.
    Only reads 'meta' (JSON) and 'num_cameras' (scalar).

    Args:
        scene_path: Path to the scene NPZ file.
        scene_idx: Index of this scene in the dataset.

    Returns:
        SceneMeta with num_frames and num_cameras.

    """
    with np.load(scene_path, allow_pickle=True, mmap_mode="r") as data:
        meta_raw = data["meta"].item()
        if isinstance(meta_raw, (bytes, bytearray)):
            meta_raw = meta_raw.decode("utf-8")
        meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
        num_cameras = int(data["num_cameras"])

    return SceneMeta(
        scene_idx=scene_idx,
        scene_path=scene_path,
        num_frames=meta["num_frames"],
        num_cameras=num_cameras,
    )


def extract_scene_meta_parallel(
    scene_files: list[Path],
    max_workers: int = 8,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[SceneMeta]:
    """Extract metadata from all scene files in parallel.

    Args:
        scene_files: List of scene NPZ paths.
        max_workers: Number of threads for parallel I/O.
        progress_callback: Optional callback(completed, total) for progress.

    Returns:
        List of SceneMeta in original order.

    """
    results: dict[int, SceneMeta] = {}
    failed: list[int] = []
    total = len(scene_files)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_scene_meta, path, idx): idx
            for idx, path in enumerate(scene_files)
        }

        for i, future in enumerate(as_completed(futures)):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                # Skip invalid scenes, log warning
                print(f"Warning: Failed to load scene {scene_files[idx]}: {e}")
                failed.append(idx)

            if progress_callback:
                progress_callback(i + 1, total)

    # Return in original order, skipping failed
    return [results[i] for i in range(total) if i in results]


class SceneCache:
    """LRU-cached scene loader.

    Provides lazy loading of full scene data with bounded memory usage.
    Uses functools.lru_cache under the hood.
    """

    def __init__(self, maxsize: int = 128) -> None:
        """Initialize cache.

        Args:
            maxsize: Maximum number of scenes to keep in memory.
                     Set based on available memory and scene size.

        """
        self.maxsize = maxsize
        # Create the cached function
        self._load_cached = lru_cache(maxsize=maxsize)(self._load_impl)

    @staticmethod
    def _load_impl(scene_path: str) -> dict[str, Any]:
        """Load scene from disk (static for lru_cache compatibility)."""
        return load_scene(scene_path)

    def get(self, scene_path: Path) -> dict[str, Any]:
        """Get scene, loading from cache or disk.

        Args:
            scene_path: Path to the scene NPZ file.

        Returns:
            Scene data as dict.

        """
        return self._load_cached(str(scene_path))

    def clear(self) -> None:
        """Clear the cache."""
        self._load_cached.cache_clear()

    @property
    def cache_info(self) -> Any:
        """Get cache statistics."""
        return self._load_cached.cache_info()


# Global singleton for sharing across datasets
_global_cache: SceneCache | None = None


def get_scene_cache(maxsize: int = 128) -> SceneCache:
    """Get or create the global scene cache.

    Args:
        maxsize: Maximum number of scenes to keep in cache.
                 Only used when creating a new cache.

    Returns:
        The global SceneCache instance.

    """
    global _global_cache
    if _global_cache is None:
        _global_cache = SceneCache(maxsize=maxsize)
    return _global_cache


def reset_scene_cache() -> None:
    """Reset the global scene cache (useful for testing)."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()
    _global_cache = None
