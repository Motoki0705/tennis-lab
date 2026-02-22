"""Scene loading utilities with LRU caching and metadata extraction."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.common.data.npz_meta import decode_meta

@dataclass(frozen=True)
class SceneMeta:
    """Lightweight metadata for scene indexing.

    Args:
        scene_idx: Index of the scene in the dataset.
        scene_path: Path to the scene file.
        num_frames: Number of frames in the scene.
        num_cameras: Number of cameras in the scene.
    """

    scene_idx: int
    scene_path: Path
    num_frames: int
    num_cameras: int


def load_npz_scene(scene_path: Path) -> dict[str, Any]:
    """Load a full NPZ scene into memory.

    Args:
        scene_path: Path to the scene file.

    Returns:
        Dictionary of arrays and metadata.
    """

    payload: dict[str, Any] = {}
    with np.load(scene_path, allow_pickle=True) as data:
        for key in data.files:
            if key == "meta":
                payload[key] = decode_meta(data[key])
            else:
                payload[key] = data[key].copy()
    return payload


def extract_scene_meta(scene_path: Path, scene_idx: int) -> SceneMeta:
    """Extract metadata from a scene file without full loading.

    Args:
        scene_path: Path to the scene NPZ file.
        scene_idx: Index of this scene in the dataset.

    Returns:
        SceneMeta with num_frames and num_cameras.
    """

    with np.load(scene_path, allow_pickle=True, mmap_mode="r") as data:
        meta = {}
        if "meta" in data:
            meta = decode_meta(data["meta"])
        num_cameras = int(data["num_cameras"]) if "num_cameras" in data else 0
    num_frames = int(meta.get("num_frames", 0)) if isinstance(meta, dict) else 0
    return SceneMeta(
        scene_idx=scene_idx,
        scene_path=scene_path,
        num_frames=num_frames,
        num_cameras=num_cameras,
    )


def extract_scene_meta_parallel(
    scene_files: list[Path],
    max_workers: int = 8,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[SceneMeta]:
    """Extract scene metadata in parallel.

    Args:
        scene_files: List of scene NPZ paths.
        max_workers: Number of threads for I/O.
        progress_callback: Optional callback with (completed, total).

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
            except Exception as exc:  # pragma: no cover - best-effort
                print(f"Warning: Failed to load scene {scene_files[idx]}: {exc}")
                failed.append(idx)

            if progress_callback:
                progress_callback(i + 1, total)

    _ = failed
    return [results[i] for i in range(total) if i in results]


class SceneCache:
    """LRU-cached scene loader.

    Args:
        load_fn: Function that loads a scene file into memory.
        maxsize: Maximum number of cached scenes.
    """

    def __init__(self, load_fn: Callable[[Path], Any], maxsize: int = 128) -> None:
        self.load_fn = load_fn
        self.maxsize = int(maxsize)
        self._load_cached = lru_cache(maxsize=self.maxsize)(self._load_impl)

    def _load_impl(self, scene_path: str) -> Any:
        """Load scene from disk (lru_cache target)."""

        return self.load_fn(Path(scene_path))

    def get(self, scene_path: Path) -> Any:
        """Get scene data from cache or disk.

        Args:
            scene_path: Path to the scene file.

        Returns:
            Loaded scene data.
        """

        return self._load_cached(str(scene_path))

    def clear(self) -> None:
        """Clear the cache."""

        self._load_cached.cache_clear()

    @property
    def cache_info(self) -> Any:
        """Return cache statistics."""

        return self._load_cached.cache_info()


_global_caches: dict[tuple[int, int], SceneCache] = {}


def get_scene_cache(
    load_fn: Callable[[Path], Any],
    maxsize: int = 128,
) -> SceneCache:
    """Get or create a global SceneCache for a load function.

    Args:
        load_fn: Function that loads a scene file.
        maxsize: Cache size for the loader.

    Returns:
        SceneCache instance.
    """

    key = (id(load_fn), int(maxsize))
    if key not in _global_caches:
        _global_caches[key] = SceneCache(load_fn=load_fn, maxsize=maxsize)
    return _global_caches[key]


def reset_scene_cache() -> None:
    """Clear all global scene caches."""

    for cache in _global_caches.values():
        cache.clear()
    _global_caches.clear()


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        scene_path = base / "scene_000.npz"
        np.savez(
            scene_path,
            meta=json.dumps({"num_frames": 3}),
            num_cameras=np.array(2),
            ball_uv=np.zeros((3, 2), dtype=np.float32),
        )

        cache = SceneCache(load_fn=load_npz_scene, maxsize=2)
        scene = cache.get(scene_path)
        assert scene["ball_uv"].shape == (3, 2)
        meta = scene["meta"]
        assert isinstance(meta, dict) and meta.get("num_frames") == 3

        metas = extract_scene_meta_parallel([scene_path], max_workers=1)
        assert metas[0].num_frames == 3
        print("common.data.scene_cache smoke ok")
