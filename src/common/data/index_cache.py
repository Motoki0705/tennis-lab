"""Index caching utilities shared across tasks.

This module caches dataset indices to avoid re-scanning scene files on every
initialization.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CachedIndex:
    """Cached index data structure.

    Args:
        index: Index entries for samples.
        scene_metas: Serializable metadata for scenes.
        config_hash: Hash of config-relevant values.
        scene_files_hash: Hash of scene file list.
    """

    index: list[tuple[int, ...]]
    scene_metas: list[dict[str, Any]]
    config_hash: str
    scene_files_hash: str


def compute_config_hash(config: dict[str, Any], relevant_keys: list[str]) -> str:
    """Compute a hash for relevant config keys.

    Args:
        config: Configuration dictionary.
        relevant_keys: Keys to include in the hash.

    Returns:
        Short hex hash string.
    """

    relevant = {k: config.get(k) for k in relevant_keys if k in config}
    config_str = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]


def compute_scene_files_hash(scene_files: list[Path]) -> str:
    """Compute a hash of scene file names + mtimes.

    Args:
        scene_files: Scene file paths.

    Returns:
        Short hex hash string.
    """

    entries = []
    for path in scene_files:
        try:
            mtime = path.stat().st_mtime
            entries.append(f"{path.name}:{mtime}")
        except OSError:
            entries.append(path.name)
    return hashlib.md5("\n".join(entries).encode()).hexdigest()[:12]


def get_index_cache_path(scene_dir: Path, dataset_class: str, config_hash: str) -> Path:
    """Return the cache path for a dataset index.

    Args:
        scene_dir: Directory containing scene files.
        dataset_class: Dataset class name.
        config_hash: Hash for config keys.

    Returns:
        Path to cache file.
    """

    cache_dir = scene_dir / ".cache"
    return cache_dir / f"index_{dataset_class}_{config_hash}.pkl"


def load_cached_index(cache_path: Path, expected_scene_hash: str) -> CachedIndex | None:
    """Load cached index if present and valid.

    Args:
        cache_path: Path to cached index file.
        expected_scene_hash: Expected hash for scene files.

    Returns:
        CachedIndex if valid; otherwise None.
    """

    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)

        if not isinstance(cached, CachedIndex):
            return None

        if cached.scene_files_hash != expected_scene_hash:
            return None

        return cached
    except Exception:
        return None


def save_cached_index(
    cache_path: Path,
    index: list[tuple[int, ...]],
    scene_metas: list[dict[str, Any]],
    config_hash: str,
    scene_files_hash: str,
) -> None:
    """Save index data to cache.

    Args:
        cache_path: Path to cache file.
        index: Sample index entries.
        scene_metas: Scene metadata list.
        config_hash: Hash for config values.
        scene_files_hash: Hash for scene files.
    """

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cached = CachedIndex(
        index=index,
        scene_metas=scene_metas,
        config_hash=config_hash,
        scene_files_hash=scene_files_hash,
    )

    with open(cache_path, "wb") as f:
        pickle.dump(cached, f)


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        scene_dir = base / "scenes"
        scene_dir.mkdir(parents=True, exist_ok=True)
        dummy_file = scene_dir / "scene_000.npz"
        dummy_file.write_bytes(b"dummy")

        config_hash = compute_config_hash({"camera_mode": "random"}, ["camera_mode"])
        scene_hash = compute_scene_files_hash([dummy_file])
        cache_path = get_index_cache_path(scene_dir, "Dummy", config_hash)

        save_cached_index(
            cache_path,
            index=[(0, 0, 0)],
            scene_metas=[{"path": str(dummy_file), "num_frames": 1, "num_cameras": 1}],
            config_hash=config_hash,
            scene_files_hash=scene_hash,
        )
        loaded = load_cached_index(cache_path, scene_hash)
        assert loaded is not None
        assert loaded.index == [(0, 0, 0)]
        print("common.data.index_cache smoke ok")
