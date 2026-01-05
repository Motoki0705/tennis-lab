"""Index caching utilities for PLCS datasets.

This module provides utilities for caching the index built by datasets,
avoiding the need to re-scan all scene files on every initialization.
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
    """Cached index data structure."""

    index: list[tuple[int, ...]]  # Variable-length tuples per dataset type
    scene_metas: list[dict[str, Any]]  # Serializable metadata
    config_hash: str
    scene_files_hash: str


def compute_config_hash(config: dict[str, Any], relevant_keys: list[str]) -> str:
    """Compute hash of relevant config values.

    Args:
        config: Configuration dictionary.
        relevant_keys: List of keys to include in the hash.

    Returns:
        12-character hex hash string.

    """
    relevant = {k: config.get(k) for k in relevant_keys if k in config}
    config_str = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]


def compute_scene_files_hash(scene_files: list[Path]) -> str:
    """Compute hash of scene file list (names + mtimes).

    This allows detecting when scene files have changed.

    Args:
        scene_files: List of scene file paths.

    Returns:
        12-character hex hash string.

    """
    entries = []
    for f in scene_files:
        try:
            mtime = f.stat().st_mtime
            entries.append(f"{f.name}:{mtime}")
        except OSError:
            entries.append(f.name)
    return hashlib.md5("\n".join(entries).encode()).hexdigest()[:12]


def get_index_cache_path(
    scene_dir: Path, dataset_class: str, config_hash: str
) -> Path:
    """Get path for index cache file.

    Args:
        scene_dir: Directory containing scene files.
        dataset_class: Name of the dataset class (e.g., "SceneDataset").
        config_hash: Hash of relevant config options.

    Returns:
        Path to the cache file.

    """
    cache_dir = scene_dir / ".cache"
    return cache_dir / f"index_{dataset_class}_{config_hash}.pkl"


def load_cached_index(cache_path: Path, expected_scene_hash: str) -> CachedIndex | None:
    """Load cached index if valid.

    Args:
        cache_path: Path to the cache file.
        expected_scene_hash: Expected hash of scene files.

    Returns:
        CachedIndex if valid, None otherwise.

    """
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)

        if not isinstance(cached, CachedIndex):
            return None

        if cached.scene_files_hash != expected_scene_hash:
            return None  # Scene files changed

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
    """Save index to cache file.

    Args:
        cache_path: Path to save the cache file.
        index: The index to cache.
        scene_metas: Scene metadata to cache.
        config_hash: Hash of relevant config options.
        scene_files_hash: Hash of scene files.

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
