"""Shared data utilities across tasks."""

from src.utils.data.index_cache import (
    CachedIndex,
    compute_config_hash,
    compute_scene_files_hash,
    get_index_cache_path,
    load_cached_index,
    save_cached_index,
)
from src.utils.data.scene_batch_sampler import (
    build_scene_sampler,
    ChunkedSceneBatchSampler,
    MixedSceneBatchSampler,
    resolve_scene_sampler_mode,
    SceneBatchSampler,
)
from src.utils.data.scene_cache import (
    SceneCache,
    SceneMeta,
    extract_scene_meta,
    extract_scene_meta_parallel,
    get_scene_cache,
    load_npz_scene,
    reset_scene_cache,
)
from src.utils.data.scene_id import resolve_scene_id
from src.utils.data.soft_labels import extract_event_indices, gaussian_soft_labels

__all__ = [
    "CachedIndex",
    "build_scene_sampler",
    "ChunkedSceneBatchSampler",
    "MixedSceneBatchSampler",
    "resolve_scene_sampler_mode",
    "SceneBatchSampler",
    "SceneCache",
    "SceneMeta",
    "compute_config_hash",
    "compute_scene_files_hash",
    "extract_scene_meta",
    "extract_scene_meta_parallel",
    "get_index_cache_path",
    "get_scene_cache",
    "load_cached_index",
    "load_npz_scene",
    "reset_scene_cache",
    "resolve_scene_id",
    "save_cached_index",
    "extract_event_indices",
    "gaussian_soft_labels",
]
