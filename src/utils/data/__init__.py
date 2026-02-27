"""Shared data utilities across tasks."""

from src.utils.data.augmentation import (
    add_gaussian_noise,
    augment_keypoints,
    random_visibility_dropout,
    scale_uv_with_visibility,
)
from src.utils.data.collate import collate_padded_batch
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
from src.utils.data.soft_labels import extract_event_indices, gaussian_soft_labels

__all__ = [
    "add_gaussian_noise",
    "augment_keypoints",
    "build_scene_sampler",
    "ChunkedSceneBatchSampler",
    "collate_padded_batch",
    "MixedSceneBatchSampler",
    "random_visibility_dropout",
    "resolve_scene_sampler_mode",
    "scale_uv_with_visibility",
    "SceneBatchSampler",
    "SceneCache",
    "SceneMeta",
    "extract_scene_meta",
    "extract_scene_meta_parallel",
    "get_scene_cache",
    "load_npz_scene",
    "reset_scene_cache",
    "extract_event_indices",
    "gaussian_soft_labels",
]
