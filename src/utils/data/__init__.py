"""Shared data utilities across tasks."""

from src.utils.data.augmentation import (
    add_gaussian_noise,
    augment_keypoints,
    random_visibility_dropout,
    scale_uv_with_visibility,
)
from src.utils.data.collate import collate_padded_batch
from src.utils.data.event_utils import (
    extract_event_frames,
    extract_event_indices,
)
from src.utils.data.heatmaps import (
    generate_gaussian_heatmap,
    generate_gaussian_heatmaps,
    heatmaps_to_argmax,
)
from src.utils.data.scene_batch_sampler import (
    build_scene_sampler,
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
from src.utils.data.soft_labels import gaussian_soft_labels

__all__ = [
    "add_gaussian_noise",
    "augment_keypoints",
    "build_scene_sampler",
    "collate_padded_batch",
    "extract_event_frames",
    "extract_event_indices",
    "generate_gaussian_heatmap",
    "generate_gaussian_heatmaps",
    "gaussian_soft_labels",
    "heatmaps_to_argmax",
    "random_visibility_dropout",
    "scale_uv_with_visibility",
    "SceneBatchSampler",
    "SceneCache",
    "SceneMeta",
    "extract_scene_meta",
    "extract_scene_meta_parallel",
    "get_scene_cache",
    "load_npz_scene",
    "reset_scene_cache",
]
