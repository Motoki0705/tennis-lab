"""Shared data utilities across tasks."""

from src.utils.data.augmentation import (
    add_gaussian_noise,
    add_temporally_correlated_jitter,
    apply_burst_visibility_dropout,
    apply_edge_aware_degradation,
    apply_speed_conditioned_localization_error,
    augment_keypoints,
    inject_false_positive_observations,
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
from src.utils.data.soft_labels import gaussian_soft_labels

__all__ = [
    "add_gaussian_noise",
    "add_temporally_correlated_jitter",
    "apply_burst_visibility_dropout",
    "apply_edge_aware_degradation",
    "apply_speed_conditioned_localization_error",
    "augment_keypoints",
    "collate_padded_batch",
    "extract_event_frames",
    "extract_event_indices",
    "generate_gaussian_heatmap",
    "generate_gaussian_heatmaps",
    "gaussian_soft_labels",
    "heatmaps_to_argmax",
    "inject_false_positive_observations",
    "random_visibility_dropout",
    "scale_uv_with_visibility",
]
