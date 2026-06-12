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
from src.utils.data.heatmaps import (
    generate_gaussian_heatmap,
    generate_gaussian_heatmaps,
    heatmaps_to_argmax,
    heatmaps_to_peaks,
    heatmaps_to_soft_argmax,
)

__all__ = [
    "add_gaussian_noise",
    "add_temporally_correlated_jitter",
    "apply_burst_visibility_dropout",
    "apply_edge_aware_degradation",
    "apply_speed_conditioned_localization_error",
    "augment_keypoints",
    "generate_gaussian_heatmap",
    "generate_gaussian_heatmaps",
    "heatmaps_to_argmax",
    "heatmaps_to_peaks",
    "heatmaps_to_soft_argmax",
    "inject_false_positive_observations",
    "random_visibility_dropout",
    "scale_uv_with_visibility",
]
