"""Shared data utilities across tasks."""

from src.utils.data.augmentation import (
    add_gaussian_noise,
    add_temporally_correlated_jitter,
    apply_burst_visibility_dropout,
    apply_edge_aware_degradation,
    apply_speed_conditioned_localization_error,
    augment_keypoints,
    denormalize_tensor_images_imagenet,
    inject_false_positive_observations,
    normalize_frames_imagenet,
    normalize_tensor_images_imagenet,
    parse_float_range,
    parse_int_range,
    random_visibility_dropout,
    scale_uv_with_visibility,
)
from src.utils.data.heatmaps import (
    generate_gaussian_heatmap,
    generate_gaussian_heatmaps,
    heatmaps_to_argmax,
    heatmaps_to_peaks,
    heatmaps_to_pixel_coords,
    heatmaps_to_soft_argmax,
)
from src.utils.data.scene_io import load_scene_payload

__all__ = [
    "add_gaussian_noise",
    "add_temporally_correlated_jitter",
    "apply_burst_visibility_dropout",
    "apply_edge_aware_degradation",
    "apply_speed_conditioned_localization_error",
    "augment_keypoints",
    "denormalize_tensor_images_imagenet",
    "generate_gaussian_heatmap",
    "generate_gaussian_heatmaps",
    "heatmaps_to_argmax",
    "heatmaps_to_peaks",
    "heatmaps_to_pixel_coords",
    "heatmaps_to_soft_argmax",
    "inject_false_positive_observations",
    "load_scene_payload",
    "normalize_frames_imagenet",
    "normalize_tensor_images_imagenet",
    "parse_float_range",
    "parse_int_range",
    "random_visibility_dropout",
    "scale_uv_with_visibility",
]
