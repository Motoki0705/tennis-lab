"""Validation utilities for e2e tests.

This package provides schema and tensor validation utilities for verifying
that generated datasets and model batches conform to the type definitions
in src/{task}/data/types.py.
"""

from tests.e2e.validation.schema_validators import (
    validate_blcs_batch,
    validate_blcs_camera_params,
    validate_blcs_sample,
    validate_blcs_scene_meta,
    validate_plcs_camera_params,
    validate_plcs_frame_batch,
    validate_plcs_scene_meta,
    validate_plcs_sequence_batch,
)
from tests.e2e.validation.tensor_validators import (
    validate_normalized_uv,
    validate_tensor_dtype,
    validate_tensor_range,
    validate_tensor_shape,
    validate_visibility_mask,
)

__all__ = [
    # Schema validators
    "validate_blcs_sample",
    "validate_blcs_batch",
    "validate_blcs_scene_meta",
    "validate_blcs_camera_params",
    "validate_plcs_frame_batch",
    "validate_plcs_sequence_batch",
    "validate_plcs_scene_meta",
    "validate_plcs_camera_params",
    # Tensor validators
    "validate_tensor_shape",
    "validate_tensor_dtype",
    "validate_tensor_range",
    "validate_normalized_uv",
    "validate_visibility_mask",
]
