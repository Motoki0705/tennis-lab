"""Schema validation utilities for e2e tests.

Provides functions to validate data structures against TypedDict/dataclass
schemas defined in src/{task}/data/types.py.
"""

from __future__ import annotations

from typing import Any

import torch

from tests.e2e.validation.tensor_validators import (
    collect_errors,
    validate_dict_has_keys,
    validate_normalized_uv,
    validate_tensor_dtype,
    validate_tensor_range,
    validate_tensor_shape,
    validate_visibility_mask,
)

# =============================================================================
# BLCS Schema Validators
# =============================================================================


def validate_blcs_sample(sample: dict[str, Any]) -> list[str]:
    """Validate a sample conforms to BLCSSample schema.

    Expected schema (from src/blcs/data/types.py):
        ball_uv: (T, 2) ball 2D trajectory in normalized UV
        ball_mask: (T,) ball visibility mask
        court_kp: (20, 2) court 2D keypoints in normalized UV
        court_vis: (20,) court keypoint visibility flags
        position_3d: (T, 3) ground truth 3D trajectory (normalized)
        velocity_3d: (T, 3) 3D velocity vectors
        seq_len: scalar, actual sequence length

    Args:
        sample: Dictionary to validate.

    Returns:
        List of validation error messages (empty if valid).

    """
    errors = []

    # Check required keys
    required_keys = [
        "ball_uv",
        "ball_mask",
        "court_kp",
        "court_vis",
        "position_3d",
        "velocity_3d",
        "seq_len",
    ]
    errors.extend(validate_dict_has_keys(sample, required_keys, "BLCSSample"))

    if errors:
        return errors  # Can't validate further without keys

    # Get sequence length for shape validation
    seq_len = sample["seq_len"]
    if isinstance(seq_len, torch.Tensor):
        T = int(seq_len.item())
    else:
        T = int(seq_len)

    # Validate shapes (use None for dynamic T dimension)
    errors.extend(
        collect_errors(
            validate_tensor_shape(sample["ball_uv"], (None, 2), "ball_uv"),
            validate_tensor_shape(sample["ball_mask"], (None,), "ball_mask"),
            validate_tensor_shape(sample["court_kp"], (20, 2), "court_kp"),
            validate_tensor_shape(sample["court_vis"], (20,), "court_vis"),
            validate_tensor_shape(sample["position_3d"], (None, 3), "position_3d"),
            validate_tensor_shape(sample["velocity_3d"], (None, 3), "velocity_3d"),
        )
    )

    # Validate dtypes
    errors.extend(
        collect_errors(
            validate_tensor_dtype(
                sample["ball_uv"], [torch.float32, torch.float64], "ball_uv"
            ),
            validate_tensor_dtype(
                sample["court_kp"], [torch.float32, torch.float64], "court_kp"
            ),
            validate_tensor_dtype(
                sample["position_3d"], [torch.float32, torch.float64], "position_3d"
            ),
        )
    )

    # Validate UV coordinates are normalized
    errors.extend(
        collect_errors(
            validate_normalized_uv(sample["ball_uv"], "ball_uv"),
            validate_normalized_uv(sample["court_kp"], "court_kp"),
        )
    )

    # Validate visibility masks
    errors.extend(
        collect_errors(
            validate_visibility_mask(sample["ball_mask"], "ball_mask"),
            validate_visibility_mask(sample["court_vis"], "court_vis"),
        )
    )

    return errors


def validate_blcs_batch(batch: dict[str, Any], batch_size: int | None = None) -> list[str]:
    """Validate a batch conforms to BLCSBatch schema.

    Expected schema (from src/blcs/data/types.py):
        ball_uv: (B, T_max, 2) padded ball trajectories
        ball_mask: (B, T_max) padded visibility masks
        court_kp: (B, 20, 2) court keypoints
        court_vis: (B, 20) court keypoint visibility
        position_3d: (B, T_max, 3) padded ground truth trajectories
        velocity_3d: (B, T_max, 3) padded velocities
        seq_len: (B,) actual sequence lengths

    Args:
        batch: Dictionary to validate.
        batch_size: Expected batch size (optional, inferred if None).

    Returns:
        List of validation error messages (empty if valid).

    """
    errors = []

    required_keys = [
        "ball_uv",
        "ball_mask",
        "court_kp",
        "court_vis",
        "position_3d",
        "velocity_3d",
        "seq_len",
    ]
    errors.extend(validate_dict_has_keys(batch, required_keys, "BLCSBatch"))

    if errors:
        return errors

    # Infer batch size
    B = batch_size or batch["ball_uv"].shape[0]

    # Validate shapes
    errors.extend(
        collect_errors(
            validate_tensor_shape(batch["ball_uv"], (B, None, 2), "ball_uv"),
            validate_tensor_shape(batch["ball_mask"], (B, None), "ball_mask"),
            validate_tensor_shape(batch["court_kp"], (B, 20, 2), "court_kp"),
            validate_tensor_shape(batch["court_vis"], (B, 20), "court_vis"),
            validate_tensor_shape(batch["position_3d"], (B, None, 3), "position_3d"),
            validate_tensor_shape(batch["velocity_3d"], (B, None, 3), "velocity_3d"),
            validate_tensor_shape(batch["seq_len"], (B,), "seq_len"),
        )
    )

    return errors


def validate_blcs_scene_meta(meta: dict[str, Any]) -> list[str]:
    """Validate scene metadata conforms to BLCSSceneMeta schema.

    Args:
        meta: Metadata dictionary to validate.

    Returns:
        List of validation error messages (empty if valid).

    """
    errors = []

    required_keys = [
        "scene_id",
        "from_cell",
        "from_side",
        "category",
        "to_cell",
        "t_net",
        "t_fence",
        "t_bounce1",
        "t_bounce2",
        "fps_out",
        "sim_fps",
        "num_frames",
        "num_cameras_sampled",
        "num_cameras",
    ]
    errors.extend(validate_dict_has_keys(meta, required_keys, "BLCSSceneMeta"))

    if errors:
        return errors

    # Validate types
    if not isinstance(meta["scene_id"], str):
        errors.append(f"scene_id: expected str, got {type(meta['scene_id']).__name__}")

    if meta["from_side"] not in ("near", "far"):
        errors.append(f"from_side: expected 'near' or 'far', got '{meta['from_side']}'")

    # Validate integer fields
    int_fields = [
        "from_cell",
        "to_cell",
        "t_net",
        "t_fence",
        "t_bounce1",
        "t_bounce2",
        "fps_out",
        "sim_fps",
        "num_frames",
        "num_cameras_sampled",
        "num_cameras",
    ]
    for field in int_fields:
        if not isinstance(meta[field], int):
            errors.append(f"{field}: expected int, got {type(meta[field]).__name__}")

    # Validate ranges
    if meta["from_cell"] < 0 or meta["from_cell"] > 11:
        errors.append(f"from_cell: expected 0-11, got {meta['from_cell']}")

    if meta["num_frames"] <= 0:
        errors.append(f"num_frames: expected > 0, got {meta['num_frames']}")

    if meta["fps_out"] <= 0:
        errors.append(f"fps_out: expected > 0, got {meta['fps_out']}")

    return errors


def validate_blcs_camera_params(params: dict[str, Any]) -> list[str]:
    """Validate camera parameters conform to BLCSCameraParams schema.

    Args:
        params: Camera parameters dictionary to validate.

    Returns:
        List of validation error messages (empty if valid).

    """
    errors = []

    required_keys = ["center", "R", "f", "cx", "cy", "w", "h"]
    errors.extend(validate_dict_has_keys(params, required_keys, "BLCSCameraParams"))

    if errors:
        return errors

    # Validate center (list of 3 floats)
    if not isinstance(params["center"], list) or len(params["center"]) != 3:
        errors.append("center: expected list of 3 floats")

    # Validate R (3x3 matrix)
    R = params["R"]
    if not isinstance(R, list) or len(R) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in R):
        errors.append("R: expected 3x3 matrix")

    # Validate scalar fields
    for field in ["f", "cx", "cy"]:
        if not isinstance(params[field], (int, float)):
            errors.append(f"{field}: expected number, got {type(params[field]).__name__}")

    for field in ["w", "h"]:
        if not isinstance(params[field], int):
            errors.append(f"{field}: expected int, got {type(params[field]).__name__}")
        elif params[field] <= 0:
            errors.append(f"{field}: expected > 0, got {params[field]}")

    return errors


# =============================================================================
# PLCS Schema Validators
# =============================================================================


def validate_plcs_frame_batch(batch: dict[str, Any]) -> list[str]:
    """Validate a batch conforms to PLCSFrameBatch schema.

    Expected schema (from src/plcs/data/types.py):
        human_kp: (34,) flattened human keypoints, normalized UV
        court_kp: (40,) flattened court keypoints, normalized UV
        human_vis: (17,) visibility flags for human keypoints
        court_vis: (20,) visibility flags for court keypoints
        position: (3,) normalized court position
        rotation: (2,) player orientation [sin(yaw), cos(yaw)]

    Args:
        batch: Dictionary to validate.

    Returns:
        List of validation error messages (empty if valid).

    """
    errors = []

    required_keys = [
        "human_kp",
        "court_kp",
        "human_vis",
        "court_vis",
        "position",
        "rotation",
    ]
    errors.extend(validate_dict_has_keys(batch, required_keys, "PLCSFrameBatch"))

    if errors:
        return errors

    # Validate shapes
    errors.extend(
        collect_errors(
            validate_tensor_shape(batch["human_kp"], (34,), "human_kp"),
            validate_tensor_shape(batch["court_kp"], (40,), "court_kp"),
            validate_tensor_shape(batch["human_vis"], (17,), "human_vis"),
            validate_tensor_shape(batch["court_vis"], (20,), "court_vis"),
            validate_tensor_shape(batch["position"], (3,), "position"),
            validate_tensor_shape(batch["rotation"], (2,), "rotation"),
        )
    )

    # Validate rotation is sin/cos (range [-1, 1])
    errors.extend(
        collect_errors(
            validate_tensor_range(batch["rotation"], -1.0, 1.0, "rotation"),
        )
    )

    # Validate visibility masks
    errors.extend(
        collect_errors(
            validate_visibility_mask(batch["human_vis"], "human_vis"),
            validate_visibility_mask(batch["court_vis"], "court_vis"),
        )
    )

    return errors


def validate_plcs_sequence_batch(batch: dict[str, Any]) -> list[str]:
    """Validate a batch conforms to PLCSSequenceBatch schema.

    Expected schema (from src/plcs/data/types.py):
        human_kp: (T, 17, 2) human keypoints over time
        court_kp: (1, 20, 2) aggregated court keypoints
        human_vis: (T, 17) visibility flags for human keypoints
        court_vis: (1, 20) aggregated visibility flags for court
        position: (T, 3) normalized court positions over time
        rotation: (T, 2) player orientations over time

    Args:
        batch: Dictionary to validate.

    Returns:
        List of validation error messages (empty if valid).

    """
    errors = []

    required_keys = [
        "human_kp",
        "court_kp",
        "human_vis",
        "court_vis",
        "position",
        "rotation",
    ]
    errors.extend(validate_dict_has_keys(batch, required_keys, "PLCSSequenceBatch"))

    if errors:
        return errors

    # Validate shapes (None for dynamic T dimension)
    errors.extend(
        collect_errors(
            validate_tensor_shape(batch["human_kp"], (None, 17, 2), "human_kp"),
            validate_tensor_shape(batch["court_kp"], (1, 20, 2), "court_kp"),
            validate_tensor_shape(batch["human_vis"], (None, 17), "human_vis"),
            validate_tensor_shape(batch["court_vis"], (1, 20), "court_vis"),
            validate_tensor_shape(batch["position"], (None, 3), "position"),
            validate_tensor_shape(batch["rotation"], (None, 2), "rotation"),
        )
    )

    # Validate rotation range
    errors.extend(
        collect_errors(
            validate_tensor_range(batch["rotation"], -1.0, 1.0, "rotation"),
        )
    )

    return errors


def validate_plcs_scene_meta(meta: dict[str, Any]) -> list[str]:
    """Validate scene metadata conforms to PLCSSceneMeta schema.

    Args:
        meta: Metadata dictionary to validate.

    Returns:
        List of validation error messages (empty if valid).

    """
    errors = []

    required_keys = [
        "scene_id",
        "motion_source",
        "motion_category",
        "gender",
        "fps",
        "num_frames",
        "initial_position",
        "initial_yaw",
        "num_cameras_sampled",
        "num_cameras",
    ]
    errors.extend(validate_dict_has_keys(meta, required_keys, "PLCSSceneMeta"))

    if errors:
        return errors

    # Validate string fields
    str_fields = ["scene_id", "motion_source", "motion_category", "gender"]
    for field in str_fields:
        if not isinstance(meta[field], str):
            errors.append(f"{field}: expected str, got {type(meta[field]).__name__}")

    # Validate gender
    if meta["gender"] not in ("male", "female", "neutral"):
        errors.append(f"gender: expected 'male'/'female'/'neutral', got '{meta['gender']}'")

    # Validate integer fields
    int_fields = ["fps", "num_frames", "num_cameras_sampled", "num_cameras"]
    for field in int_fields:
        if not isinstance(meta[field], int):
            errors.append(f"{field}: expected int, got {type(meta[field]).__name__}")

    # Validate initial_position (list of floats)
    if not isinstance(meta["initial_position"], list):
        errors.append("initial_position: expected list")

    # Validate initial_yaw (float)
    if not isinstance(meta["initial_yaw"], (int, float)):
        errors.append(
            f"initial_yaw: expected number, got {type(meta['initial_yaw']).__name__}"
        )

    return errors


def validate_plcs_camera_params(params: dict[str, Any]) -> list[str]:
    """Validate camera parameters conform to PLCSCameraParams schema.

    Args:
        params: Camera parameters dictionary to validate.

    Returns:
        List of validation error messages (empty if valid).

    """
    # Same structure as BLCS
    return validate_blcs_camera_params(params)
