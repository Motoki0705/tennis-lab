"""Tensor validation utilities for e2e tests.

Provides functions to validate tensor shapes, dtypes, and value ranges.
"""

from __future__ import annotations

from typing import Any

import torch


def validate_tensor_shape(
    tensor: torch.Tensor,
    expected_shape: tuple[int | None, ...],
    name: str,
) -> str | None:
    """Validate tensor shape against expected pattern.

    Args:
        tensor: Tensor to validate.
        expected_shape: Expected shape tuple. Use None for dynamic dimensions.
        name: Field name for error messages.

    Returns:
        Error message if validation fails, None otherwise.

    Example:
        >>> validate_tensor_shape(t, (None, 2), "ball_uv")  # (T, 2)
        >>> validate_tensor_shape(t, (32, None, 3), "positions")  # (B, T, 3)

    """
    if not isinstance(tensor, torch.Tensor):
        return f"{name}: expected torch.Tensor, got {type(tensor).__name__}"

    actual_shape = tensor.shape
    if len(actual_shape) != len(expected_shape):
        return (
            f"{name}: expected {len(expected_shape)} dims, "
            f"got {len(actual_shape)} dims (shape={tuple(actual_shape)})"
        )

    for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape, strict=True)):
        if expected is not None and actual != expected:
            return (
                f"{name}: dim {i} expected {expected}, "
                f"got {actual} (shape={tuple(actual_shape)})"
            )

    return None


def validate_tensor_dtype(
    tensor: torch.Tensor,
    expected_dtype: torch.dtype | list[torch.dtype],
    name: str,
) -> str | None:
    """Validate tensor dtype.

    Args:
        tensor: Tensor to validate.
        expected_dtype: Expected dtype or list of acceptable dtypes.
        name: Field name for error messages.

    Returns:
        Error message if validation fails, None otherwise.

    """
    if not isinstance(tensor, torch.Tensor):
        return f"{name}: expected torch.Tensor, got {type(tensor).__name__}"

    if isinstance(expected_dtype, list):
        if tensor.dtype not in expected_dtype:
            return (
                f"{name}: expected dtype in {expected_dtype}, got {tensor.dtype}"
            )
    elif tensor.dtype != expected_dtype:
        return f"{name}: expected dtype {expected_dtype}, got {tensor.dtype}"

    return None


def validate_tensor_range(
    tensor: torch.Tensor,
    min_val: float | None,
    max_val: float | None,
    name: str,
    *,
    allow_nan: bool = False,
) -> str | None:
    """Validate tensor values are within expected range.

    Args:
        tensor: Tensor to validate.
        min_val: Minimum allowed value (inclusive). None to skip.
        max_val: Maximum allowed value (inclusive). None to skip.
        name: Field name for error messages.
        allow_nan: Whether to allow NaN values.

    Returns:
        Error message if validation fails, None otherwise.

    """
    if not isinstance(tensor, torch.Tensor):
        return f"{name}: expected torch.Tensor, got {type(tensor).__name__}"

    if tensor.numel() == 0:
        return None  # Empty tensor is valid

    if not allow_nan and torch.isnan(tensor).any():
        return f"{name}: contains NaN values"

    if torch.isinf(tensor).any():
        return f"{name}: contains Inf values"

    # Get non-NaN values for range check
    valid_values = tensor[~torch.isnan(tensor)] if allow_nan else tensor

    if valid_values.numel() > 0:
        if min_val is not None and valid_values.min().item() < min_val:
            return (
                f"{name}: min value {valid_values.min().item():.4f} "
                f"is below {min_val}"
            )
        if max_val is not None and valid_values.max().item() > max_val:
            return (
                f"{name}: max value {valid_values.max().item():.4f} "
                f"is above {max_val}"
            )

    return None


def validate_normalized_uv(
    tensor: torch.Tensor,
    name: str,
    *,
    strict: bool = False,
) -> str | None:
    """Validate tensor contains normalized UV coordinates in [0, 1].

    Args:
        tensor: Tensor to validate, expected shape (..., 2).
        name: Field name for error messages.
        strict: If True, values must be strictly in [0, 1].
                If False, allow small margin for numerical errors.

    Returns:
        Error message if validation fails, None otherwise.

    """
    if not isinstance(tensor, torch.Tensor):
        return f"{name}: expected torch.Tensor, got {type(tensor).__name__}"

    if tensor.shape[-1] != 2:
        return f"{name}: expected last dim to be 2 (UV), got {tensor.shape[-1]}"

    margin = 0.0 if strict else 0.01
    min_val = 0.0 - margin
    max_val = 1.0 + margin

    return validate_tensor_range(tensor, min_val, max_val, name)


def validate_visibility_mask(
    tensor: torch.Tensor,
    name: str,
) -> str | None:
    """Validate visibility mask contains only 0/1 or boolean values.

    Args:
        tensor: Tensor to validate.
        name: Field name for error messages.

    Returns:
        Error message if validation fails, None otherwise.

    """
    if not isinstance(tensor, torch.Tensor):
        return f"{name}: expected torch.Tensor, got {type(tensor).__name__}"

    if tensor.dtype == torch.bool:
        return None  # Boolean tensor is valid

    # For numeric tensors, check values are 0 or 1
    unique_vals = tensor.unique()
    valid_vals = torch.tensor([0, 1], dtype=tensor.dtype, device=tensor.device)

    for val in unique_vals:
        if val not in valid_vals:
            return (
                f"{name}: visibility mask should contain only 0/1, "
                f"found {unique_vals.tolist()}"
            )

    return None


def collect_errors(*errors: str | None) -> list[str]:
    """Collect non-None errors into a list.

    Args:
        *errors: Variable number of error messages (or None).

    Returns:
        List of non-None error messages.

    """
    return [e for e in errors if e is not None]


def validate_dict_has_keys(
    data: dict[str, Any],
    required_keys: list[str],
    name: str,
) -> list[str]:
    """Validate dictionary contains all required keys.

    Args:
        data: Dictionary to validate.
        required_keys: List of required key names.
        name: Context name for error messages.

    Returns:
        List of error messages for missing keys.

    """
    errors = []
    for key in required_keys:
        if key not in data:
            errors.append(f"{name}: missing required key '{key}'")
    return errors
