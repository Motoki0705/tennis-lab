"""Input-mode specific adapters for ball detection models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

_RGB_TO_LUMINANCE = (0.299, 0.587, 0.114)


def resolve_input_mode(model_cfg: Mapping[str, Any] | Any) -> str:
    """Resolve and validate the configured model input mode."""
    input_mode = str(model_cfg.get("input_mode", "rgb")).strip().lower()
    if input_mode not in {"rgb", "mdd"}:
        raise ValueError(
            "model.input_mode must be one of ['rgb', 'mdd'], "
            f"got '{input_mode}'."
        )
    return input_mode


def resolve_model_in_channels(model_cfg: Mapping[str, Any] | Any) -> int:
    """Return the effective model input channel count for the configured mode."""
    input_mode = resolve_input_mode(model_cfg)
    configured_channels = model_cfg.get("in_channels")
    if configured_channels is None:
        return 3 if input_mode == "rgb" else 2

    in_channels = int(configured_channels)
    expected_channels = 3 if input_mode == "rgb" else 2
    if in_channels != expected_channels:
        raise ValueError(
            f"model.in_channels must be {expected_channels} when "
            f"model.input_mode='{input_mode}', got {in_channels}."
        )
    return in_channels


def to_model_input(images: Tensor, model_cfg: Mapping[str, Any] | Any) -> Tensor:
    """Convert ``(B, T, C, H, W)`` RGB frames into the configured model input."""
    if images.ndim != 5:
        raise ValueError(
            "Expected images with shape (B, T, C, H, W), "
            f"got {tuple(images.shape)}."
        )

    input_mode = resolve_input_mode(model_cfg)
    if input_mode == "rgb":
        return images.permute(0, 2, 1, 3, 4).contiguous()
    return _rgb_frames_to_mdd(images, model_cfg)


def _rgb_frames_to_mdd(images: Tensor, model_cfg: Mapping[str, Any] | Any) -> Tensor:
    """Convert RGB frame sequences into ``(B, 2, T, H, W)`` MDD features."""
    if images.shape[2] != 3:
        raise ValueError(
            "MDD conversion expects RGB images with shape (B, T, 3, H, W), "
            f"got {tuple(images.shape)}."
        )

    weights = images.new_tensor(_RGB_TO_LUMINANCE).view(1, 1, 3, 1, 1)
    luminance = (images * weights).sum(dim=2)
    brighten = torch.zeros_like(luminance)
    darken = torch.zeros_like(luminance)

    if images.shape[1] > 1:
        frame_diffs = luminance[:, 1:] - luminance[:, :-1]
        gain, offset = _resolve_mdd_normalization(model_cfg)
        brighten[:, 1:] = _power_normalize(torch.clamp(frame_diffs, min=0.0), gain, offset)
        darken[:, 1:] = _power_normalize(torch.clamp(-frame_diffs, min=0.0), gain, offset)

    return torch.stack([brighten, darken], dim=1)


def _resolve_mdd_normalization(model_cfg: Mapping[str, Any] | Any) -> tuple[float, float]:
    """Resolve the experiments-compatible MDD normalization constants."""
    a_raw = float(model_cfg.get("mdd_a", 0.2))
    b_raw = float(model_cfg.get("mdd_b", 0.15))
    a_val = abs(float(torch.tanh(torch.tensor(a_raw)).item()))
    b_val = float(torch.tanh(torch.tensor(b_raw)).item())
    gain = 5.0 / (0.45 * a_val + 1.0e-6)
    offset = 0.6 * b_val
    return gain, offset


def _power_normalize(values: Tensor, gain: float, offset: float) -> Tensor:
    """Apply the same power normalization used by the experiments MDD path."""
    logits = torch.clamp(gain * (values.abs() - offset), min=-80.0, max=80.0)
    return torch.sigmoid(logits)


__all__ = [
    "resolve_input_mode",
    "resolve_model_in_channels",
    "to_model_input",
]
