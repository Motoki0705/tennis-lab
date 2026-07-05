"""Common data augmentation utilities for datasets.

This module provides shared augmentation functions used across PLCS, BLCS,
and other modules to avoid code duplication.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
# float32 array views with identical values, for numpy-based normalization paths.
_IMAGENET_MEAN_ARR = np.asarray(IMAGENET_MEAN, dtype=np.float32)
_IMAGENET_STD_ARR = np.asarray(IMAGENET_STD, dtype=np.float32)


def _as_dict(value: Any) -> dict[str, Any]:
    """Convert plain dicts or DictConfig-like objects into a shallow dict."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "items"):
        return dict(value.items())
    return {}


def _enabled(config: Mapping[str, Any], *, default: bool = False) -> bool:
    return bool(config.get("enabled", default))


def _prob(config: Mapping[str, Any], *, default: float = 1.0) -> float:
    return float(config.get("prob", default))


def _should_apply(prob: float, reference: Tensor) -> bool:
    if prob <= 0:
        return False
    if prob >= 1:
        return True
    return bool(torch.rand((), device=reference.device).item() < prob)


def parse_float_range(value: Any, name: str) -> tuple[float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise ValueError(f"{name} must be a two-element list/tuple.")
    out = (float(value[0]), float(value[1]))
    if out[0] > out[1]:
        raise ValueError(f"{name} min must be <= max, got {out}.")
    return out


def parse_int_range(value: Any, name: str) -> tuple[int, int]:
    """Parse a non-negative two-element integer range ``(min, max)`` from config."""
    if not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{name} must be a sequence with two elements.")
    low = int(value[0])
    high = int(value[1])
    if low < 0 or high < 0:
        raise ValueError(f"{name} must be non-negative.")
    if low > high:
        raise ValueError(f"{name} must satisfy min <= max.")
    return low, high


def normalize_tensor_images_imagenet(
    images: Tensor,
    *,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> Tensor:
    """Apply ImageNet normalization to ``(..., 3, H, W)`` image tensors."""
    if images.ndim < 3 or images.shape[-3] != 3:
        raise ValueError(
            "Expected images with shape (..., 3, H, W) for ImageNet normalization, "
            f"got {tuple(images.shape)}."
        )
    view_shape = [1] * images.ndim
    view_shape[-3] = 3
    mean_tensor = images.new_tensor(mean).view(*view_shape)
    std_tensor = images.new_tensor(std).view(*view_shape)
    return (images - mean_tensor) / std_tensor


def denormalize_tensor_images_imagenet(
    images: Tensor,
    *,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> Tensor:
    """Invert ImageNet normalization for ``(..., 3, H, W)`` image tensors."""
    if images.ndim < 3 or images.shape[-3] != 3:
        raise ValueError(
            "Expected images with shape (..., 3, H, W) for ImageNet denormalization, "
            f"got {tuple(images.shape)}."
        )
    view_shape = [1] * images.ndim
    view_shape[-3] = 3
    mean_tensor = images.new_tensor(mean).view(*view_shape)
    std_tensor = images.new_tensor(std).view(*view_shape)
    return images * std_tensor + mean_tensor


def tensor_images_to_uint8_rgb(images: Tensor) -> np.ndarray:
    """Convert ``(..., 3, H, W)`` float images in ``[0, 1]`` to uint8 ``(..., H, W, 3)``.

    Clamps to ``[0, 1]``, moves channels last, scales by 255 and casts —
    the tail shared by visualization renderers after ImageNet denormalization.
    """
    if images.ndim < 3 or images.shape[-3] != 3:
        raise ValueError(
            "Expected images with shape (..., 3, H, W) for uint8 RGB conversion, "
            f"got {tuple(images.shape)}."
        )
    array = images.detach().cpu().float().clamp(0.0, 1.0).movedim(-3, -1).numpy()
    return cast("np.ndarray", (array * 255.0).astype(np.uint8))


def normalize_frames_imagenet(
    frames: list[np.ndarray],
    *,
    mean: np.ndarray = _IMAGENET_MEAN_ARR,
    std: np.ndarray = _IMAGENET_STD_ARR,
) -> list[np.ndarray]:
    """Apply ImageNet normalization to a list of HWC float32 numpy frames."""
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    return [((frame - mean_arr) / std_arr).astype(np.float32) for frame in frames]


def _rand_like_shape(
    reference: Tensor,
    shape: Sequence[int],
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Return uniform random values on the same device as ``reference``."""
    return torch.rand(
        tuple(int(dim) for dim in shape),
        device=reference.device,
        generator=generator,
    )


def _randn_like_tensor(
    reference: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Return normal random values preserving tensor dtype/device."""
    return torch.randn(
        reference.shape,
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    )


def _apply_visibility_mask(delta: Tensor, visibility: Tensor | None) -> Tensor:
    """Zero coordinate deltas where the corresponding observation is hidden."""
    if visibility is None:
        return delta
    return delta * (visibility > 0).to(dtype=delta.dtype).unsqueeze(-1)


def _update_visibility_from_mask(visibility: Tensor, keep_mask: Tensor) -> Tensor:
    """Apply a boolean keep mask to bool or float visibility tensors."""
    if visibility.dtype == torch.bool:
        return visibility & keep_mask
    return visibility * keep_mask.to(dtype=visibility.dtype)


def _validate_uv_tensor(uv: Tensor) -> None:
    if uv.shape[-1] != 2:
        raise ValueError(f"uv must have last dimension 2, got shape {tuple(uv.shape)}.")


def _validate_temporal_uv_tensor(uv: Tensor) -> None:
    _validate_uv_tensor(uv)
    if uv.ndim < 2:
        raise ValueError(f"uv must include a temporal dimension, got {tuple(uv.shape)}.")


def scale_uv_with_visibility(
    uv: Tensor,
    visibility: Tensor,
    scale: float,
    center: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """Scale normalized UV coordinates and update visibility by bounds.

    Args:
        uv: UV tensor of shape (..., 2) in normalized coordinates [0, 1].
        visibility: Visibility tensor matching uv prefix shape (...,).
        scale: Isotropic scaling factor.
        center: Scaling center in normalized UV space.

    Returns:
        Tuple of (scaled_uv, updated_visibility).

    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}.")
    _validate_uv_tensor(uv)

    uv_scaled = (uv - center) * scale + center
    in_bounds = (
        (uv_scaled[..., 0] >= 0.0)
        & (uv_scaled[..., 0] <= 1.0)
        & (uv_scaled[..., 1] >= 0.0)
        & (uv_scaled[..., 1] <= 1.0)
    )

    if visibility.dtype == torch.bool:
        visibility_scaled = visibility & in_bounds
    else:
        visibility_scaled = visibility * in_bounds.to(visibility.dtype)

    return uv_scaled.clamp(0.0, 1.0), visibility_scaled


def add_gaussian_noise(
    tensor: Tensor,
    noise_std: float,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Add Gaussian noise to a tensor.

    Args:
        tensor: Input tensor of any shape.
        noise_std: Standard deviation of Gaussian noise.

    Returns:
        Tensor with added noise (same shape as input).

    """
    if noise_std <= 0:
        return tensor

    noise = _randn_like_tensor(tensor, generator=generator) * noise_std
    return tensor + noise


def random_visibility_dropout(
    visibility: Tensor,
    drop_prob: float,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Randomly drop visibility flags for data augmentation.

    Args:
        visibility: Boolean or float visibility tensor of any shape.
        drop_prob: Probability of dropping each visibility flag.

    Returns:
        Updated visibility tensor with some flags set to False/0.

    """
    if drop_prob <= 0:
        return visibility

    drop_mask = _rand_like_shape(
        visibility,
        visibility.shape,
        generator=generator,
    ) < drop_prob
    return _update_visibility_from_mask(visibility, ~drop_mask)


def add_temporally_correlated_jitter(
    uv: Tensor,
    visibility: Tensor | None = None,
    *,
    jitter_std: float = 0.0,
    drift_std: float = 0.0,
    drift_decay: float = 0.9,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Add detector-like temporally correlated coordinate noise.

    The last two dimensions are interpreted as ``(time, xy)``.  The
    autoregressive drift term models low-frequency detector bias, while the
    independent jitter term keeps frame-level localization noise.
    """
    _validate_temporal_uv_tensor(uv)
    if jitter_std <= 0 and drift_std <= 0:
        return uv
    if not 0 <= drift_decay < 1:
        raise ValueError(f"drift_decay must be in [0, 1), got {drift_decay}.")

    delta = torch.zeros_like(uv)
    if drift_std > 0:
        innovations = _randn_like_tensor(uv, generator=generator) * drift_std
        running = torch.zeros_like(uv[..., 0, :])
        drift_steps: list[Tensor] = []
        for frame_idx in range(int(uv.shape[-2])):
            running = running * drift_decay + innovations[..., frame_idx, :]
            drift_steps.append(running)
        delta = delta + torch.stack(drift_steps, dim=-2)
    if jitter_std > 0:
        delta = delta + _randn_like_tensor(uv, generator=generator) * jitter_std

    delta = _apply_visibility_mask(delta, visibility)
    return (uv + delta).clamp(0.0, 1.0)


def apply_burst_visibility_dropout(
    visibility: Tensor,
    *,
    prob: float,
    min_len: int,
    max_len: int,
    max_bursts: int = 1,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Drop contiguous temporal spans from visibility observations.

    The final dimension is interpreted as time.  Leading dimensions are treated
    as independent tracks, which maps naturally to ``(camera, time)`` BLCS
    visibility tensors.
    """
    if prob <= 0 or max_bursts <= 0:
        return visibility
    if visibility.ndim < 1:
        raise ValueError("visibility must have at least one dimension.")
    if min_len <= 0 or max_len <= 0:
        raise ValueError("burst lengths must be positive.")
    if min_len > max_len:
        raise ValueError(f"min_len must be <= max_len, got {min_len}>{max_len}.")

    out = visibility.clone()
    time_len = int(out.shape[-1])
    if time_len == 0:
        return out

    flat = out.reshape(-1, time_len)
    for track in range(flat.shape[0]):
        if _rand_like_shape(out, (), generator=generator).item() >= prob:
            continue
        num_bursts = int(
            torch.randint(
                1,
                max_bursts + 1,
                (),
                device=out.device,
                generator=generator,
            ).item()
        )
        for _ in range(num_bursts):
            burst_len = int(
                torch.randint(
                    min_len,
                    max_len + 1,
                    (),
                    device=out.device,
                    generator=generator,
                ).item()
            )
            burst_len = min(burst_len, time_len)
            max_start = time_len - burst_len
            start = int(
                torch.randint(
                    0,
                    max_start + 1,
                    (),
                    device=out.device,
                    generator=generator,
                ).item()
            )
            flat[track, start : start + burst_len] = False if out.dtype == torch.bool else 0.0
    return out


def dilate_temporal_mask(mask: Tensor, radius: int) -> Tensor:
    """Dilate a boolean mask along the final temporal dimension."""
    if radius <= 0:
        return mask.bool()
    if mask.ndim < 1:
        raise ValueError("mask must have at least one dimension.")

    out = mask.bool().clone()
    time_len = int(out.shape[-1])
    for offset in range(1, radius + 1):
        if offset >= time_len:
            break
        out[..., offset:] |= mask[..., :-offset].bool()
        out[..., :-offset] |= mask[..., offset:].bool()
    return out


def inject_false_positive_observations(
    uv: Tensor,
    visibility: Tensor,
    *,
    false_positive_prob: float = 0.0,
    after_dropout_mask: Tensor | None = None,
    after_dropout_prob: float = 0.0,
    after_dropout_window: int = 0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Inject fake detector observations into currently invisible frames."""
    _validate_temporal_uv_tensor(uv)
    if visibility.shape != uv.shape[:-1]:
        raise ValueError(
            "visibility shape must match uv without coordinate dimension: "
            f"got visibility={tuple(visibility.shape)}, uv={tuple(uv.shape)}."
        )
    if false_positive_prob <= 0 and after_dropout_prob <= 0:
        return uv, visibility

    absent = visibility <= 0
    fp_mask = torch.zeros_like(absent, dtype=torch.bool)
    if false_positive_prob > 0:
        fp_mask |= (
            _rand_like_shape(visibility, visibility.shape, generator=generator)
            < false_positive_prob
        ) & absent

    if after_dropout_mask is not None and after_dropout_prob > 0:
        near_dropout = dilate_temporal_mask(after_dropout_mask, after_dropout_window)
        fp_mask |= (
            _rand_like_shape(visibility, visibility.shape, generator=generator)
            < after_dropout_prob
        ) & absent & near_dropout

    if not fp_mask.any():
        return uv, visibility

    out_uv = uv.clone()
    out_visibility = visibility.clone()
    out_uv[fp_mask] = _rand_like_shape(
        uv,
        (int(fp_mask.sum().item()), 2),
        generator=generator,
    ).to(dtype=uv.dtype)
    if visibility.dtype == torch.bool:
        out_visibility[fp_mask] = True
    else:
        out_visibility[fp_mask] = 1.0
    return out_uv, out_visibility


def apply_edge_aware_degradation(
    uv: Tensor,
    visibility: Tensor,
    *,
    edge_margin: float,
    noise_std: float = 0.0,
    drop_prob: float = 0.0,
    clip_out_prob: float = 0.0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Increase detector noise and misses near normalized image boundaries."""
    _validate_temporal_uv_tensor(uv)
    if edge_margin <= 0:
        return uv, visibility
    if visibility.shape != uv.shape[:-1]:
        raise ValueError(
            "visibility shape must match uv without coordinate dimension: "
            f"got visibility={tuple(visibility.shape)}, uv={tuple(uv.shape)}."
        )

    edge_distance = torch.minimum(
        torch.minimum(uv[..., 0], 1.0 - uv[..., 0]),
        torch.minimum(uv[..., 1], 1.0 - uv[..., 1]),
    )
    edge_weight = ((edge_margin - edge_distance) / edge_margin).clamp(0.0, 1.0)
    visible = visibility > 0

    out_uv = uv.clone()
    out_visibility = visibility.clone()
    if noise_std > 0:
        noise = _randn_like_tensor(uv, generator=generator) * noise_std
        noise = noise * edge_weight.unsqueeze(-1).to(dtype=uv.dtype)
        out_uv = out_uv + _apply_visibility_mask(noise, visibility)

    keep_mask = torch.ones_like(visible, dtype=torch.bool)
    if drop_prob > 0:
        keep_mask &= ~(
            _rand_like_shape(visibility, visibility.shape, generator=generator)
            < (drop_prob * edge_weight)
        )
    if clip_out_prob > 0:
        keep_mask &= ~(
            _rand_like_shape(visibility, visibility.shape, generator=generator)
            < (clip_out_prob * edge_weight)
        )

    in_bounds = (
        (out_uv[..., 0] >= 0.0)
        & (out_uv[..., 0] <= 1.0)
        & (out_uv[..., 1] >= 0.0)
        & (out_uv[..., 1] <= 1.0)
    )
    out_visibility = _update_visibility_from_mask(
        out_visibility,
        keep_mask & in_bounds,
    )
    return out_uv.clamp(0.0, 1.0), out_visibility


def apply_speed_conditioned_localization_error(
    uv: Tensor,
    visibility: Tensor,
    *,
    prob: float,
    speed_threshold: float,
    lag_overshoot_range: Sequence[float],
    noise_std: float = 0.0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Add lag/overshoot localization errors on fast-moving observations."""
    _validate_temporal_uv_tensor(uv)
    if prob <= 0:
        return uv, visibility
    if len(lag_overshoot_range) != 2:
        raise ValueError("lag_overshoot_range must contain [min, max].")
    coeff_min = float(lag_overshoot_range[0])
    coeff_max = float(lag_overshoot_range[1])
    if coeff_min > coeff_max:
        raise ValueError(
            f"lag_overshoot_range min must be <= max, got {lag_overshoot_range}."
        )

    velocity = torch.zeros_like(uv)
    velocity[..., 1:, :] = uv[..., 1:, :] - uv[..., :-1, :]
    speed = torch.linalg.norm(velocity, dim=-1)
    if speed_threshold > 0:
        speed_mask = speed >= speed_threshold
        speed_weight = ((speed - speed_threshold) / speed_threshold).clamp(0.0, 2.0)
    else:
        speed_mask = speed > 0
        speed_weight = torch.ones_like(speed)

    visible = visibility > 0
    apply_mask = (
        _rand_like_shape(visibility, visibility.shape, generator=generator) < prob
    ) & speed_mask & visible
    if not apply_mask.any():
        return uv, visibility

    coeff = (
        _rand_like_shape(uv, uv.shape[:-1], generator=generator)
        * (coeff_max - coeff_min)
        + coeff_min
    ).to(dtype=uv.dtype)
    delta = velocity * coeff.unsqueeze(-1)
    if noise_std > 0:
        delta = delta + (
            _randn_like_tensor(uv, generator=generator)
            * noise_std
            * speed_weight.unsqueeze(-1).to(dtype=uv.dtype)
        )
    delta = delta * apply_mask.unsqueeze(-1).to(dtype=uv.dtype)

    out_uv = uv + delta
    in_bounds = (
        (out_uv[..., 0] >= 0.0)
        & (out_uv[..., 0] <= 1.0)
        & (out_uv[..., 1] >= 0.0)
        & (out_uv[..., 1] <= 1.0)
    )
    out_visibility = _update_visibility_from_mask(visibility, in_bounds)
    return out_uv.clamp(0.0, 1.0), out_visibility


def augment_keypoints(
    keypoints: Tensor,
    visibility: Tensor,
    noise_std: float = 0.0,
    visibility_drop_prob: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Apply common keypoint augmentations: noise and visibility dropout.

    This is a convenience function that combines Gaussian noise and visibility
    dropout, commonly used together in PLCS and BLCS datasets.

    Args:
        keypoints: Keypoint coordinates, shape (..., N, 2) or (..., 2).
        visibility: Visibility flags, shape (..., N) or (...,).
        noise_std: Standard deviation of Gaussian noise to add.
        visibility_drop_prob: Probability of dropping visibility flags.

    Returns:
        Tuple of (augmented_keypoints, augmented_visibility).

    """
    # Add Gaussian noise
    augmented_kp = add_gaussian_noise(keypoints, noise_std)

    # Random visibility dropout
    augmented_vis = random_visibility_dropout(visibility, visibility_drop_prob)

    return augmented_kp, augmented_vis
