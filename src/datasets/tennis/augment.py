"""Augmentation utilities for the Tennis pose datasets."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def sample_camera_indices(
    num_available: int,
    max_cameras: int,
    min_cameras: int | None = None,
) -> Tensor:
    """Sample a subset of camera indices for a single sample.

    Args:
        num_available: Number of cameras available in the source arrays.
        max_cameras: Maximum number of cameras to materialize in the output.
        min_cameras: Optional minimum number of cameras to select. If ``None``,
            defaults to ``max_cameras``.

    Returns:
        1D LongTensor of shape ``[K]`` with selected camera indices in the
        range ``[0, num_available)``.

    """
    if num_available <= 0:
        raise ValueError("num_available must be positive")
    if max_cameras <= 0:
        raise ValueError("max_cameras must be positive")
    max_sel = min(max_cameras, num_available)
    if min_cameras is None:
        min_sel = max_sel
    else:
        min_sel = int(min_cameras)
    if min_sel <= 0:
        raise ValueError("min_cameras must be positive")
    if min_sel > max_sel:
        msg = f"min_cameras={min_sel} exceeds available/max_cameras={max_sel}"
        raise ValueError(msg)
    if max_sel == min_sel:
        k = max_sel
    else:
        # Use torch randint so that DataLoader worker seeding applies.
        k = int(torch.randint(min_sel, max_sel + 1, (1,)).item())
    # randperm on CPU is fine; DataLoader seeds each worker deterministically.
    perm = torch.randperm(num_available)
    return perm[:k]


def apply_random_2d_affine(
    sample: dict[str, Tensor],
    *,
    enabled: bool,
    split: str,
    max_rotation_deg: float = 10.0,
    max_scale: float = 0.1,
    max_translate: float = 0.1,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """Apply per-camera random affine transforms to 2D coordinates.

    This function operates in-place on ``sample["keypoints_2d"]`` and
    ``sample["court_2d"]`` if augmentation is enabled and ``split == "train"``.

    Args:
        sample: A single-sample dict as returned by ``TennisSceneWindowDataset``.
        enabled: Whether 2D augmentation is enabled.
        split: Dataset split name (e.g. ``"train"``, ``"val"``, ``"test"``).
        max_rotation_deg: Maximum absolute rotation angle in degrees.
        max_scale: Maximum relative isotropic scaling (e.g. 0.1 → ±10%).
        max_translate: Maximum translation in normalized coordinates.
        generator: Optional torch.Generator for deterministic sampling.

    Returns:
        The same dict, potentially modified in-place.

    """
    if not enabled or split != "train":
        return sample

    keypoints_2d = sample.get("keypoints_2d")
    court_2d = sample.get("court_2d")
    if keypoints_2d is None or court_2d is None:
        return sample
    if keypoints_2d.ndim != 5:
        return sample

    T, V, M, J, _ = keypoints_2d.shape
    if V == 0:
        return sample

    device = keypoints_2d.device
    rng_kwargs: dict[str, Any] = {}
    if generator is not None:
        rng_kwargs["generator"] = generator

    # Helper to sample uniform values on the same device as keypoints.
    def _rand_uniform(low: float, high: float, shape: tuple[int, ...]) -> Tensor:
        return torch.empty(shape, device=device).uniform_(low, high, **rng_kwargs)

    # Sample per-camera parameters.
    theta = _rand_uniform(
        -max_rotation_deg, max_rotation_deg, (V,)
    ) * (torch.pi / 180.0)
    scale = 1.0 + _rand_uniform(-max_scale, max_scale, (V,))
    tx = _rand_uniform(-max_translate, max_translate, (V,))
    ty = _rand_uniform(-max_translate, max_translate, (V,))

    cos_t = torch.cos(theta) * scale
    sin_t = torch.sin(theta) * scale

    # Build affine matrices: [V, 2, 3]
    # [[a, b, tx],
    #  [c, d, ty]]
    a = cos_t
    b = -sin_t
    c = sin_t
    d = cos_t
    A = torch.zeros((V, 2, 3), device=device, dtype=keypoints_2d.dtype)
    A[:, 0, 0] = a
    A[:, 0, 1] = b
    A[:, 0, 2] = tx
    A[:, 1, 0] = c
    A[:, 1, 1] = d
    A[:, 1, 2] = ty

    # Apply per-camera affine to keypoints_2d and court_2d.
    # keypoints_2d: [T, V, M, J, 2]
    pts = keypoints_2d.view(T, V, M * J, 2)
    ones = torch.ones((T, V, M * J, 1), device=device, dtype=pts.dtype)
    homo = torch.cat([pts, ones], dim=-1)  # [T, V, M*J, 3]
    # Move camera axis to batch for matmul: [V, T*M*J, 3]
    homo_v = homo.permute(1, 0, 2, 3).reshape(V, T * M * J, 3)
    out_v = torch.bmm(homo_v, A.transpose(1, 2))  # [V, T*M*J, 2]
    out = out_v.view(V, T, M, J, 2).permute(1, 0, 2, 3, 4)
    sample["keypoints_2d"] = out

    # court_2d: [V, 20, 2]
    if court_2d.ndim == 3 and court_2d.shape[0] == V:
        C = court_2d.shape[1]
        c_pts = court_2d.view(V, C, 2)
        c_ones = torch.ones((V, C, 1), device=device, dtype=c_pts.dtype)
        c_homo = torch.cat([c_pts, c_ones], dim=-1)  # [V, C, 3]
        c_out = torch.bmm(c_homo, A.transpose(1, 2))  # [V, C, 2]
        sample["court_2d"] = c_out

    return sample


