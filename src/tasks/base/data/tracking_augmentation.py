"""Shared augmentation primitives for unordered tracking observations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor


def permute_court_keypoint_sets(
    court_kp: Tensor,
    court_vis: Tensor,
    config: Mapping[str, Any] | None,
) -> tuple[Tensor, Tensor]:
    """Randomize 14-point court input order independently for each view.

    The same permutation is retained for every frame of a view. Coordinates
    and visibility are always permuted together, so the court remains an
    unordered geometric set rather than a mislabeled tensor.
    """
    cfg = config or {}
    if not bool(cfg.get("enabled", False)):
        return court_kp, court_vis
    if court_kp.ndim != 4 or court_vis.shape != court_kp.shape[:-1]:
        raise ValueError(
            "court tracking tensors must have shapes (V,T,K,2)/(V,T,K)."
        )
    probability = float(cfg.get("prob", 1.0))
    if not 0.0 <= probability <= 1.0:
        raise ValueError("court_keypoint_permutation.prob must be in [0, 1].")
    output_kp = court_kp.clone()
    output_vis = court_vis.clone()
    for view_index in range(court_kp.shape[0]):
        if torch.rand((), device=court_kp.device).item() >= probability:
            continue
        permutation = torch.randperm(court_kp.shape[2], device=court_kp.device)
        output_kp[view_index] = court_kp[view_index, :, permutation]
        output_vis[view_index] = court_vis[view_index, :, permutation]
    return output_kp, output_vis


__all__ = ["permute_court_keypoint_sets"]
