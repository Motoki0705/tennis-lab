"""Physical-width BLCS observations and post-association debug alignment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class PhysicalObservationCandidates:
    """Unpacked camera observations carried on the physical ``P`` axis.

    ``gt_index`` is debug provenance only. Association callers must pass only
    ``uv`` and ``vis`` to the shared observation tracker.
    """

    uv: Tensor
    vis: Tensor
    gt_index: Tensor


@dataclass(frozen=True, slots=True)
class AlignedObservationDebug:
    """Clean values and provenance aligned to a tracked exact-``Q`` axis."""

    clean_uv: Tensor
    clean_vis: Tensor
    gt_index: Tensor


def build_physical_observation_candidates(
    *,
    ball_uv: Tensor,
    ball_vis: Tensor,
    physical_presence: Tensor,
) -> PhysicalObservationCandidates:
    """Build clean ``(V,T,P)`` carriers without assigning query slots."""
    if ball_uv.ndim != 4 or ball_uv.shape[-1] != 2:
        raise ValueError("ball_uv must have shape (V,T,P,2).")
    if ball_vis.shape != ball_uv.shape[:-1]:
        raise ValueError("ball_vis must match ball_uv without the UV axis.")
    if physical_presence.shape != ball_uv.shape[1:3]:
        raise ValueError("physical_presence must match ball_uv's (T,P) axes.")
    if ball_vis.dtype != torch.bool or physical_presence.dtype != torch.bool:
        raise TypeError("ball_vis and physical_presence must have dtype bool.")
    if ball_uv.device != ball_vis.device or ball_uv.device != physical_presence.device:
        raise ValueError("physical observation tensors must share one device.")
    if ball_uv.shape[2] <= 0:
        raise ValueError("physical observations must contain at least one carrier.")
    if bool((ball_vis & ~physical_presence.unsqueeze(0)).any()):
        raise ValueError("ball_vis cannot be true for an absent physical object.")

    visible = ball_vis & physical_presence.unsqueeze(0)
    clean_uv = torch.where(visible.unsqueeze(-1), ball_uv, torch.zeros_like(ball_uv))
    physical_ids = torch.arange(
        ball_uv.shape[2], dtype=torch.long, device=ball_uv.device
    ).view(1, 1, -1)
    gt_index = torch.where(
        visible,
        physical_ids.expand_as(visible),
        torch.full_like(visible, -1, dtype=torch.long),
    )
    return PhysicalObservationCandidates(
        uv=clean_uv,
        vis=visible,
        gt_index=gt_index,
    )


def align_clean_observations_after_tracking(
    *,
    clean: PhysicalObservationCandidates,
    detection_indices: Tensor,
    candidate_gt_index: Tensor,
) -> AlignedObservationDebug:
    """Gather clean debug values only after noisy observation association."""
    if detection_indices.ndim != 3:
        raise ValueError("detection_indices must have shape (V,T,Q).")
    if detection_indices.dtype != torch.long:
        raise TypeError("detection_indices must have dtype torch.long.")
    if candidate_gt_index.shape != detection_indices.shape:
        raise ValueError("candidate_gt_index must match detection_indices.")
    if candidate_gt_index.dtype != torch.long:
        raise TypeError("candidate_gt_index must have dtype torch.long.")
    if clean.uv.shape[:2] != detection_indices.shape[:2]:
        raise ValueError("Clean and tracked view/time axes must match.")
    num_carriers = clean.uv.shape[2]
    if bool((detection_indices < -1).any()) or bool(
        (detection_indices >= num_carriers).any()
    ):
        raise ValueError("detection_indices contain an out-of-range carrier index.")

    safe_indices = detection_indices.clamp_min(0)
    gathered_uv = torch.gather(
        clean.uv,
        2,
        safe_indices.unsqueeze(-1).expand(*safe_indices.shape, 2),
    )
    gathered_vis = torch.gather(clean.vis, 2, safe_indices)
    valid_debug = (detection_indices >= 0) & (candidate_gt_index >= 0)
    aligned_vis = gathered_vis & valid_debug
    aligned_uv = torch.where(
        aligned_vis.unsqueeze(-1), gathered_uv, torch.zeros_like(gathered_uv)
    )
    aligned_gt_index = torch.where(
        aligned_vis,
        candidate_gt_index,
        torch.full_like(candidate_gt_index, -1),
    )
    return AlignedObservationDebug(
        clean_uv=aligned_uv,
        clean_vis=aligned_vis,
        gt_index=aligned_gt_index,
    )


__all__ = [
    "AlignedObservationDebug",
    "PhysicalObservationCandidates",
    "align_clean_observations_after_tracking",
    "build_physical_observation_candidates",
]
