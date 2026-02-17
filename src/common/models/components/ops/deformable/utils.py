"""Validation and shape helpers for deformable attention."""

from __future__ import annotations

import torch
from torch import Tensor


def validate_msda_inputs(
    value: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> None:
    """Validate tensor ranks and key shape constraints for MSDA op."""
    if value.dim() != 4:
        raise ValueError(f"value must be (B,S,H,D), got {tuple(value.shape)}")
    if spatial_shapes.dim() != 2 or spatial_shapes.size(1) != 2:
        raise ValueError(f"spatial_shapes must be (L,2), got {tuple(spatial_shapes.shape)}")
    if level_start_index.dim() != 1:
        raise ValueError(f"level_start_index must be (L,), got {tuple(level_start_index.shape)}")
    if sampling_locations.dim() != 6:
        raise ValueError(f"sampling_locations must be (B,Q,H,L,P,2), got {tuple(sampling_locations.shape)}")
    if attention_weights.dim() != 5:
        raise ValueError(f"attention_weights must be (B,Q,H,L,P), got {tuple(attention_weights.shape)}")
    if sampling_locations.shape[:-1] != attention_weights.shape:
        raise ValueError("sampling_locations and attention_weights leading dims must match.")

    bsz, total_tokens, heads, _ = value.shape
    b2, _, h2, levels, _, two = sampling_locations.shape
    if two != 2:
        raise ValueError("sampling_locations last dim must be 2.")
    if bsz != b2:
        raise ValueError(f"Batch mismatch between value({bsz}) and sampling_locations({b2}).")
    if heads != h2:
        raise ValueError(f"Head mismatch between value({heads}) and sampling_locations({h2}).")
    if levels != spatial_shapes.size(0) or levels != level_start_index.numel():
        raise ValueError("Level dimension mismatch among spatial_shapes/level_start_index/sampling_locations.")

    expected = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item())
    if expected != total_tokens:
        raise ValueError(f"value token length mismatch: expected {expected}, got {total_tokens}.")


def build_level_start_index(spatial_shapes: Tensor) -> Tensor:
    """Create flattened level start offsets from `(L,2)` spatial shapes."""
    if spatial_shapes.dim() != 2 or spatial_shapes.size(1) != 2:
        raise ValueError(f"spatial_shapes must be (L,2), got {tuple(spatial_shapes.shape)}")
    areas = spatial_shapes[:, 0] * spatial_shapes[:, 1]
    starts = torch.zeros_like(areas)
    if areas.numel() > 1:
        starts[1:] = torch.cumsum(areas[:-1], dim=0)
    return starts
