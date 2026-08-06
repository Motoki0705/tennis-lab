from __future__ import annotations

import torch


def build_local_attention_keep_mask(
    valid_mask: torch.Tensor,
    window_radius: int,
) -> torch.Tensor:
    """Prepare the local-attention keep-mask and empty-row policy at a boundary."""
    seq_len = valid_mask.shape[1]
    positions = torch.arange(seq_len, device=valid_mask.device)

    local_keep = (positions[:, None] - positions[None, :]).abs() <= window_radius
    keep_mask = valid_mask[:, None, :] & local_keep.unsqueeze(0)
    fallback_keep = valid_mask[:, None, :].expand_as(keep_mask)
    has_key = keep_mask.any(dim=-1, keepdim=True)
    return torch.where(has_key, keep_mask, fallback_keep)


def normalize_valid_mask(valid_mask: torch.Tensor) -> torch.Tensor:
    if valid_mask.ndim != 2:
        raise ValueError(
            f"valid_mask must have shape [B, T], got {tuple(valid_mask.shape)}"
        )
    valid_fixed = valid_mask.bool()
    fully_masked = ~valid_fixed.any(dim=1)
    if fully_masked.any():
        valid_fixed = valid_fixed.clone()
        valid_fixed[fully_masked, 0] = True
    return valid_fixed


def build_sliding_window_layout(
    *,
    seq_len: int,
    window_radius: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(seq_len, device=device, dtype=torch.long)

    offsets = torch.arange(
        -window_radius,
        window_radius + 1,
        device=device,
        dtype=torch.long,
    )
    raw_indices = positions[:, None] + offsets[None, :]

    valid = (raw_indices >= 0) & (raw_indices < seq_len)
    indices = raw_indices.clamp(0, seq_len - 1)
    return indices, valid
