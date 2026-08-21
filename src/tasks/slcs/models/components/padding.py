"""Padding-derived validity and attention masks for the SLCS token layout."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.utils.models.transformer_utils import build_self_attn_mask


@dataclass(frozen=True, slots=True)
class SLCSPaddingMasks:
    """Internal valid-polarity state derived from the two public padding masks."""

    frame_state_valid: Tensor
    entity_state_valid: Tensor
    entity_attention_keep_mask: Tensor
    time_state_valid: Tensor
    time_attention_keep_mask: Tensor
    dino_sample_valid: Tensor
    dino_attention_keep_mask: Tensor
    dino_batch_has_evidence: Tensor


def _validate_padding_mask(
    name: str,
    padding_mask: Tensor,
    *,
    batch_size: int | None = None,
) -> None:
    if not isinstance(padding_mask, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if padding_mask.dtype != torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool, got {padding_mask.dtype}.")
    if padding_mask.ndim != 2:
        raise ValueError(
            f"{name} must have shape (B,T), got shape {tuple(padding_mask.shape)}."
        )
    if any(axis_size == 0 for axis_size in padding_mask.shape):
        raise ValueError(
            f"{name} axes must be nonempty, got shape {tuple(padding_mask.shape)}."
        )
    if batch_size is not None and padding_mask.shape[0] != batch_size:
        raise ValueError(
            f"{name} batch axis must be {batch_size}, got {padding_mask.shape[0]}."
        )


def build_slcs_padding_masks(
    padding_mask: Tensor,
    dino_padding_mask: Tensor,
    *,
    num_entities: int,
    dino_tokens_per_sample: int,
) -> SLCSPaddingMasks:
    """Derive all SLCS internal valid/keep masks from public padding masks.

    Public masks use ``True=padding``. Returned state-valid and attention
    keep-masks use ``True=valid/keep``. Fully padded attention rows receive a
    token-0 repair solely to keep softmax finite; raw state-valid tensors are
    left unchanged so callers can zero the repaired state afterward.
    """
    _validate_padding_mask("padding_mask", padding_mask)
    _validate_padding_mask(
        "dino_padding_mask",
        dino_padding_mask,
        batch_size=padding_mask.shape[0],
    )
    if type(num_entities) is not int or num_entities <= 0:
        raise ValueError(f"num_entities must be a positive int, got {num_entities!r}.")
    if type(dino_tokens_per_sample) is not int or dino_tokens_per_sample <= 0:
        raise ValueError(
            "dino_tokens_per_sample must be a positive int, "
            f"got {dino_tokens_per_sample!r}."
        )

    batch_size, seq_len = padding_mask.shape
    dino_samples = dino_padding_mask.shape[1]
    frame_state_valid = ~padding_mask
    entity_state_valid = frame_state_valid.unsqueeze(-1).expand(
        batch_size, seq_len, num_entities
    )

    entity_axis_valid = entity_state_valid.reshape(batch_size * seq_len, num_entities)
    entity_attention_keep_mask, _ = build_self_attn_mask(entity_axis_valid)

    time_state_valid = entity_state_valid.permute(0, 2, 1).reshape(
        batch_size * num_entities, seq_len
    )
    time_attention_keep_mask, _ = build_self_attn_mask(time_state_valid)

    dino_sample_valid = ~dino_padding_mask
    dino_key_valid = (
        dino_sample_valid.unsqueeze(-1)
        .expand(batch_size, dino_samples, dino_tokens_per_sample)
        .reshape(batch_size, dino_samples * dino_tokens_per_sample)
    )
    dino_batch_has_evidence = dino_key_valid.any(dim=1)
    repaired_dino_key_valid = dino_key_valid.clone()
    repaired_dino_key_valid[:, 0] |= ~dino_batch_has_evidence
    dino_attention_keep_mask = repaired_dino_key_valid[:, None, :].expand(
        batch_size,
        seq_len * num_entities,
        dino_samples * dino_tokens_per_sample,
    )

    return SLCSPaddingMasks(
        frame_state_valid=frame_state_valid,
        entity_state_valid=entity_state_valid,
        entity_attention_keep_mask=entity_attention_keep_mask,
        time_state_valid=time_state_valid,
        time_attention_keep_mask=time_attention_keep_mask,
        dino_sample_valid=dino_sample_valid,
        dino_attention_keep_mask=dino_attention_keep_mask,
        dino_batch_has_evidence=dino_batch_has_evidence,
    )


__all__ = ["SLCSPaddingMasks", "build_slcs_padding_masks"]
