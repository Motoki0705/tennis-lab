"""Shared helper mixin for axial multi-view transformer models.

Extracts the helper methods that are byte-identical between
``PLCSMultiViewAxialModel`` and ``BLCSMultiViewAxialModel``.

The mixin holds no parameters or buffers, so it does not affect the
``state_dict`` of consuming models.  Consuming models inherit it as::

    class X(AxialMultiViewMixin, nn.Module):
        def __init__(self, ...):
            super().__init__()
            ...

so that ``nn.Module.__init__`` still runs (the mixin defines no ``__init__``).

The instance methods reference attributes that the consuming model provides:
``self.rope_dim`` and ``self.token_freqs_cis`` (registered as a buffer by the
consuming model).
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.utils.models.transformer_utils import (
    build_self_attn_mask,
    validate_rope_dim,
)


class AxialMultiViewMixin:
    """Helper methods shared by axial multi-view PLCS/BLCS models."""

    # Provided by the consuming model.
    rope_dim: int
    token_freqs_cis: Tensor

    @staticmethod
    def _validate_rope_dim(*, rope_dim: int, head_dim: int) -> None:
        validate_rope_dim(rope_dim=rope_dim, head_dim=head_dim)

    @staticmethod
    def _build_self_attn_mask(valid: Tensor) -> tuple[Tensor, Tensor]:
        """Build self-attention mask from valid mask.

        Delegates to :func:`src.utils.models.build_self_attn_mask`.
        See that function for full documentation.
        """
        return build_self_attn_mask(valid)

    @staticmethod
    def _build_token_positions(*, seq_len: int, n_cams: int) -> Tensor:
        time_idx = torch.arange(seq_len, dtype=torch.long) + 1
        camera_idx = torch.arange(n_cams, dtype=torch.long)
        return torch.stack(
            [
                time_idx[:, None].expand(seq_len, n_cams),
                camera_idx[None, :].expand(seq_len, n_cams),
            ],
            dim=-1,
        )

    def _camera_freqs(self, *, batch_size: int, seq_len: int, n_cams: int) -> Tensor:
        freqs = self.token_freqs_cis[:seq_len, :n_cams]
        return (
            freqs.unsqueeze(0)
            .expand(
                batch_size,
                seq_len,
                n_cams,
                self.rope_dim // 2,
            )
            .reshape(batch_size * seq_len, n_cams, self.rope_dim // 2)
        )

    def _time_freqs(self, *, batch_size: int, seq_len: int, n_cams: int) -> Tensor:
        freqs = self.token_freqs_cis[:seq_len, :n_cams].permute(1, 0, 2)
        return (
            freqs.unsqueeze(0)
            .expand(
                batch_size,
                n_cams,
                seq_len,
                self.rope_dim // 2,
            )
            .reshape(batch_size * n_cams, seq_len, self.rope_dim // 2)
        )
