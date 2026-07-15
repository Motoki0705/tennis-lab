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

The instance methods reference ``self.rope_dim`` and ``self.token_freqs_cis``
(registered as a buffer by the consuming model). The buffer is either a
``(time, camera, freq)`` grid or a ``(time, camera, token, freq)`` grid.
"""

from __future__ import annotations

from typing import cast

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
    def _coalesce_theta(theta: float | None, fallback: float) -> float:
        return fallback if theta is None else float(theta)

    @staticmethod
    def _validate_rope_dim(*, rope_dim: int, head_dim: int) -> None:
        validate_rope_dim(rope_dim=rope_dim, head_dim=head_dim)

    @staticmethod
    def _validate_rope_axis_capacity(*, rope_dim: int, n_axes: int) -> None:
        if rope_dim // 2 < n_axes:
            raise ValueError(
                f"rope_dim={rope_dim} has fewer rotary pairs than n_axes={n_axes}."
            )

    @staticmethod
    def _build_self_attn_mask(valid: Tensor) -> tuple[Tensor, Tensor]:
        """Build self-attention mask from valid mask.

        Delegates to :func:`src.utils.models.build_self_attn_mask`.
        See that function for full documentation.
        """
        return cast(tuple[Tensor, Tensor], build_self_attn_mask(valid))

    @staticmethod
    def _build_token_positions(
        *,
        seq_len: int,
        n_cams: int,
        token_type_ids: Tensor | None = None,
    ) -> Tensor:
        time_idx = torch.arange(seq_len, dtype=torch.long) + 1
        camera_idx = torch.arange(n_cams, dtype=torch.long)
        positions = torch.stack(
            [
                time_idx[:, None].expand(seq_len, n_cams),
                camera_idx[None, :].expand(seq_len, n_cams),
            ],
            dim=-1,
        )
        if token_type_ids is None:
            return positions
        if token_type_ids.ndim != 1 or token_type_ids.numel() <= 0:
            raise ValueError("token_type_ids must be a non-empty 1D tensor.")
        token_count = int(token_type_ids.numel())
        positions = positions.unsqueeze(2).expand(seq_len, n_cams, token_count, 2)
        type_positions = token_type_ids.to(dtype=torch.long).view(1, 1, token_count, 1)
        type_positions = type_positions.expand(seq_len, n_cams, token_count, 1)
        return torch.cat((positions, type_positions), dim=-1)

    @staticmethod
    def _build_line_token_type_ids(num_court_tokens: int) -> Tensor:
        """Return pure token types: object=0 and every court token=1."""
        if num_court_tokens <= 0:
            raise ValueError("num_court_tokens must be positive.")
        return torch.cat(
            (
                torch.zeros(1, dtype=torch.long),
                torch.ones(num_court_tokens, dtype=torch.long),
            )
        )

    def _camera_freqs(
        self,
        *,
        batch_size: int,
        seq_len: int,
        n_cams: int,
        tokens_per_camera: int = 1,
    ) -> Tensor:
        if tokens_per_camera <= 0:
            raise ValueError("tokens_per_camera must be positive.")
        freqs = self.token_freqs_cis[:seq_len, :n_cams]
        if freqs.ndim == 3:
            if tokens_per_camera != 1:
                raise ValueError("2-axis RoPE only supports one token per camera.")
            freqs = freqs.unsqueeze(2)
        elif freqs.ndim == 4:
            if freqs.shape[2] != tokens_per_camera:
                raise ValueError(
                    "RoPE token axis does not match tokens_per_camera: "
                    f"{freqs.shape[2]} != {tokens_per_camera}."
                )
        else:
            raise ValueError(f"Unexpected token_freqs_cis rank: {freqs.ndim}.")
        axis_tokens = n_cams * tokens_per_camera
        return (
            freqs.reshape(seq_len, axis_tokens, self.rope_dim // 2).unsqueeze(0)
            .expand(
                batch_size,
                seq_len,
                axis_tokens,
                self.rope_dim // 2,
            )
            .reshape(batch_size * seq_len, axis_tokens, self.rope_dim // 2)
        )

    def _time_freqs(
        self,
        *,
        batch_size: int,
        seq_len: int,
        n_cams: int,
        tokens_per_camera: int = 1,
    ) -> Tensor:
        if tokens_per_camera <= 0:
            raise ValueError("tokens_per_camera must be positive.")
        freqs = self.token_freqs_cis[:seq_len, :n_cams]
        if freqs.ndim == 3:
            if tokens_per_camera != 1:
                raise ValueError("2-axis RoPE only supports one token per camera.")
            freqs = freqs.permute(1, 0, 2)
        elif freqs.ndim == 4:
            if freqs.shape[2] != tokens_per_camera:
                raise ValueError(
                    "RoPE token axis does not match tokens_per_camera: "
                    f"{freqs.shape[2]} != {tokens_per_camera}."
                )
            freqs = freqs.permute(1, 2, 0, 3).reshape(
                n_cams * tokens_per_camera,
                seq_len,
                self.rope_dim // 2,
            )
        else:
            raise ValueError(f"Unexpected token_freqs_cis rank: {freqs.ndim}.")
        axis_tokens = n_cams * tokens_per_camera
        return (
            freqs.unsqueeze(0)
            .expand(
                batch_size,
                axis_tokens,
                seq_len,
                self.rope_dim // 2,
            )
            .reshape(batch_size * axis_tokens, seq_len, self.rope_dim // 2)
        )
