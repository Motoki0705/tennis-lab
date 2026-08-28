from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Literal, Protocol, TypeVar, cast

import torch
from torch import nn

from src.utils.models.components.attention import (
    GroupedQuerySelfAttention,
    MultiHeadCrossAttention,
    MultiHeadSelfAttention,
)
from src.utils.models.components.cswa import (
    CompressedSlidingWindowSelfAttention,
    CSWAConfig,
)
from src.utils.models.components.ffn_layers import FFNType, build_ffn
from src.utils.models.components.norm import RMSNorm


class _AttentionInvocation(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor | None,
        state_valid: torch.Tensor | None,
    ) -> torch.Tensor: ...


class _AttentionArgumentValidator(Protocol):
    def __call__(
        self,
        *,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        state_valid: torch.Tensor | None,
    ) -> None: ...


class _TransformerBlockForward(Protocol):
    def __call__(
        self,
        block: TransformerBlock,
        /,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        state_valid: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


_TransformerBlockForwardT = TypeVar(
    "_TransformerBlockForwardT",
    bound=_TransformerBlockForward,
)


def _attention_argument_boundary(
    forward: _TransformerBlockForwardT,
) -> _TransformerBlockForwardT:
    """Validate public attention arguments outside the computation-only method."""

    @wraps(forward)
    def validated_forward(
        self: TransformerBlock,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        state_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_attention_arguments(
            x=x,
            attn_mask=attn_mask,
            state_valid=state_valid,
        )
        return forward(
            self,
            x,
            freqs_cis=freqs_cis,
            attn_mask=attn_mask,
            state_valid=state_valid,
        )

    return cast(_TransformerBlockForwardT, validated_forward)


@dataclass
class TransformerBlockConfig:
    """Configuration for TransformerBlock.

    Args:
        dim: Token embedding dimension.
        n_heads: Number of attention heads.
        ffn_dim: Hidden dimension for the FFN. Defaults to the repository-wide transformer FFN heuristic.
        head_dim: Per-head dimension (defaults to dim // n_heads).
        rope_dim: Rotary dimension per head for 1D RoPE.
        attn_dropout: Dropout probability for attention.
        attention_type: Self-attention implementation to use.
        n_kv_heads: Number of key/value heads for GQA.
        rope_base: Base theta for 1D RoPE.
        ffn_type: FFN implementation to use.
        cswa: Compressed sliding-window attention configuration. Required only
            when ``attention_type='cswa'``.
        ffn_enabled: Whether this block owns and applies an FFN. The default
            preserves the legacy block parameter and execution contract.
    """

    dim: int
    n_heads: int
    ffn_dim: int
    # attention
    head_dim: int
    rope_dim: int
    attn_dropout: float
    attention_type: Literal["mha", "gqa", "cswa"]
    n_kv_heads: int | None
    # RoPE
    rope_base: float
    # FFN
    ffn_type: FFNType
    # Optional fields keep their legacy positional order; new options append.
    cswa: CSWAConfig | None = None
    ffn_enabled: bool = True


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with explicit residual additions.
    """

    def __init__(self, cfg: TransformerBlockConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.attn_norm = RMSNorm(cfg.dim)
        self.attn: (
            MultiHeadSelfAttention
            | GroupedQuerySelfAttention
            | CompressedSlidingWindowSelfAttention
        )
        if cfg.attention_type == "mha":
            if cfg.n_kv_heads is not None:
                raise ValueError("n_kv_heads must be None when attention_type='mha'")
            if cfg.cswa is not None:
                raise ValueError("cswa must be None when attention_type='mha'")
            self.attn = MultiHeadSelfAttention(
                dim=cfg.dim,
                n_heads=cfg.n_heads,
                head_dim=cfg.head_dim,
                rope_dim=cfg.rope_dim,
                attn_dropout=cfg.attn_dropout,
                bias=False,
            )
            self._invoke_attention: _AttentionInvocation = self._invoke_dense_attention
            self._validate_attention_arguments: _AttentionArgumentValidator = (
                self._validate_dense_attention_arguments
            )
        elif cfg.attention_type == "gqa":
            if cfg.n_kv_heads is None:
                raise ValueError("n_kv_heads must be set when attention_type='gqa'")
            if cfg.cswa is not None:
                raise ValueError("cswa must be None when attention_type='gqa'")
            self.attn = GroupedQuerySelfAttention(
                dim=cfg.dim,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                head_dim=cfg.head_dim,
                rope_dim=cfg.rope_dim,
                attn_dropout=cfg.attn_dropout,
                bias=False,
            )
            self._invoke_attention = self._invoke_dense_attention
            self._validate_attention_arguments = (
                self._validate_dense_attention_arguments
            )
        elif cfg.attention_type == "cswa":
            if cfg.n_kv_heads is not None:
                raise ValueError("n_kv_heads must be None when attention_type='cswa'")
            if cfg.cswa is None:
                raise ValueError("cswa must be set when attention_type='cswa'")
            self._validate_cswa_config(cfg, cfg.cswa)
            self.attn = CompressedSlidingWindowSelfAttention(cfg.cswa)
            self._invoke_attention = self._invoke_cswa_attention
            self._validate_attention_arguments = self._validate_cswa_attention_arguments
        else:
            raise ValueError(f"Unsupported attention_type={cfg.attention_type}")

        if type(cfg.ffn_enabled) is not bool:
            raise TypeError("ffn_enabled must be exactly bool")
        self.ffn_norm: RMSNorm | None
        self.ffn: nn.Module | None
        if cfg.ffn_enabled:
            self.ffn_norm = RMSNorm(cfg.dim)
            self.ffn = build_ffn(
                ffn_type=cfg.ffn_type,
                dim=cfg.dim,
                ffn_dim=cfg.ffn_dim,
            )
        else:
            self.ffn_norm = None
            self.ffn = None
        self.register_forward_pre_hook(
            self._validate_forward_arguments,
            with_kwargs=True,
        )

    @staticmethod
    def _validate_cswa_config(
        cfg: TransformerBlockConfig,
        cswa: CSWAConfig,
    ) -> None:
        for field in ("dim", "n_heads", "head_dim", "rope_dim", "attn_dropout"):
            block_value = getattr(cfg, field)
            cswa_value = getattr(cswa, field)
            if block_value != cswa_value:
                raise ValueError(
                    f"cswa.{field} must match TransformerBlockConfig.{field}: "
                    f"{cswa_value!r} != {block_value!r}"
                )

    def _invoke_dense_attention(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor | None,
        state_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        del state_valid
        dense_attention = cast(
            MultiHeadSelfAttention | GroupedQuerySelfAttention,
            self.attn,
        )
        return cast(
            torch.Tensor,
            dense_attention(
                x,
                freqs_cis=freqs_cis,
                attn_mask=cast(torch.Tensor, attn_mask),
            ),
        )

    def _invoke_cswa_attention(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor | None,
        state_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        del attn_mask
        cswa_attention = cast(CompressedSlidingWindowSelfAttention, self.attn)
        return cast(
            torch.Tensor,
            cswa_attention(
                x,
                freqs_cis=freqs_cis,
                state_valid=cast(torch.Tensor, state_valid),
            ),
        )

    @staticmethod
    def _validate_dense_attention_arguments(
        *,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        state_valid: torch.Tensor | None,
    ) -> None:
        if state_valid is not None:
            raise ValueError("state_valid is prohibited for MHA/GQA attention")
        if attn_mask is None:
            raise ValueError("attn_mask is required for MHA/GQA attention")
        if attn_mask.dtype is not torch.bool:
            raise ValueError(
                "attn_mask must have dtype torch.bool for MHA/GQA attention, "
                f"got {attn_mask.dtype}"
            )
        if attn_mask.device != x.device:
            raise ValueError(
                "attn_mask must be on the same device as x for MHA/GQA attention, "
                f"got attn_mask.device={attn_mask.device} and x.device={x.device}"
            )
        expected_shape = (x.shape[0], x.shape[1], x.shape[1])
        if attn_mask.shape != expected_shape:
            raise ValueError(
                "attn_mask must have exact shape "
                f"{expected_shape} for MHA/GQA attention, got {tuple(attn_mask.shape)}"
            )

    @staticmethod
    def _validate_cswa_attention_arguments(
        *,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        state_valid: torch.Tensor | None,
    ) -> None:
        del x
        if attn_mask is not None:
            raise ValueError("attn_mask is prohibited for CSWA attention")
        if state_valid is None:
            raise ValueError("state_valid is required for CSWA attention")

    def _validate_forward_arguments(
        self,
        _module: nn.Module,
        _args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        try:
            raw_attn_mask = kwargs["attn_mask"]
        except KeyError:
            raw_attn_mask = None
        try:
            raw_state_valid = kwargs["state_valid"]
        except KeyError:
            raw_state_valid = None
        raw_x = _args[0] if _args else kwargs["x"]
        x = cast(torch.Tensor, raw_x)
        attn_mask = cast(torch.Tensor | None, raw_attn_mask)
        state_valid = cast(torch.Tensor | None, raw_state_valid)
        self._validate_attention_arguments(
            x=x,
            attn_mask=attn_mask,
            state_valid=state_valid,
        )

    def _forward_update_computation(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor | None,
        state_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        attn_output = self._invoke_attention(
            self.attn_norm(x),
            freqs_cis=freqs_cis,
            attn_mask=attn_mask,
            state_valid=state_valid,
        )
        x_attn = x + attn_output
        if self.ffn is None or self.ffn_norm is None:
            return attn_output
        ffn_output = self.ffn(self.ffn_norm(x_attn))
        return cast(torch.Tensor, attn_output + ffn_output)

    def forward_update(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        state_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return only the attention-plus-FFN update, without the outer residual."""
        self._validate_attention_arguments(
            x=x,
            attn_mask=attn_mask,
            state_valid=state_valid,
        )
        return self._forward_update_computation(
            x,
            freqs_cis=freqs_cis,
            attn_mask=attn_mask,
            state_valid=state_valid,
        )

    @_attention_argument_boundary
    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        state_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the residual state after the configured attention and FFN."""
        return x + self._forward_update_computation(
            x,
            freqs_cis=freqs_cis,
            attn_mask=attn_mask,
            state_valid=state_valid,
        )


@dataclass
class CrossAttnBlockConfig:
    """Configuration for CrossAttnBlock."""

    dim: int
    n_heads: int
    ffn_dim: int
    # attention
    head_dim: int
    rope_dim: int
    attn_dropout: float
    # FFN
    ffn_type: FFNType


class CrossAttnBlock(nn.Module):
    """Pre-norm cross-attention block over boundary-prepared tensors."""

    def __init__(self, cfg: CrossAttnBlockConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.q_norm = RMSNorm(cfg.dim)
        self.kv_norm = RMSNorm(cfg.dim)
        self.attn = MultiHeadCrossAttention(
            dim=cfg.dim,
            n_heads=cfg.n_heads,
            head_dim=cfg.head_dim,
            rope_dim=cfg.rope_dim,
            attn_dropout=cfg.attn_dropout,
            bias=False,
        )
        self.ffn_norm = RMSNorm(cfg.dim)
        self.ffn = build_ffn(
            ffn_type=cfg.ffn_type,
            dim=cfg.dim,
            ffn_dim=cfg.ffn_dim,
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        *,
        attn_mask: torch.Tensor,
        freqs_q_cis: torch.Tensor,
        freqs_k_cis: torch.Tensor,
    ) -> torch.Tensor:
        q_norm = self.q_norm(q)
        kv_norm = self.kv_norm(kv)

        q = q + self.attn(
            q_norm,
            kv_norm,
            freqs_q_cis=freqs_q_cis,
            freqs_k_cis=freqs_k_cis,
            attn_mask=attn_mask,
        )
        q = q + self.ffn(self.ffn_norm(q))
        return q
