"""Compressed sliding-window self-attention."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.utils.models.components.compressor import (
    TokenLevelCompressorConfig,
    TokenLevelKVCompressor,
)
from src.utils.models.components.ops.compressed_time_local import (
    CompressedTimeLocalAttentionExecutor,
    resolve_compressed_time_local_attention,
)
from src.utils.models.components.rope import RotaryFrequencyComputer


@dataclass(frozen=True, slots=True)
class CSWAConfig:
    """Construction-time contract for compressed sliding-window attention."""

    dim: int
    n_heads: int
    head_dim: int
    rope_dim: int
    attn_dropout: float
    compression_ratio: int
    window_radius: int
    backend: Literal["reference", "cuda"]

    def __post_init__(self) -> None:
        for name in (
            "dim",
            "n_heads",
            "head_dim",
            "rope_dim",
            "compression_ratio",
            "window_radius",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer, got {value!r}")
        for name in ("dim", "n_heads", "head_dim"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        projected_dim = self.n_heads * self.head_dim
        if self.dim != projected_dim:
            raise ValueError(
                f"dim must equal n_heads * head_dim ({projected_dim}), got {self.dim}"
            )
        if self.rope_dim <= 0 or self.rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be positive and even, got {self.rope_dim}")
        if self.rope_dim > self.head_dim:
            raise ValueError(
                f"rope_dim={self.rope_dim} cannot exceed head_dim={self.head_dim}"
            )
        if isinstance(self.attn_dropout, bool) or not isinstance(
            self.attn_dropout, (int, float)
        ):
            raise TypeError(
                f"attn_dropout must be a real number, got {self.attn_dropout!r}"
            )
        if not math.isfinite(float(self.attn_dropout)) or not (
            0.0 <= float(self.attn_dropout) < 1.0
        ):
            raise ValueError(
                f"attn_dropout must be finite and in [0, 1), got {self.attn_dropout!r}"
            )
        if self.compression_ratio < 2:
            raise ValueError(
                f"compression_ratio must be at least 2, got {self.compression_ratio}"
            )
        if self.window_radius < 0:
            raise ValueError(
                f"window_radius must be non-negative, got {self.window_radius}"
            )
        if self.backend not in ("reference", "cuda"):
            raise ValueError(f"Unsupported CSWA backend: {self.backend!r}")


class CompressedSlidingWindowSelfAttention(nn.Module):
    """Self-attention from raw queries to token-level compressed K/V states.

    ``x`` has shape ``[N,T,D]``, ``state_valid`` is boolean ``[N,T]``, and
    ``freqs_cis`` contains caller-prepared query-position RoPE frequencies.
    Compressed-key frequencies are computed separately from the compressor's
    deterministic block-center positions.
    """

    _SUPPORTED_DTYPES = {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }

    def __init__(self, config: CSWAConfig) -> None:
        super().__init__()
        if not isinstance(config, CSWAConfig):
            raise TypeError(f"config must be CSWAConfig, got {type(config).__name__}")
        self.config = config
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.rope_dim = config.rope_dim
        self.attn_dropout = float(config.attn_dropout)
        self.compression_ratio = config.compression_ratio
        self.window_radius = config.window_radius
        self.backend = config.backend

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.compressor = TokenLevelKVCompressor(
            TokenLevelCompressorConfig(
                dim=self.dim,
                n_heads=self.n_heads,
                head_dim=self.head_dim,
                compression_ratio=self.compression_ratio,
                overlap=True,
            )
        )
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)
        self.compressed_frequency_computer = RotaryFrequencyComputer(
            dim=self.rope_dim,
            base=10000.0,
            n_axes=1,
        )
        self.executor: CompressedTimeLocalAttentionExecutor = (
            resolve_compressed_time_local_attention(
                self.backend,
                compression_ratio=self.compression_ratio,
                window_radius=self.window_radius,
            )
        )
        self.register_forward_pre_hook(
            self._validate_forward_inputs,
            with_kwargs=True,
        )

    @staticmethod
    def _project(x: Tensor, layer: nn.Linear) -> Tensor:
        """Project in input dtype so compressor and query dtypes stay aligned."""
        bias = None if layer.bias is None else layer.bias.to(dtype=x.dtype)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return F.linear(x, layer.weight.to(dtype=x.dtype), bias)

    @staticmethod
    def _apply_rope(x: Tensor, freqs_cis: Tensor) -> Tensor:
        """Apply RoPE while retaining float64 precision for gradcheck."""
        output_dtype = x.dtype
        real_dtype = torch.float64 if output_dtype == torch.float64 else torch.float32
        complex_dtype = (
            torch.complex128 if real_dtype == torch.float64 else torch.complex64
        )
        x_complex = torch.view_as_complex(
            x.to(dtype=real_dtype).reshape(*x.shape[:-1], -1, 2)
        )
        rotated = x_complex * freqs_cis.to(dtype=complex_dtype)
        return torch.view_as_real(rotated).flatten(-2).to(dtype=output_dtype)

    def validate_inputs(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        state_valid: Tensor,
    ) -> None:
        """Validate the public module-call boundary before tensor computation."""
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape [N, T, {self.dim}], got {tuple(x.shape)}"
            )
        n, query_len, dim = x.shape
        if n <= 0 or query_len <= 0:
            raise ValueError("x batch and sequence dimensions must be positive")
        if dim != self.dim:
            raise ValueError(f"x feature dimension must be {self.dim}, got {dim}")
        if x.dtype not in self._SUPPORTED_DTYPES:
            raise TypeError(f"x must be floating point, got {x.dtype}")
        if state_valid.shape != (n, query_len):
            raise ValueError(
                f"state_valid must have shape {(n, query_len)}, "
                f"got {tuple(state_valid.shape)}"
            )
        if state_valid.dtype != torch.bool:
            raise TypeError(
                f"state_valid must have dtype bool, got {state_valid.dtype}"
            )
        if state_valid.device != x.device:
            raise ValueError("x and state_valid must be on the same device")
        if not freqs_cis.is_complex():
            raise TypeError(f"freqs_cis must be complex, got {freqs_cis.dtype}")
        if freqs_cis.device != x.device:
            raise ValueError("x and freqs_cis must be on the same device")
        pairs = self.rope_dim // 2
        if freqs_cis.ndim not in (3, 4):
            raise ValueError(
                "freqs_cis must be broadcastable as [N,T,H,rope_dim/2], "
                f"got shape {tuple(freqs_cis.shape)}"
            )
        if freqs_cis.shape[-3] != query_len:
            raise ValueError(
                f"freqs_cis query length must be {query_len}, got {freqs_cis.shape[-3]}"
            )
        if freqs_cis.shape[-2] not in (1, self.n_heads):
            raise ValueError(
                f"freqs_cis head dimension must be 1 or {self.n_heads}, "
                f"got {freqs_cis.shape[-2]}"
            )
        if freqs_cis.shape[-1] != pairs:
            raise ValueError(
                f"freqs_cis last dimension must be {pairs}, got {freqs_cis.shape[-1]}"
            )
        if freqs_cis.ndim == 4 and freqs_cis.shape[0] not in (1, n):
            raise ValueError(
                f"freqs_cis batch dimension must be 1 or {n}, got {freqs_cis.shape[0]}"
            )

        parameter_device = self.wq.weight.device
        if x.device != parameter_device:
            raise ValueError(
                f"x must be on module device {parameter_device}, got {x.device}"
            )

    def _validate_forward_inputs(
        self,
        _module: nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        x = cast(Tensor, args[0] if args else kwargs["x"])
        freqs_cis = cast(Tensor, kwargs["freqs_cis"])
        state_valid = cast(Tensor, kwargs["state_valid"])
        self.validate_inputs(x, freqs_cis, state_valid)

    def forward(
        self,
        x: Tensor,
        *,
        freqs_cis: Tensor,
        state_valid: Tensor,
    ) -> Tensor:
        """Return compressed-window attention output with shape ``[N,T,D]``."""
        n, query_len, _ = x.shape
        masked_x = torch.where(state_valid.unsqueeze(-1), x, torch.zeros_like(x))

        query = self._project(masked_x, self.wq).reshape(
            n, query_len, self.n_heads, self.head_dim
        )
        query_rope = self._apply_rope(query[..., : self.rope_dim], freqs_cis)
        query = torch.cat((query_rope, query[..., self.rope_dim :]), dim=-1)
        query = query.transpose(1, 2)

        compressed = self.compressor(masked_x, state_valid)
        key = compressed.key.transpose(1, 2)
        key_positions = compressed.positions.to(
            device=x.device,
            dtype=self.compressed_frequency_computer.inverse_frequencies.dtype,
        ).unsqueeze(-1)
        key_freqs_cis = self.compressed_frequency_computer(key_positions)
        key_rope = self._apply_rope(key[..., : self.rope_dim], key_freqs_cis)
        key = torch.cat((key_rope, key[..., self.rope_dim :]), dim=-1).transpose(1, 2)

        output = self.executor(
            query,
            key,
            compressed.value,
            query_valid=state_valid,
            key_valid=compressed.state_valid,
            dropout_p=self.attn_dropout,
            training=self.training,
        )
        output = output.transpose(1, 2).reshape(
            n, query_len, self.n_heads * self.head_dim
        )
        output = self._project(output, self.wo)
        return torch.where(state_valid.unsqueeze(-1), output, torch.zeros_like(output))
