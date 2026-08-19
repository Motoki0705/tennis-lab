"""Fixed-ratio token-level key/value compression for temporal attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class TokenLevelCompressorConfig:
    """Construction-time contract for :class:`TokenLevelKVCompressor`.

    ``dim`` must equal ``n_heads * head_dim``.  Version 1 always uses two
    branches: the previous block and the current block.  Consequently,
    ``overlap`` must be ``True`` and ``compression_ratio`` must be at least 2.
    """

    dim: int
    n_heads: int
    head_dim: int
    compression_ratio: int
    overlap: bool

    def __post_init__(self) -> None:
        for name in ("dim", "n_heads", "head_dim", "compression_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, got {value!r}")
        for name in ("dim", "n_heads", "head_dim"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.compression_ratio < 2:
            raise ValueError(
                f"compression_ratio must be at least 2, got {self.compression_ratio}"
            )
        if not isinstance(self.overlap, bool):
            raise TypeError(f"overlap must be a bool, got {self.overlap!r}")
        if not self.overlap:
            raise ValueError("overlap=False is unsupported; overlap must be True")
        projected_dim = self.n_heads * self.head_dim
        if self.dim != projected_dim:
            raise ValueError(
                f"dim must equal n_heads * head_dim ({projected_dim}), got {self.dim}"
            )


@dataclass(slots=True)
class CompressedKV:
    """Compressed key/value tensors and their temporal metadata.

    Attributes:
        key: Key tensor with shape ``[N, H, Tc, Dh]``.
        value: Value tensor with shape ``[N, H, Tc, Dh]``.
        state_valid: Boolean compressed-state mask with shape ``[N, Tc]``.
        positions: Static block centers with shape ``[Tc]`` and dtype float32.
    """

    key: Tensor
    value: Tensor
    state_valid: Tensor
    positions: Tensor


@dataclass(frozen=True, slots=True)
class _CompressionLayout:
    """Data-independent source layout for one sequence length."""

    source_indices: Tensor
    source_branches: Tensor
    boundary_valid: Tensor


def _build_compression_layout(
    sequence_length: int,
    compression_ratio: int,
    device: torch.device,
) -> _CompressionLayout:
    """Build ``[Tc, 2m]`` previous/current source indices without dense masks."""
    compressed_length = (sequence_length + compression_ratio - 1) // compression_ratio
    offsets = torch.arange(compression_ratio, device=device)
    current_starts = torch.arange(compressed_length, device=device) * compression_ratio
    current = current_starts[:, None] + offsets[None, :]
    previous = current - compression_ratio
    source_indices = torch.cat((previous, current), dim=1)
    boundary_valid = (source_indices >= 0) & (source_indices < sequence_length)
    safe_indices = source_indices.clamp(min=0, max=sequence_length - 1)
    source_branches = torch.cat(
        (
            torch.zeros_like(previous),
            torch.ones_like(current),
        ),
        dim=1,
    )
    return _CompressionLayout(
        source_indices=safe_indices,
        source_branches=source_branches,
        boundary_valid=boundary_valid,
    )


class TokenLevelKVCompressor(nn.Module):
    """Compress normalized temporal tokens into overlapped key/value states.

    The input ``x`` has shape ``[N, T, D]`` and ``state_valid`` has shape
    ``[N, T]``.  For compression ratio ``m``, each output state pools the
    previous block through branch 0 and the current block through branch 1.
    Channel-wise masked softmax and weighted accumulation use float32, except
    that float64 inputs retain float64 for numerical gradient checking.  Key
    and value outputs are converted back to the input dtype.

    Invalid input tokens are zeroed before either projection.  A compressed
    row with no valid source is returned as exactly zero and marked invalid.
    """

    _SUPPORTED_DTYPES = {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }

    def __init__(self, config: TokenLevelCompressorConfig) -> None:
        super().__init__()
        if not isinstance(config, TokenLevelCompressorConfig):
            raise TypeError(
                "config must be TokenLevelCompressorConfig, "
                f"got {type(config).__name__}"
            )

        self.config = config
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.compression_ratio = config.compression_ratio
        self.overlap = config.overlap
        self.branches = 2
        self.kv_dim = 2 * self.n_heads * self.head_dim

        projection_dim = self.branches * self.kv_dim
        self.w_kv = nn.Linear(self.dim, projection_dim)
        self.w_gate = nn.Linear(self.dim, projection_dim)
        self.ape = nn.Parameter(torch.zeros(self.compression_ratio, projection_dim))
        nn.init.zeros_(self.w_gate.weight)
        nn.init.zeros_(self.w_gate.bias)
        self.register_forward_pre_hook(
            self._validate_forward_inputs,
            with_kwargs=True,
        )

    @staticmethod
    def _accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
        return torch.float64 if dtype == torch.float64 else torch.float32

    @staticmethod
    def _project(x: Tensor, layer: nn.Linear) -> Tensor:
        """Project in the input dtype, independent of an outer autocast scope."""
        bias = None if layer.bias is None else layer.bias.to(dtype=x.dtype)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return F.linear(x, layer.weight.to(dtype=x.dtype), bias)

    def validate_inputs(self, x: Tensor, state_valid: Tensor) -> None:
        """Validate the public module-call boundary before tensor computation."""
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape [N, T, {self.dim}], got {tuple(x.shape)}"
            )
        if x.shape[2] != self.dim:
            raise ValueError(
                f"x feature dimension must be {self.dim}, got {x.shape[2]}"
            )
        if x.shape[1] <= 0:
            raise ValueError("x sequence length T must be positive")
        if x.dtype not in self._SUPPORTED_DTYPES:
            raise TypeError(
                f"x must use float16, bfloat16, float32, or float64, got {x.dtype}"
            )
        expected_mask_shape = x.shape[:2]
        if state_valid.shape != expected_mask_shape:
            raise ValueError(
                f"state_valid shape must be {tuple(expected_mask_shape)}, "
                f"got {tuple(state_valid.shape)}"
            )
        if state_valid.dtype != torch.bool:
            raise TypeError(
                f"state_valid must have dtype bool, got {state_valid.dtype}"
            )
        if state_valid.device != x.device:
            raise ValueError(
                "x and state_valid must be on the same device, got "
                f"{x.device} and {state_valid.device}"
            )
        parameter_device = self.w_kv.weight.device
        if any(parameter.device != parameter_device for parameter in self.parameters()):
            raise RuntimeError("all compressor parameters must be on the same device")
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
        state_valid = cast(
            Tensor,
            args[1] if len(args) > 1 else kwargs["state_valid"],
        )
        self.validate_inputs(x, state_valid)

    def forward(self, x: Tensor, state_valid: Tensor) -> CompressedKV:
        """Return fixed-ratio compressed key/value states for ``x``."""
        batch_size, sequence_length, _ = x.shape
        input_dtype = x.dtype
        accumulation_dtype = self._accumulation_dtype(input_dtype)

        masked_x = torch.where(state_valid.unsqueeze(-1), x, torch.zeros_like(x))
        raw_kv = self._project(masked_x, self.w_kv).reshape(
            batch_size,
            sequence_length,
            self.branches,
            self.kv_dim,
        )
        raw_gate = self._project(masked_x, self.w_gate).reshape(
            batch_size,
            sequence_length,
            self.branches,
            self.kv_dim,
        )

        offsets = (
            torch.arange(sequence_length, device=x.device) % self.compression_ratio
        )
        positional_gate = self.ape.index_select(0, offsets).reshape(
            sequence_length,
            self.branches,
            self.kv_dim,
        )
        raw_gate = raw_gate.to(dtype=accumulation_dtype) + positional_gate.to(
            dtype=accumulation_dtype
        ).unsqueeze(0)

        layout = _build_compression_layout(
            sequence_length,
            self.compression_ratio,
            x.device,
        )
        gathered_kv = raw_kv[:, layout.source_indices, layout.source_branches, :].to(
            dtype=accumulation_dtype
        )
        gathered_gate = raw_gate[:, layout.source_indices, layout.source_branches, :]

        source_valid = (
            layout.boundary_valid.unsqueeze(0) & state_valid[:, layout.source_indices]
        )
        channel_mask = source_valid.unsqueeze(-1)
        minimum = torch.finfo(accumulation_dtype).min
        masked_gate = gathered_gate.masked_fill(~channel_mask, minimum)
        maxima = masked_gate.amax(dim=2, keepdim=True)
        numerator = torch.exp(masked_gate - maxima) * channel_mask.to(
            dtype=accumulation_dtype
        )
        denominator = numerator.sum(dim=2, keepdim=True)
        weights = torch.where(
            denominator > 0,
            numerator / denominator.clamp_min(torch.finfo(accumulation_dtype).tiny),
            torch.zeros_like(numerator),
        )
        compressed = (weights * gathered_kv).sum(dim=2).to(dtype=input_dtype)
        compressed_valid = source_valid.any(dim=2)
        compressed = torch.where(
            compressed_valid.unsqueeze(-1),
            compressed,
            torch.zeros_like(compressed),
        )

        compressed_length = compressed.shape[1]
        split = compressed.reshape(
            batch_size,
            compressed_length,
            2,
            self.n_heads,
            self.head_dim,
        )
        key, value = split.unbind(dim=2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        positions = (
            torch.arange(
                compressed_length,
                device=x.device,
                dtype=torch.float32,
            )
            * self.compression_ratio
            + (self.compression_ratio - 1) / 2
        ).clamp_max(float(sequence_length - 1))
        return CompressedKV(
            key=key,
            value=value,
            state_valid=compressed_valid,
            positions=positions,
        )
