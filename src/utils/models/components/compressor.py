"""Fixed-ratio token-level key/value compression for temporal attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.utils.models.components.ops.token_compressor import (
    resolve_token_compressor_pool,
)


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
        key: Shared key tensor with shape ``[N, 1, Tc, Dh]``.
        value: The same shared latent as ``key``, with shape ``[N, 1, Tc, Dh]``.
        state_valid: Boolean compressed-state mask with shape ``[N, Tc]``.
        positions: Static block centers with shape ``[Tc]`` and dtype float32.
    """

    key: Tensor
    value: Tensor
    state_valid: Tensor
    positions: Tensor


class TokenLevelKVCompressor(nn.Module):
    """Compress normalized temporal tokens into one overlapped shared KV state.

    The input ``x`` has shape ``[N, T, D]`` and ``state_valid`` has shape
    ``[N, T]``.  For compression ratio ``m``, each output state pools the
    previous block through branch 0 and the current block through branch 1.
    Channel-wise masked softmax and weighted accumulation use float32, except
    that float64 inputs retain float64 for numerical gradient checking.  Key
    and value outputs are converted back to the input dtype.  The resulting
    ``head_dim`` latent is exposed as both a one-head key and one-head value;
    query heads share it through multi-query attention.

    Invalid input tokens are zeroed before either projection.  A compressed
    row with no valid source is returned as exactly zero and marked invalid.
    """

    _SUPPORTED_DTYPES = {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }

    def __init__(
        self,
        config: TokenLevelCompressorConfig,
        *,
        backend: Literal["reference", "cuda"] = "reference",
    ) -> None:
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
        self.backend = backend
        self.branches = 2
        self.kv_dim = self.head_dim

        projection_dim = self.branches * self.kv_dim
        self.w_kv = nn.Linear(self.dim, projection_dim)
        self.w_gate = nn.Linear(self.dim, projection_dim)
        self.ape = nn.Parameter(torch.zeros(self.compression_ratio, projection_dim))
        nn.init.zeros_(self.w_gate.weight)
        nn.init.zeros_(self.w_gate.bias)
        self.pool = resolve_token_compressor_pool(
            self.backend,
            compression_ratio=self.compression_ratio,
            head_dim=self.head_dim,
        )
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
        masked_x = torch.where(state_valid.unsqueeze(-1), x, torch.zeros_like(x))
        return self._forward_masked(masked_x, state_valid)

    def forward_masked(self, masked_x: Tensor, state_valid: Tensor) -> CompressedKV:
        """Compress input already zeroed at invalid states after validation.

        This seam lets an attention caller share one masked token materialization
        between its query projection and compressor projections. Standalone
        callers should use :meth:`forward`, which always applies the mask.
        """
        self.validate_inputs(masked_x, state_valid)
        return self._forward_masked(masked_x, state_valid)

    def validate_projected_inputs(
        self,
        raw_kv: Tensor,
        raw_gate: Tensor,
        state_valid: Tensor,
    ) -> None:
        """Validate the explicit post-projection pooling boundary."""
        expected_suffix = (self.branches, self.kv_dim)
        if raw_kv.ndim != 4 or raw_kv.shape[2:] != expected_suffix:
            raise ValueError(
                "raw_kv must have shape "
                f"[N, T, {self.branches}, {self.kv_dim}], got {tuple(raw_kv.shape)}"
            )
        batch_size, sequence_length = raw_kv.shape[:2]
        if batch_size <= 0 or sequence_length <= 0:
            raise ValueError("raw_kv batch and sequence dimensions must be positive")
        if raw_gate.shape != raw_kv.shape:
            raise ValueError(
                "raw_gate shape must equal raw_kv shape, got "
                f"{tuple(raw_gate.shape)} and {tuple(raw_kv.shape)}"
            )
        if raw_kv.dtype not in self._SUPPORTED_DTYPES:
            raise TypeError(
                "raw_kv must use float16, bfloat16, float32, or float64, "
                f"got {raw_kv.dtype}"
            )
        if raw_gate.dtype != raw_kv.dtype:
            raise TypeError(
                "raw_gate dtype must equal raw_kv dtype, got "
                f"{raw_gate.dtype} and {raw_kv.dtype}"
            )
        if raw_gate.device != raw_kv.device:
            raise ValueError(
                "raw_gate must be on the same device as raw_kv, got "
                f"{raw_gate.device} and {raw_kv.device}"
            )
        expected_mask_shape = (batch_size, sequence_length)
        if state_valid.shape != expected_mask_shape:
            raise ValueError(
                f"state_valid shape must be {expected_mask_shape}, "
                f"got {tuple(state_valid.shape)}"
            )
        if state_valid.dtype != torch.bool:
            raise TypeError(
                f"state_valid must have dtype bool, got {state_valid.dtype}"
            )
        if state_valid.device != raw_kv.device:
            raise ValueError(
                "state_valid must be on the same device as raw_kv, got "
                f"{state_valid.device} and {raw_kv.device}"
            )
        parameter_device = self.w_kv.weight.device
        if any(parameter.device != parameter_device for parameter in self.parameters()):
            raise RuntimeError("all compressor parameters must be on the same device")
        if raw_kv.device != parameter_device:
            raise ValueError(
                "raw_kv must be on module device "
                f"{parameter_device}, got {raw_kv.device}"
            )

    def forward_projected(
        self,
        raw_kv: Tensor,
        raw_gate: Tensor,
        state_valid: Tensor,
    ) -> CompressedKV:
        """Pool validated raw KV/gate projections without projecting tokens."""
        self.validate_projected_inputs(raw_kv, raw_gate, state_valid)
        return self._pool_projected(raw_kv, raw_gate, state_valid)

    def _forward_masked(self, masked_x: Tensor, state_valid: Tensor) -> CompressedKV:
        """Project and pool a validated, already-masked token tensor."""
        batch_size, sequence_length, _ = masked_x.shape
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
        return self.forward_projected(raw_kv, raw_gate, state_valid)

    def _pool_projected(
        self,
        raw_kv: Tensor,
        raw_gate: Tensor,
        state_valid: Tensor,
    ) -> CompressedKV:
        """Apply positional gates and pooling to validated raw projections."""
        _, sequence_length, _, _ = raw_kv.shape
        input_dtype = raw_kv.dtype
        accumulation_dtype = self._accumulation_dtype(input_dtype)

        offsets = (
            torch.arange(sequence_length, device=raw_kv.device) % self.compression_ratio
        )
        positional_gate = self.ape.index_select(0, offsets).reshape(
            sequence_length,
            self.branches,
            self.kv_dim,
        )
        raw_gate = raw_gate.to(dtype=accumulation_dtype) + positional_gate.to(
            dtype=accumulation_dtype
        ).unsqueeze(0)

        compressed, compressed_valid = self.pool(
            raw_kv,
            raw_gate,
            state_valid,
        )
        compressed = compressed.to(dtype=input_dtype)

        compressed_length = compressed.shape[1]
        shared_kv = compressed.unsqueeze(1)

        positions = (
            torch.arange(
                compressed_length,
                device=raw_kv.device,
                dtype=torch.float32,
            )
            * self.compression_ratio
            + (self.compression_ratio - 1) / 2
        ).clamp_max(float(sequence_length - 1))
        return CompressedKV(
            key=shared_kv,
            value=shared_kv,
            state_valid=compressed_valid,
            positions=positions,
        )
