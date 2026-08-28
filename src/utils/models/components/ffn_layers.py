from __future__ import annotations

from abc import ABC, abstractmethod
from math import isfinite
from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

FFNType = Literal[
    "swiglu",
    "mlp",
    "kimi_k3_situglu",
    "deepseek_v4_swiglu",
    "gpt_oss_swiglu",
]


def default_ffn_dim(hidden_dim: int) -> int:
    """Default FFN width shared across transformer-style blocks."""
    ffn_dim = int((8 * hidden_dim) / 3)
    return (ffn_dim + 63) // 64 * 64


def _positive_finite(name: str, value: float) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return resolved


class _GatedFFN(nn.Module, ABC):
    """Projection layout shared by SwiGLU-family feed-forward networks."""

    def __init__(self, dim: int, ffn_dim: int, *, bias: bool = False) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, ffn_dim, bias=bias)
        self.w2 = nn.Linear(ffn_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, ffn_dim, bias=bias)

    @abstractmethod
    def _combine(self, gate: Tensor, up: Tensor) -> Tensor: ...

    def forward(self, x: Tensor) -> Tensor:
        hidden = self._combine(self.w1(x), self.w3(x))
        return cast(Tensor, self.w2(hidden))


class SwiGLU(_GatedFFN):
    """Standard SwiGLU: ``W2(silu(W1(x)) * W3(x))``."""

    def _combine(self, gate: Tensor, up: Tensor) -> Tensor:
        return F.silu(gate) * up


class KimiK3SiTUGLU(_GatedFFN):
    """Kimi K3 SiTU-GLU with smooth upper bounds on both input branches."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        *,
        beta_gate: float = 4.0,
        beta_up: float = 25.0,
        bias: bool = False,
    ) -> None:
        super().__init__(dim, ffn_dim, bias=bias)
        self.beta_gate = _positive_finite("beta_gate", beta_gate)
        self.beta_up = _positive_finite("beta_up", beta_up)

    def _combine(self, gate: Tensor, up: Tensor) -> Tensor:
        bounded_gate = self.beta_gate * torch.tanh(gate / self.beta_gate)
        bounded_up = self.beta_up * torch.tanh(up / self.beta_up)
        return bounded_gate * torch.sigmoid(gate) * bounded_up


class DeepSeekV4SwiGLU(_GatedFFN):
    """DeepSeek-V4 SwiGLU with asymmetric gate/up clamping."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        *,
        limit: float = 10.0,
        bias: bool = False,
    ) -> None:
        super().__init__(dim, ffn_dim, bias=bias)
        self.limit = _positive_finite("limit", limit)

    def _combine(self, gate: Tensor, up: Tensor) -> Tensor:
        bounded_gate = gate.clamp(max=self.limit)
        bounded_up = up.clamp(min=-self.limit, max=self.limit)
        return F.silu(bounded_gate) * bounded_up


class GPTOSSSwiGLU(_GatedFFN):
    """GPT-OSS clipped SwiGLU with a shifted linear branch."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        *,
        alpha: float = 1.702,
        limit: float = 7.0,
        bias: bool = False,
    ) -> None:
        super().__init__(dim, ffn_dim, bias=bias)
        self.alpha = _positive_finite("alpha", alpha)
        self.limit = _positive_finite("limit", limit)

    def _combine(self, gate: Tensor, up: Tensor) -> Tensor:
        bounded_gate = gate.clamp(max=self.limit)
        bounded_up = up.clamp(min=-self.limit, max=self.limit)
        activated_gate = bounded_gate * torch.sigmoid(self.alpha * bounded_gate)
        return activated_gate * (bounded_up + 1.0)


class MLP(nn.Module):
    """Standard 2-layer GELU MLP FFN."""

    def __init__(self, dim: int, ffn_dim: int, *, bias: bool = False) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, ffn_dim, bias=bias)
        self.fc2 = nn.Linear(ffn_dim, dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.fc2(F.gelu(self.fc1(x), approximate="tanh")))


def build_ffn(
    *,
    ffn_type: str,
    dim: int,
    ffn_dim: int,
    bias: bool = False,
) -> nn.Module:
    """Construct one supported dense FFN without silently changing variants."""
    if ffn_type == "swiglu":
        return SwiGLU(dim, ffn_dim, bias=bias)
    if ffn_type == "mlp":
        return MLP(dim, ffn_dim, bias=bias)
    if ffn_type == "kimi_k3_situglu":
        return KimiK3SiTUGLU(dim, ffn_dim, bias=bias)
    if ffn_type == "deepseek_v4_swiglu":
        return DeepSeekV4SwiGLU(dim, ffn_dim, bias=bias)
    if ffn_type == "gpt_oss_swiglu":
        return GPTOSSSwiGLU(dim, ffn_dim, bias=bias)
    raise ValueError(f"Unsupported ffn_type={ffn_type}")
