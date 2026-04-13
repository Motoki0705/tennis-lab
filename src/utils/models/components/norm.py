from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square LayerNorm (RMSNorm).

    If residual is provided, returns (normed, residual_sum) for "fused" residual patterns.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None):
        dtype = x.dtype
        if residual is None:
            x_f = x.float()
            var = x_f.pow(2).mean(dim=-1, keepdim=True)
            x_f = x_f * torch.rsqrt(var + self.eps)
            return (self.weight * x_f).to(dtype)
        else:
            # Maintain deepseek-style ordering: residual_sum is computed in fp32.
            residual_sum = x.float() + residual.float()
            var = residual_sum.pow(2).mean(dim=-1, keepdim=True)
            normed = residual_sum * torch.rsqrt(var + self.eps)
            return (self.weight * normed).to(dtype), residual_sum.to(dtype)


class LayerNorm(nn.Module):
    """
    LayerNorm wrapper that keeps parameters in fp32 and performs normalization in fp32.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(self.dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.float(),
            (self.dim,),
            self.weight,
            self.bias,
            self.eps,
        ).type_as(x)
