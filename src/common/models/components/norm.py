# ==============================================================================
# NOTE ON ORIGIN / LICENSE
#
# This file is derived from (and/or inspired by) DeepSeek's inference reference:
#   https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py
#
# MIT License
#
# Copyright (c) 2025 DeepSeek
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

"""
norm.py (pure PyTorch)

- RMSNorm with optional "fused residual" I/F:
    y = RMSNorm(x)                      if residual is None
    y, new_residual = RMSNorm(x, residual)  where new_residual = x + residual

- LayerNorm wrapper (float32 params, float32 compute) to mimic common LLM practice.
"""

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

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.dim, dtype=torch.float32))
            self.bias = nn.Parameter(torch.zeros(self.dim, dtype=torch.float32))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.float(),
            (self.dim,),
            self.weight,
            self.bias,
            self.eps,
        ).type_as(x)
