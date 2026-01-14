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
moe.py (pure PyTorch)

- SwiGLU MLP (dense FFN)
- Top-k gated Mixture-of-Experts (MoE)

This is a "single-GPU / single-process" implementation:
- No distributed expert parallelism
- No custom kernels

Notes:
- This code favors clarity over peak performance.
- For large expert counts, you will likely want a more vectorized routing/expert dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class MoEConfig:
    dim: int
    inter_dim: int
    moe_inter_dim: int
    n_routed_experts: int = 64
    n_shared_experts: int = 0
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.0


class SwiGLU(nn.Module):
    """
    Standard SwiGLU MLP used in many LLMs:
        y = W2( silu(W1 x) * (W3 x) )
    """

    def __init__(self, dim: int, inter_dim: int, *, bias: bool = False) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=bias)
        self.w2 = nn.Linear(inter_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, inter_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    """
    Top-k gating, optionally with "group limiting" (as in DeepSeek code).

    Returns:
        weights: (N, topk)
        indices: (N, topk) expert ids
    """

    def __init__(self, cfg: MoEConfig) -> None:
        super().__init__()
        self.dim = int(cfg.dim)
        self.topk = int(cfg.n_activated_experts)
        self.n_groups = int(cfg.n_expert_groups)
        self.topk_groups = int(cfg.n_limited_groups)
        self.score_func = cfg.score_func
        self.route_scale = float(cfg.route_scale)

        self.weight = nn.Parameter(torch.empty(cfg.n_routed_experts, cfg.dim))
        nn.init.normal_(self.weight, std=0.02)

        # DeepSeek special-case bias at a specific dim; keep optional for compatibility.
        self.bias = nn.Parameter(torch.zeros(cfg.n_routed_experts, dtype=torch.float32)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (N, dim)
        scores = F.linear(x.float(), self.weight.float())  # (N, E)

        if self.score_func == "softmax":
            probs = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            probs = scores.sigmoid()
        else:
            raise ValueError(f"Unknown score_func={self.score_func}")

        original_probs = probs

        if self.bias is not None:
            probs = probs + self.bias  # broadcast

        if self.n_groups > 1:
            # reshape: (N, G, E/G)
            probs_g = probs.view(x.size(0), self.n_groups, -1)

            # group scoring follows DeepSeek's logic
            if self.bias is None:
                group_scores = probs_g.amax(dim=-1)
            else:
                group_scores = probs_g.topk(2, dim=-1).values.sum(dim=-1)

            chosen_groups = group_scores.topk(self.topk_groups, dim=-1).indices  # (N, topk_groups)
            mask = probs_g.new_ones((x.size(0), self.n_groups), dtype=torch.bool).scatter_(1, chosen_groups, False)
            probs_g = probs_g.masked_fill(mask.unsqueeze(-1), float("-inf"))
            probs = probs_g.flatten(1)  # back to (N, E)

        indices = probs.topk(self.topk, dim=-1).indices  # (N, topk)
        weights = original_probs.gather(1, indices)  # (N, topk)

        if self.score_func == "sigmoid":
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)

        weights = weights * self.route_scale
        return weights, indices


class Expert(nn.Module):
    """Per-expert SwiGLU MLP."""

    def __init__(self, dim: int, inter_dim: int, *, bias: bool = False) -> None:
        super().__init__()
        self.ffn = SwiGLU(dim, inter_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class MoE(nn.Module):
    """
    Single-process MoE.

    The forward follows DeepSeek's reference structure:
      - gate selects top-k experts per token
      - dispatch tokens to each expert (loop per expert)
      - weighted sum + shared experts (optional)
    """

    def __init__(self, cfg: MoEConfig, *, bias: bool = False) -> None:
        super().__init__()
        self.dim = int(cfg.dim)
        self.n_routed_experts = int(cfg.n_routed_experts)
        self.n_activated_experts = int(cfg.n_activated_experts)

        self.gate = Gate(cfg)
        self.experts = nn.ModuleList([Expert(self.dim, cfg.moe_inter_dim, bias=bias) for _ in range(self.n_routed_experts)])

        self.shared_experts: SwiGLU | None
        if cfg.n_shared_experts > 0:
            self.shared_experts = SwiGLU(self.dim, cfg.n_shared_experts * cfg.moe_inter_dim, bias=bias)
        else:
            self.shared_experts = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, dim) or (N, dim)

        Returns:
            same shape as x
        """
        orig_shape = x.shape
        if x.dim() == 3:
            x_flat = x.view(-1, self.dim)
        elif x.dim() == 2:
            x_flat = x
        else:
            raise ValueError(f"Expected x rank 2 or 3, got {x.dim()}")

        weights, indices = self.gate(x_flat)  # (N, topk), (N, topk)
        y = torch.zeros_like(x_flat, dtype=torch.float32)

        # Count tokens per expert (for short-circuiting)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)

        for expert_id in range(self.n_routed_experts):
            if counts[expert_id].item() == 0:
                continue
            expert = self.experts[expert_id]
            token_idx, top_slot = torch.where(indices == expert_id)  # both (M,)
            out = expert(x_flat[token_idx])  # (M, dim)
            y[token_idx] += out.float() * weights[token_idx, top_slot].float().unsqueeze(-1)

        if self.shared_experts is not None:
            y += self.shared_experts(x_flat).float()

        y = y.type_as(x_flat)
        return y.view(orig_shape)
