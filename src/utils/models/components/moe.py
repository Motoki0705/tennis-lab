from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from src.utils.models.components.ffn_layers import MLP, SwiGLU
from src.utils.models.components.ops.moe.api import resolve_moe_operations
from src.utils.models.components.ops.moe.reference import (
    MoEDispatchResult,
)

FFNType = Literal["swiglu", "mlp"]
DropPolicy = Literal["none", "capacity"]


@dataclass(frozen=True)
class MoERouting:
    """Routing outputs used by `MoELayer` and low-level MoE ops."""

    router_logits: torch.Tensor
    expert_weights: torch.Tensor
    expert_indices: torch.Tensor


@dataclass(frozen=True)
class MoEConfig:
    """Configuration for a token-choice MoE layer."""

    dim: int
    num_experts: int
    top_k: int
    ffn_dim: int
    ffn_type: FFNType
    router_bias: bool
    router_jitter_noise: float
    normalize_router_weights: bool
    capacity_factor: float | None
    drop_policy: DropPolicy
    use_cuda_ops: bool


class TopKRouter(nn.Module):
    """Top-k token router with FP32 softmax for stable routing weights."""

    def __init__(
        self,
        dim: int,
        num_experts: int,
        *,
        top_k: int,
        bias: bool,
        jitter_noise: float,
        normalize_router_weights: bool,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError(
                f"top_k must be in [1, num_experts], got top_k={top_k}, "
                f"num_experts={num_experts}"
            )
        if jitter_noise < 0:
            raise ValueError(f"jitter_noise must be non-negative, got {jitter_noise}")

        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = float(jitter_noise)
        self.gate = nn.Linear(dim, num_experts, bias=bias)
        self._prepare_tokens: Callable[[torch.Tensor], torch.Tensor]
        self._prepare_weights: Callable[[torch.Tensor], torch.Tensor]
        self._prepare_tokens = (
            self._apply_training_jitter if self.jitter_noise > 0.0 else self._identity
        )
        self._prepare_weights = (
            self._normalize_weights if normalize_router_weights else self._identity
        )

    def forward(self, hidden_states: torch.Tensor) -> MoERouting:
        tokens = self._prepare_tokens(hidden_states.reshape(-1, self.dim))

        router_logits = self.gate(tokens)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        expert_weights, expert_indices = torch.topk(
            router_probs,
            self.top_k,
            dim=-1,
            sorted=False,
        )
        expert_weights = self._prepare_weights(expert_weights)
        return MoERouting(
            router_logits=router_logits,
            expert_weights=expert_weights.to(dtype=hidden_states.dtype),
            expert_indices=expert_indices,
        )

    def _apply_training_jitter(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.training:
            noise = torch.empty_like(tokens).uniform_(
                1.0 - self.jitter_noise,
                1.0 + self.jitter_noise,
            )
            return tokens * noise
        return tokens

    @staticmethod
    def _normalize_weights(expert_weights: torch.Tensor) -> torch.Tensor:
        denominator = expert_weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(expert_weights.dtype).eps
        )
        return expert_weights / denominator

    @staticmethod
    def _identity(tensor: torch.Tensor) -> torch.Tensor:
        return tensor


class MoELayer(nn.Module):
    """Mixture-of-Experts FFN layer with pluggable dispatch/combine ops."""

    def __init__(self, cfg: MoEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.drop_policy not in ("none", "capacity"):
            raise ValueError(f"Unsupported drop_policy={cfg.drop_policy}")
        if cfg.capacity_factor is not None and cfg.capacity_factor <= 0.0:
            raise ValueError("capacity_factor must be positive when provided.")
        if cfg.drop_policy == "capacity" and cfg.capacity_factor is None:
            raise ValueError("capacity_factor is required when drop_policy='capacity'.")

        self.router = TopKRouter(
            dim=cfg.dim,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
            bias=cfg.router_bias,
            jitter_noise=cfg.router_jitter_noise,
            normalize_router_weights=cfg.normalize_router_weights,
        )
        self.experts = nn.ModuleList(
            self._make_expert(cfg.dim, cfg.ffn_dim, cfg.ffn_type)
            for _ in range(cfg.num_experts)
        )
        operations = resolve_moe_operations(
            backend="cuda" if cfg.use_cuda_ops else "reference",
            num_experts=cfg.num_experts,
            capacity_factor=cfg.capacity_factor,
            drop_policy=cfg.drop_policy,
        )
        self._dispatch = operations.dispatch
        self._combine = operations.combine

    @staticmethod
    def _make_expert(dim: int, ffn_dim: int, ffn_type: FFNType) -> nn.Module:
        if ffn_type == "swiglu":
            return SwiGLU(dim=dim, ffn_dim=ffn_dim)
        if ffn_type == "mlp":
            return MLP(dim=dim, ffn_dim=ffn_dim)
        raise ValueError(f"Unsupported ffn_type={ffn_type}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        tokens = hidden_states.reshape(-1, self.cfg.dim)
        routing = self.router(tokens)
        dispatch_result = self._dispatch(
            tokens,
            routing.expert_indices,
            routing.expert_weights,
        )
        expert_outputs = self._run_experts(dispatch_result)
        combined = self._combine(expert_outputs, dispatch_result)
        return combined.reshape(original_shape)

    def _run_experts(self, dispatch_result: MoEDispatchResult) -> torch.Tensor:
        expert_inputs = dispatch_result.expert_inputs
        expert_outputs = torch.empty_like(expert_inputs)
        for expert_idx, expert in enumerate(self.experts):
            expert_outputs[expert_idx] = expert(expert_inputs[expert_idx])
        return expert_outputs
