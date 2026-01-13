"""MLP and Mixture of Experts (MoE) layers for Transformer models.

This module provides FFN implementations:
- SwiGLUMLP: Standard SwiGLU Feed-Forward Network
- MoELayer: Mixture of Experts with shared and routed experts

Reference:
    - SwiGLU: https://arxiv.org/abs/2002.05202
    - DeepSeekMoE: https://arxiv.org/abs/2401.06066
    - Mixtral: https://arxiv.org/abs/2401.04088
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SwiGLUMLP(nn.Module):
    """SwiGLU Feed-Forward Network.

    A gated linear unit with SiLU (Swish) activation, commonly used
    in modern LLM architectures like LLaMA.

    Can be used as both a standalone FFN and as an expert in MoE.

    Reference: https://arxiv.org/abs/2002.05202
    """

    def __init__(self, dim: int, ffn_dim: int, dropout: float = 0.0) -> None:
        """Initialize SwiGLU MLP.

        Args:
            dim: Input and output dimension.
            ffn_dim: Hidden dimension (intermediate size).
            dropout: Dropout probability.

        """
        super().__init__()
        self.wu = nn.Linear(dim, ffn_dim, bias=False)
        self.wg = nn.Linear(dim, ffn_dim, bias=False)
        self.wd = nn.Linear(ffn_dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Output tensor of shape (..., dim).

        """
        h = self.wu(x) * F.silu(self.wg(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.wd(h)


@dataclass
class MoEConfig:
    """Configuration for Mixture of Experts layer.

    Attributes:
        dim: Model dimension.
        ffn_dim: Hidden dimension per expert.
        num_experts: Total number of routed experts.
        num_shared_experts: Number of always-active shared experts.
        top_k: Number of experts to route each token to.
        dropout: Dropout probability.
        aux_loss_weight: Weight for auxiliary load balancing loss.
            Set to 0 for auxiliary-loss-free mode.
        use_bias_balancing: Use dynamic bias for load balancing.
            Recommended when aux_loss_weight=0.

    """

    dim: int
    ffn_dim: int
    num_experts: int = 8
    num_shared_experts: int = 1
    top_k: int = 2
    dropout: float = 0.0
    aux_loss_weight: float = 0.01
    use_bias_balancing: bool = False


class MoEGate(nn.Module):
    """Router/gate for MoE layer.

    Routes tokens to top-k experts based on learned routing scores.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        use_bias_balancing: bool = False,
    ) -> None:
        """Initialize gate.

        Args:
            dim: Input dimension.
            num_experts: Number of experts to route to.
            top_k: Number of experts per token.
            use_bias_balancing: Use dynamic bias for load balancing.

        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_bias_balancing = use_bias_balancing

        self.gate = nn.Linear(dim, num_experts, bias=False)

        if use_bias_balancing:
            self.expert_bias = nn.Parameter(torch.zeros(num_experts))
        else:
            self.register_parameter("expert_bias", None)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute routing weights.

        Args:
            x: Input tensor of shape (B, S, D).

        Returns:
            Tuple of:
                - weights: Routing weights of shape (B, S, top_k)
                - indices: Expert indices of shape (B, S, top_k)
                - gate_logits: Raw logits for aux loss, shape (B, S, E)

        """
        gate_logits = self.gate(x)

        if self.use_bias_balancing and self.expert_bias is not None:
            gate_logits_biased = gate_logits + self.expert_bias
        else:
            gate_logits_biased = gate_logits

        weights, indices = torch.topk(gate_logits_biased, self.top_k, dim=-1)
        original_weights = torch.gather(gate_logits, -1, indices)
        weights = F.softmax(original_weights, dim=-1)

        return weights, indices, gate_logits


class MoELayer(nn.Module):
    """Mixture of Experts layer with shared and routed experts.

    Architecture (DeepSeekMoE style):
        1. Shared experts: Always process all tokens
        2. Routed experts: Top-k selected per token
        3. Output = shared_output + routed_output

    Reference:
        - DeepSeekMoE: https://arxiv.org/abs/2401.06066
        - Mixtral: https://arxiv.org/abs/2401.04088
    """

    def __init__(self, cfg: MoEConfig) -> None:
        """Initialize MoE layer.

        Args:
            cfg: MoE configuration.

        """
        super().__init__()
        self.cfg = cfg
        self.dim = cfg.dim
        self.num_experts = cfg.num_experts
        self.num_shared_experts = cfg.num_shared_experts
        self.top_k = cfg.top_k
        self.dropout = cfg.dropout
        self.aux_loss_weight = cfg.aux_loss_weight

        # Shared experts (always active)
        if cfg.num_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [SwiGLUMLP(cfg.dim, cfg.ffn_dim) for _ in range(cfg.num_shared_experts)]
            )
        else:
            self.shared_experts = nn.ModuleList()

        # Routed experts
        self.experts = nn.ModuleList(
            [SwiGLUMLP(cfg.dim, cfg.ffn_dim) for _ in range(cfg.num_experts)]
        )

        # Router
        self.gate = MoEGate(
            dim=cfg.dim,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
            use_bias_balancing=cfg.use_bias_balancing,
        )

        # Auxiliary loss storage
        self._aux_loss: Optional[Tensor] = None

    def _compute_aux_loss(
        self,
        gate_logits: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute auxiliary load balancing loss."""
        if self.aux_loss_weight == 0:
            return torch.tensor(0.0, device=gate_logits.device)

        probs = F.softmax(gate_logits, dim=-1)

        if mask is not None:
            probs = probs * mask.unsqueeze(-1)
            num_tokens = mask.sum()
        else:
            num_tokens = probs.shape[0] * probs.shape[1]

        expert_probs = probs.sum(dim=(0, 1)) / (num_tokens + 1e-8)
        uniform = 1.0 / self.num_experts
        aux_loss = ((expert_probs - uniform) ** 2).sum() * self.num_experts

        return aux_loss * self.aux_loss_weight

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass through MoE layer.

        Args:
            x: Input tensor of shape (B, S, D).
            mask: Optional mask for valid positions, shape (B, S).

        Returns:
            Output tensor of shape (B, S, D).

        """
        B, S, D = x.shape

        # Compute shared expert output
        shared_out = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_out = shared_out + expert(x)
        if self.num_shared_experts > 0:
            shared_out = shared_out / self.num_shared_experts

        # Route to experts
        weights, indices, gate_logits = self.gate(x)

        # Compute auxiliary loss
        self._aux_loss = self._compute_aux_loss(gate_logits, mask)

        # Compute routed expert output
        x_flat = x.view(-1, D)
        weights_flat = weights.view(-1, self.top_k)
        indices_flat = indices.view(-1, self.top_k)

        routed_out = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            expert_idx = indices_flat[:, k]
            weight_k = weights_flat[:, k : k + 1]

            for e in range(self.num_experts):
                expert_mask = expert_idx == e
                if expert_mask.any():
                    expert_input = x_flat[expert_mask]
                    expert_output = self.experts[e](expert_input)
                    routed_out[expert_mask] += weight_k[expert_mask] * expert_output

        routed_out = routed_out.view(B, S, D)

        # Combine shared and routed outputs
        out = shared_out + routed_out
        out = F.dropout(out, p=self.dropout, training=self.training)

        return out

    def get_aux_loss(self) -> Tensor:
        """Get the auxiliary loss from the last forward pass."""
        if self._aux_loss is None:
            return torch.tensor(0.0)
        return self._aux_loss
