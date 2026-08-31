"""Permutation-equivariant query competition for presence logits."""

from __future__ import annotations

from typing import Literal, TypeAlias

import torch
from torch import Tensor, nn

PresenceCompetitionMode: TypeAlias = Literal[
    "none",
    "deepsets",
    "deepsets_centered",
]


class DeepSetsPresenceResidual(nn.Module):
    """Predict a per-query residual from query-local and frame-pooled state."""

    def __init__(self, hidden_dim: int, center_queries: bool = False) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        self.hidden_dim = hidden_dim
        self.center_queries = center_queries
        self.feature_projection = nn.Linear(3 * hidden_dim, hidden_dim)
        self.output_projection = (
            nn.Linear(hidden_dim, 1, bias=False)
            if center_queries
            else nn.Linear(hidden_dim, 1)
        )
        self.reset_output_projection()

    def reset_output_projection(self) -> None:
        """Restore the exact zero-output residual used for legacy migration."""
        nn.init.zeros_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def forward(self, query_hidden: Tensor) -> Tensor:
        """Return ``(B,T,Q)`` residuals for ``query_hidden[B,T,Q,D]``."""
        if query_hidden.ndim != 4:
            raise ValueError("query_hidden must have shape (B,T,Q,D).")
        if query_hidden.shape[-1] != self.hidden_dim:
            raise ValueError(
                "query_hidden width must equal the configured hidden_dim."
            )
        if query_hidden.shape[-2] <= 0:
            raise ValueError("query_hidden must contain at least one query.")
        frame_mean = query_hidden.mean(dim=-2, keepdim=True)
        pooled = frame_mean.expand_as(query_hidden)
        features = torch.cat(
            (query_hidden, pooled, query_hidden - pooled),
            dim=-1,
        )
        hidden = torch.nn.functional.gelu(self.feature_projection(features))
        residual = self.output_projection(hidden).squeeze(-1)
        if self.center_queries:
            query_mean = residual.mean(dim=-1, keepdim=True)
            residual = residual - query_mean
        return residual


def build_presence_competition(
    mode: PresenceCompetitionMode,
    *,
    hidden_dim: int,
) -> DeepSetsPresenceResidual | None:
    """Build only the explicitly selected presence-competition branch."""
    if mode == "none":
        return None
    if mode == "deepsets":
        return DeepSetsPresenceResidual(hidden_dim)
    if mode == "deepsets_centered":
        return DeepSetsPresenceResidual(hidden_dim, center_queries=True)
    raise ValueError(f"Unsupported presence competition mode {mode!r}.")


def decode_presence_logits(
    presence_head: nn.Module,
    competition: DeepSetsPresenceResidual | None,
    query_hidden: Tensor,
    *,
    frame_valid: Tensor,
) -> Tensor:
    """Apply the optional residual and retain the fixed-query padding contract."""
    logits = presence_head(query_hidden).squeeze(-1)
    if competition is not None:
        logits = logits + competition(query_hidden)
    return logits * frame_valid[:, :, None]


__all__ = [
    "DeepSetsPresenceResidual",
    "PresenceCompetitionMode",
    "build_presence_competition",
    "decode_presence_logits",
]
