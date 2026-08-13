"""Order-invariant CourtKP7 fusion for object-conditioned observations."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SymmetricCourtPeakEncoder(nn.Module):
    """Encode peak geometry, score, covariance, and semantic class identity."""

    def __init__(self, hidden_dim: int, *, num_classes: int = 7) -> None:
        super().__init__()
        if hidden_dim <= 0 or num_classes != 7:
            raise ValueError("Court peak encoding requires hidden_dim>0 and 7 classes.")
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.geometry_projection = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.class_embedding = nn.Embedding(num_classes, hidden_dim)

    def forward(
        self,
        peak_uv: Tensor,
        peak_score: Tensor,
        peak_covariance: Tensor,
        peak_valid: Tensor,
        *,
        class_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return flattened set values ``[...,7*N,H]`` and validity."""
        if class_ids is None:
            class_ids = torch.arange(self.num_classes, device=peak_uv.device)
        geometry = torch.cat(
            (peak_uv, peak_score.unsqueeze(-1), peak_covariance.flatten(-2)),
            dim=-1,
        )
        encoded = self.geometry_projection(geometry)
        class_shape = (1,) * (encoded.ndim - 3) + (
            self.num_classes,
            1,
            self.hidden_dim,
        )
        encoded = encoded + self.class_embedding(class_ids).view(class_shape)
        encoded = encoded.masked_fill(~peak_valid.unsqueeze(-1), 0.0)
        return encoded.flatten(-3, -2), peak_valid.flatten(-2)


class CourtObjectSetFusion(nn.Module):
    """Fuse object queries with an unordered encoded court-peak set."""

    def __init__(self, hidden_dim: int, *, object_feature_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0 or object_feature_dim <= 0:
            raise ValueError("fusion dimensions must be positive.")
        self.hidden_dim = hidden_dim
        self.object_feature_dim = object_feature_dim
        self.object_projection = nn.Sequential(
            nn.Linear(object_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relative_attention_bias = nn.Sequential(
            nn.Linear(2, hidden_dim // 2 if hidden_dim >= 2 else 1),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2 if hidden_dim >= 2 else 1, 1),
        )
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        encoded_peaks: Tensor,
        peak_uv: Tensor,
        peak_valid: Tensor,
        object_anchor: Tensor,
        object_features: Tensor,
    ) -> Tensor:
        """Return one token per object with shape ``[B,V,T,D,H]``."""
        object_token = self.object_projection(object_features)
        query = self.query_projection(object_token)
        key = self.key_projection(encoded_peaks)
        value = self.value_projection(encoded_peaks)
        logits = torch.einsum("...dh,...kh->...dk", query, key) / math.sqrt(
            self.hidden_dim
        )
        relative_uv = peak_uv.unsqueeze(-3) - object_anchor.unsqueeze(-2)
        logits = logits + self.relative_attention_bias(relative_uv).squeeze(-1)
        valid = peak_valid.unsqueeze(-2)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        weights = weights.masked_fill(~valid, 0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        pooled = torch.einsum("...dk,...kh->...dh", weights, value)
        fused: Tensor = self.output_projection(
            torch.cat((object_token, pooled), dim=-1)
        )
        return fused


class ReferenceViewConditioning(nn.Module):
    """Add one learned value-stream delta to the selected view after set fusion."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        self.reference_delta = nn.Parameter(torch.randn(hidden_dim) * 0.02)

    def forward(self, tokens: Tensor, reference_view_mask: Tensor) -> Tensor:
        """Broadcast the reference role across time and detection axes."""
        delta = reference_view_mask[:, :, None, None, None].to(tokens.dtype)
        return tokens + delta * self.reference_delta.view(1, 1, 1, 1, -1)


__all__ = [
    "CourtObjectSetFusion",
    "ReferenceViewConditioning",
    "SymmetricCourtPeakEncoder",
]
