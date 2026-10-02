"""Computation-only temporal MDN; validation and decoding live in model_io."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from src.tasks.ball_refiner.refiner_2d.config import (
    CandidateAnchoredConfig,
    Refiner2DConfig,
)
from src.tasks.ball_refiner.refiner_2d.mean_anchors import candidate_mean_logits


class SetAttention(nn.Module):
    """Cross-attention with an always-valid null key, including empty sets."""

    def __init__(self, hidden_dim: int, heads: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.key_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.null = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(self, query: Tensor, tokens: Tensor, valid: Tensor) -> Tensor:
        b = query.shape[0]
        keys = torch.cat((self.null.expand(b, -1, -1), tokens), dim=1)
        mask = torch.cat((valid.new_ones((b, 1)), valid), dim=1)
        keys = self.key_norm(keys)
        attended, _ = self.attention(
            self.query_norm(query),
            keys,
            keys,
            key_padding_mask=~mask,
            need_weights=False,
        )
        return attended


class GatedContextAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int) -> None:
        super().__init__()
        self.cross_attention = SetAttention(hidden_dim, heads)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, query: Tensor, tokens: Tensor, valid: Tensor) -> Tensor:
        delta = self.cross_attention(query, tokens, valid)
        # The null key prevents all-masked softmax; missing context adds nothing.
        present = valid.any(dim=-1)[:, None, None]
        return query + torch.tanh(self.gate) * delta * present


class Refiner2DModel(nn.Module):
    """Independent batch rows, permutation-invariant candidates and people."""

    def __init__(self, config: Refiner2DConfig) -> None:
        super().__init__()
        d = config.hidden_dim
        self.config = config
        # Resolve the head schema once; forward only performs tensor computation.
        self.mean_anchor_config = config if isinstance(config, CandidateAnchoredConfig) else None
        self.candidate_encoder = nn.Sequential(
            nn.Linear(3 + 2 * config.patch_size**2, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.ball_query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.candidate_attention = SetAttention(d, config.attention_heads)
        self.pose_encoder = nn.Linear(3, d)
        self.joint_embedding = nn.Parameter(torch.randn(1, 1, 4, d) * 0.02)
        self.court_encoder = nn.Linear(3, d)
        self.court_embedding = nn.Parameter(
            torch.randn(1, config.court_keypoints, d) * 0.02
        )
        self.pose_context = GatedContextAttention(d, config.attention_heads)
        self.court_context = GatedContextAttention(d, config.attention_heads)
        self.register_buffer("time_frequencies", 2 * math.pi * torch.logspace(-1, 1, 8))
        self.time_encoder = nn.Linear(17, d)
        self.temporal_before = self._temporal_encoder(config)
        self.temporal_after = self._temporal_encoder(config)
        self.head_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, config.components * 6 + 1)
        initial_scale_logit = math.log(
            (config.initial_std - config.min_std)
            / (config.max_std - config.initial_std),
        )
        with torch.no_grad():
            self.head.bias.zero_()
            self.head.bias[:-1].view(config.components, 6)[:, 2:4] = initial_scale_logit
            if isinstance(config, CandidateAnchoredConfig):
                # Zero offsets preserve subpixel peaks at initialization; other
                # head rows and the trunk retain the same seed/initialization.
                self.head.weight[:-1].view(config.components, 6, d)[:config.anchored_components, :2].zero_()

    @staticmethod
    def _temporal_encoder(config: Refiner2DConfig) -> nn.TransformerEncoder:
        return nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                config.hidden_dim,
                config.attention_heads,
                dim_feedforward=4 * config.hidden_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=config.temporal_layers,
            enable_nested_tensor=False,
        )

    def forward(
        self,
        candidate_features: Tensor,
        candidate_valid: Tensor,
        pose_features: Tensor,
        pose_valid: Tensor,
        court_features: Tensor,
        court_valid: Tensor,
        relative_seconds: Tensor,
    ) -> Tensor:
        b, t, n, _ = candidate_features.shape
        d = self.config.hidden_dim
        candidates = self.candidate_encoder(candidate_features).reshape(b * t, n, d)
        query = self.ball_query.expand(b * t, -1, -1)
        h = query + self.candidate_attention(
            query, candidates, candidate_valid.reshape(b * t, n)
        )
        phase = relative_seconds.unsqueeze(-1) * self.time_frequencies
        time_features = torch.cat(
            (relative_seconds.unsqueeze(-1), phase.sin(), phase.cos()), dim=-1
        )
        h = self.temporal_before(h.reshape(b, t, d) + self.time_encoder(time_features))

        pose = self.pose_encoder(pose_features) + self.joint_embedding
        if self.training:
            keep_pose = (
                torch.rand((b, 1, 1, 1), device=pose.device) >= self.config.pose_dropout
            )
            pose_valid = pose_valid & keep_pose
        h = self.pose_context(
            h.reshape(b * t, 1, d),
            pose.reshape(b * t, -1, d),
            pose_valid.reshape(b * t, -1),
        )
        court = self.court_encoder(court_features) + self.court_embedding
        court = court[:, None].expand(-1, t, -1, -1).reshape(b * t, -1, d)
        court_mask = court_valid[:, None].expand(-1, t, -1).reshape(b * t, -1)
        h = self.court_context(h, court, court_mask)
        h = self.temporal_after(h.reshape(b, t, d))
        raw = self.head(self.head_norm(h))
        if self.mean_anchor_config is not None:
            values = raw[..., :-1].reshape(b, t, self.config.components, 6)
            means = candidate_mean_logits(values[..., :2], candidate_features, candidate_valid,
                                          count=self.mean_anchor_config.anchored_components, max_offset_uv=self.mean_anchor_config.max_offset_uv)
            return torch.cat((torch.cat((means, values[..., 2:].float()), dim=-1).flatten(-2), raw[..., -1:].float()), dim=-1)
        return raw
