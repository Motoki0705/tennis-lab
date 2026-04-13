"""UV-based event detection model.

Architecture is aligned with src/tasks/blcs/models/blcs_model.py:
- Tokenize court keypoints + ball UV as tokens
- Decoder-only Transformer blocks with RoPE
- Predict per-frame event logits from ball tokens
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from src.utils.models import (
    MoEConfig,
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    YaRNConfig,
    precompute_freqs_cis,
)
from src.utils.models.embeddings import BallUVEmbedding, CourtKPUVEmbedding, InvisibleTokenEmbedding
from src.tasks.event_detection.models.components.heads import EventLogitsHead
from src.utils.schema.court import NUM_COURT_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig


class UVEventModel(nn.Module):
    """Predict event logits from ball UV trajectory and court keypoints."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        rope_theta: float = 10000.0,
        yarn: YaRNConfig | None = None,
        max_seq_len: int = 256,
        num_events: int = 2,
        mlp_inter_dim: int | None = None,
        use_moe: bool = False,
        moe_config: MoEConfig | None = None,
        invisible_init_std: float = 0.02,
        num_court_tokens: int = NUM_COURT_KP,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)
        self.num_events = int(num_events)
        self.num_court_tokens = int(num_court_tokens)

        if self.hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        head_dim = self.hidden_dim // int(num_heads)
        rope_dim = head_dim
        max_tokens = int(self.num_court_tokens + self.max_seq_len)
        mlp_inter_dim_value = (
            int(mlp_inter_dim) if mlp_inter_dim is not None else int((8 * self.hidden_dim) / 3)
        )

        if use_moe and moe_config is None:
            raise ValueError("use_moe=True requires moe_config.")

        self.invisible_token = InvisibleTokenEmbedding(
            dim=self.hidden_dim, init_std=invisible_init_std
        )
        self.court_embed = CourtKPUVEmbedding(
            dim=self.hidden_dim,
            dropout=dropout,
            invisible_token=self.invisible_token,
        )
        self.ball_embed = BallUVEmbedding(
            dim=self.hidden_dim,
            dropout=dropout,
            invisible_token=self.invisible_token,
        )
        self.type_embed = nn.Embedding(2, self.hidden_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=self.hidden_dim,
                        n_heads=int(num_heads),
                        mlp_inter_dim=mlp_inter_dim_value,
                        head_dim=head_dim,
                        rope_dim=rope_dim,
                        attn_dropout=dropout,
                        rope_base=float(rope_theta),
                        yarn=yarn,
                        use_moe=bool(use_moe),
                        moe_config=moe_config,
                    )
                )
                for _ in range(int(num_layers))
            ]
        )
        self.final_norm = RMSNorm(self.hidden_dim)
        self.head = EventLogitsHead(input_dim=self.hidden_dim, num_events=self.num_events, dropout=dropout)

        freqs_cis = precompute_freqs_cis(
            dim=rope_dim,
            seqlen=max_tokens,
            base=float(rope_theta),
            yarn=yarn,
            device=None,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    @staticmethod
    def _parse_yarn_config(model_cfg: DictConfig) -> YaRNConfig | None:
        yarn_cfg = model_cfg.get("yarn")
        if yarn_cfg is None:
            return None
        return YaRNConfig(
            original_seq_len=int(yarn_cfg.get("original_seq_len")),
            rope_factor=float(yarn_cfg.get("rope_factor")),
            beta_fast=int(yarn_cfg.get("beta_fast", 32)),
            beta_slow=int(yarn_cfg.get("beta_slow", 1)),
        )

    @staticmethod
    def _build_moe_config(model_cfg: DictConfig, *, dim: int, mlp_inter_dim: int) -> MoEConfig:
        moe_cfg = model_cfg.get("moe") or model_cfg.get("moe_config")
        if moe_cfg is None:
            return MoEConfig(dim=dim, moe_inter_dim=mlp_inter_dim)
        return MoEConfig(
            dim=int(moe_cfg.get("dim", dim)),
            moe_inter_dim=int(moe_cfg.get("moe_inter_dim", mlp_inter_dim)),
            n_routed_experts=int(moe_cfg.get("n_routed_experts", 64)),
            n_shared_experts=int(moe_cfg.get("n_shared_experts", 0)),
            n_activated_experts=int(moe_cfg.get("n_activated_experts", 6)),
            n_expert_groups=int(moe_cfg.get("n_expert_groups", 1)),
            n_limited_groups=int(moe_cfg.get("n_limited_groups", 1)),
            score_func=moe_cfg.get("score_func", "softmax"),
            route_scale=float(moe_cfg.get("route_scale", 1.0)),
        )

    @classmethod
    def from_config(cls, config: DictConfig) -> UVEventModel:
        model_cfg = config.get("model", {}) or {}
        data_cfg = config.get("data", {}) or {}
        hidden_dim = int(model_cfg.get("hidden_dim", 256))
        mlp_inter_dim = model_cfg.get("mlp_inter_dim")
        mlp_inter_dim_value = int(mlp_inter_dim) if mlp_inter_dim is not None else None
        use_moe = bool(model_cfg.get("use_moe", False))
        moe_config = None
        if use_moe:
            moe_config = cls._build_moe_config(
                model_cfg,
                dim=hidden_dim,
                mlp_inter_dim=int(mlp_inter_dim_value or (8 * hidden_dim / 3)),
            )
        return cls(
            hidden_dim=hidden_dim,
            num_layers=int(model_cfg.get("num_layers", 6)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            rope_theta=float(model_cfg.get("rope_theta", 10000.0)),
            yarn=cls._parse_yarn_config(model_cfg),
            max_seq_len=int(model_cfg.get("max_seq_len", 256)),
            num_events=int(model_cfg.get("num_events", 2)),
            mlp_inter_dim=mlp_inter_dim_value,
            use_moe=use_moe,
            moe_config=moe_config,
            invisible_init_std=float(model_cfg.get("invisible_init_std", 0.02)),
            num_court_tokens=int(data_cfg.get("num_court_kp", NUM_COURT_KP)),
        )

    def forward(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_vis: Tensor | None = None,
        ball_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
        seq_len: Tensor | None = None,
    ) -> Tensor:
        """Forward.

        Args:
            ball_uv: (B, T, 2)
            court_kp: (B, K, 2)
            ball_vis: (B, T) or None
            ball_mask: (B, T) or None
            court_vis: (B, K) or None

        Returns:
            Logits (B, T, E)
        """
        B, T, _ = ball_uv.shape
        if T > self.max_seq_len:
            ball_uv = ball_uv[:, : self.max_seq_len]
            if ball_vis is not None:
                ball_vis = ball_vis[:, : self.max_seq_len]
            if ball_mask is not None:
                ball_mask = ball_mask[:, : self.max_seq_len]
            if seq_len is not None:
                seq_len = torch.clamp(seq_len, max=self.max_seq_len)
            T = self.max_seq_len

        court_tokens = self.court_embed(court_kp, court_vis)  # (B, K, D)
        if court_tokens.shape[1] != self.num_court_tokens:
            raise ValueError(
                f"Expected {self.num_court_tokens} court tokens, got {court_tokens.shape[1]}"
            )
        ball_tokens = self.ball_embed(ball_uv, ball_vis)  # (B, T, D)

        K = self.num_court_tokens
        court_type = self.type_embed(
            torch.zeros(K, device=ball_uv.device, dtype=torch.long)
        )[None, :, :]
        ball_type = self.type_embed(
            torch.ones(T, device=ball_uv.device, dtype=torch.long)
        )[None, :, :]

        x = torch.cat([court_tokens + court_type, ball_tokens + ball_type], dim=1)  # (B, S, D)
        S = x.shape[1]

        key_padding_mask: Tensor | None = None
        if ball_mask is None and seq_len is not None:
            t = torch.arange(T, device=x.device)[None, :]
            ball_mask = t < seq_len.to(torch.long).view(B, 1)
        if ball_mask is not None:
            court_valid = torch.ones(B, K, device=x.device, dtype=torch.bool)
            ball_valid = ball_mask > 0
            key_padding_mask = torch.cat([court_valid, ball_valid], dim=1)  # (B, S)

        if S > self.freqs_cis.shape[0]:
            raise ValueError("Sequence length exceeds max_seq_len; increase model.max_seq_len.")
        freqs_cis = self.freqs_cis[:S]
        if freqs_cis.device != x.device:
            freqs_cis = freqs_cis.to(x.device)

        attn_mask: Tensor | None = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, :].expand(B, S, S)

        residual = None
        for block in self.blocks:
            x, residual = block(
                x,
                residual,
                freqs_cis=freqs_cis,
                attn_mask=attn_mask,
            )

        if residual is None:
            x = self.final_norm(x)
        else:
            x, _ = self.final_norm(x, residual)

        ball_h = x[:, K:, :]
        return self.head(ball_h)


if __name__ == "__main__":
    num_court_tokens = 12
    model = UVEventModel(
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        max_seq_len=32,
        num_events=2,
        num_court_tokens=num_court_tokens,
    )
    ball_uv = torch.rand(2, 32, 2)
    ball_vis = torch.ones(2, 32)
    ball_mask = torch.ones(2, 32)
    court_kp = torch.rand(2, num_court_tokens, 2)
    court_vis = torch.ones(2, num_court_tokens)
    seq_len = torch.tensor([32, 16])
    logits = model(ball_uv, court_kp, ball_vis=ball_vis, ball_mask=ball_mask, court_vis=court_vis, seq_len=seq_len)
    assert logits.shape == (2, 32, 2)
    print("uv model smoke ok")
