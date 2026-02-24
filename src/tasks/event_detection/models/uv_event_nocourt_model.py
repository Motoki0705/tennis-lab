"""UV-based event detection model without court tokens."""

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
from src.utils.models.embeddings import BallUVEmbedding, InvisibleTokenEmbedding
from src.tasks.event_detection.models.components.heads import EventLogitsHead

if TYPE_CHECKING:
    from omegaconf import DictConfig


class UVEventNoCourtModel(nn.Module):
    """Predict event logits from ball UV trajectory only."""

    uses_court_context = False

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        rope_theta: float = 10000.0,
        yarn: YaRNConfig | None = None,
        causal: bool = False,
        max_seq_len: int = 256,
        num_events: int = 2,
        mlp_inter_dim: int | None = None,
        use_moe: bool = False,
        moe_config: MoEConfig | None = None,
        invisible_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)
        self.num_events = int(num_events)
        self.causal = bool(causal)

        if self.hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        head_dim = self.hidden_dim // int(num_heads)
        rope_dim = head_dim
        mlp_inter_dim_value = (
            int(mlp_inter_dim) if mlp_inter_dim is not None else int((8 * self.hidden_dim) / 3)
        )

        if use_moe and moe_config is None:
            raise ValueError("use_moe=True requires moe_config.")

        self.invisible_token = InvisibleTokenEmbedding(
            dim=self.hidden_dim, init_std=invisible_init_std
        )
        self.ball_embed = BallUVEmbedding(
            dim=self.hidden_dim,
            dropout=dropout,
            invisible_token=self.invisible_token,
        )

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
            seqlen=self.max_seq_len,
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
    def from_config(cls, config: DictConfig) -> UVEventNoCourtModel:
        model_cfg = config.get("model", {}) or {}
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
            causal=bool(model_cfg.get("causal", False)),
            max_seq_len=int(model_cfg.get("max_seq_len", 256)),
            num_events=int(model_cfg.get("num_events", 2)),
            mlp_inter_dim=mlp_inter_dim_value,
            use_moe=use_moe,
            moe_config=moe_config,
            invisible_init_std=float(model_cfg.get("invisible_init_std", 0.02)),
        )

    def forward(
        self,
        ball_uv: Tensor,
        ball_vis: Tensor | None = None,
        ball_mask: Tensor | None = None,
        seq_len: Tensor | None = None,
    ) -> Tensor:
        """Forward.

        Args:
            ball_uv: (B, T, 2)
            ball_vis: (B, T) or None
            ball_mask: (B, T) or None
            seq_len: (B,) optional sequence lengths

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

        x = self.ball_embed(ball_uv, ball_vis)
        freqs_cis = self.freqs_cis[:T]
        if freqs_cis.device != x.device:
            freqs_cis = freqs_cis.to(x.device)

        attn_mask: Tensor | None = None
        if ball_mask is None and seq_len is not None:
            t = torch.arange(T, device=x.device)[None, :]
            ball_mask = t < seq_len.to(torch.long).view(B, 1)
        if ball_mask is not None:
            key_padding_mask = ball_mask > 0
            attn_mask = key_padding_mask[:, None, :].expand(B, T, T)

        residual = None
        for block in self.blocks:
            x, residual = block(
                x,
                residual,
                start_pos=0,
                freqs_cis=freqs_cis,
                attn_mask=attn_mask,
                is_causal=self.causal,
            )

        if residual is None:
            x = self.final_norm(x)
        else:
            x, _ = self.final_norm(x, residual)
        return self.head(x)

