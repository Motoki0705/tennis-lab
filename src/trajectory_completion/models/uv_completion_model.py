"""UV trajectory completion model.

Architecture follows the BLCS style:
- Tokenize court keypoints + ball UV (with observation mask as a feature)
- Decoder-only Transformer blocks with RoPE
- Predict completed UV per frame with a small head
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from src.common.models import (
    MoEConfig,
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    YaRNConfig,
    precompute_freqs_cis,
)
from src.common.models.embeddings import BallUVEmbedding, CourtKPUVEmbedding, InvisibleTokenEmbedding
from src.utils.geometry import NUM_COURT_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig


class UVTrajectoryCompletionModel(nn.Module):
    """Complete ball UV trajectory from noisy/masked inputs and court keypoints."""

    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        rope_theta: float = 10000.0,
        yarn: YaRNConfig | None = None,
        causal: bool = False,
        max_seq_len: int = 256,
        mlp_inter_dim: int | None = None,
        use_moe: bool = False,
        moe_config: MoEConfig | None = None,
        invisible_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)
        self.causal = bool(causal)

        if self.hidden_dim % int(num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        head_dim = self.hidden_dim // int(num_heads)
        rope_dim = head_dim
        max_tokens = int(NUM_COURT_KP + self.max_seq_len)
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
            dropout=float(dropout),
            invisible_token=self.invisible_token,
        )
        self.ball_embed = BallUVEmbedding(
            dim=self.hidden_dim,
            dropout=float(dropout),
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
                        attn_dropout=float(dropout),
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
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 2),
        )

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
    def from_config(cls, config: DictConfig) -> UVTrajectoryCompletionModel:
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
            causal=bool(model_cfg.get("causal", False)),
            max_seq_len=int(model_cfg.get("max_seq_len", data_cfg.get("max_seq_len", 256))),
            mlp_inter_dim=mlp_inter_dim_value,
            use_moe=use_moe,
            moe_config=moe_config,
            invisible_init_std=float(model_cfg.get("invisible_init_std", 0.02)),
        )

    def forward(
        self,
        *,
        ball_uv_in: Tensor,
        ball_obs_mask: Tensor,
        court_kp: Tensor,
        court_vis: Tensor | None = None,
        seq_len: Tensor | None = None,
        ball_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward.

        Args:
            ball_uv_in: (B, T, 2) corrupted inputs (missing frames should be 0).
            ball_obs_mask: (B, T) observed mask (1=observed, 0=missing).
            court_kp: (B, 20, 2) court keypoints.
            court_vis: (B, 20) court visibility mask.
            seq_len: (B,) valid sequence lengths.
            ball_mask: (B, T) padding mask (1=valid).

        Returns:
            Completed UV predictions: (B, T, 2)
        """
        B, T, _ = ball_uv_in.shape
        if T > self.max_seq_len:
            ball_uv_in = ball_uv_in[:, : self.max_seq_len]
            ball_obs_mask = ball_obs_mask[:, : self.max_seq_len]
            if ball_mask is not None:
                ball_mask = ball_mask[:, : self.max_seq_len]
            if seq_len is not None:
                seq_len = torch.clamp(seq_len, max=self.max_seq_len)
            T = self.max_seq_len

        court_tok = self.court_embed(court_kp, court_vis)  # (B, 20, D)
        ball_tok = self.ball_embed(ball_uv_in, ball_obs_mask)  # (B, T, D)

        court_type = self.type_embed(
            torch.zeros(NUM_COURT_KP, device=ball_uv_in.device, dtype=torch.long)
        )[None, :, :]
        ball_type = self.type_embed(
            torch.ones(T, device=ball_uv_in.device, dtype=torch.long)
        )[None, :, :]

        x = torch.cat([court_tok + court_type, ball_tok + ball_type], dim=1)
        S = x.shape[1]

        key_padding_mask: Tensor | None = None
        if ball_mask is None and seq_len is not None:
            t = torch.arange(T, device=x.device)[None, :]
            ball_mask = t < seq_len.to(torch.long).view(B, 1)
        if ball_mask is not None:
            court_valid = torch.ones(B, NUM_COURT_KP, device=x.device, dtype=torch.bool)
            ball_valid = ball_mask > 0
            key_padding_mask = torch.cat([court_valid, ball_valid], dim=1)

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
                start_pos=0,
                freqs_cis=freqs_cis,
                attn_mask=attn_mask,
                is_causal=self.causal,
            )

        if residual is None:
            x = self.final_norm(x)
        else:
            x, _ = self.final_norm(x, residual)

        ball_h = x[:, NUM_COURT_KP:, :]
        return self.head(ball_h)


if __name__ == "__main__":
    model = UVTrajectoryCompletionModel(hidden_dim=64, num_layers=2, num_heads=4, max_seq_len=32)
    ball_uv_in = torch.rand(2, 32, 2)
    ball_obs_mask = torch.randint(0, 2, (2, 32)).float()
    court_kp = torch.rand(2, 20, 2)
    court_vis = torch.ones(2, 20)
    seq_len = torch.tensor([32, 24])
    out = model(
        ball_uv_in=ball_uv_in,
        ball_obs_mask=ball_obs_mask,
        court_kp=court_kp,
        court_vis=court_vis,
        seq_len=seq_len,
    )
    assert out.shape == (2, 32, 2)
    print("trajectory_completion.model smoke ok")
