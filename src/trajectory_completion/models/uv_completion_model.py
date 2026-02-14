"""UV trajectory completion model with staged cross/self attention.

Architecture:
- Stage 1: Ball->Court cross-attention + Ball temporal self-attention
- Stage 2: Query->[processed ball tokens, raw ball tokens] cross-attention
           + Query temporal self-attention

Forward I/F is kept unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from src.common.models import (
    CrossAttnBlock,
    CrossAttnBlockConfig,
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
        num_query2ball_layers: int = 2,
        rope_dim: int | None = None,
        query_init_std: float = 0.02,
        num_court_tokens: int = NUM_COURT_KP,
        use_rope_stage1_cross: bool = False,
        use_rope_stage1_self: bool = True,
        use_rope_stage2_cross: bool = True,
        use_rope_stage2_self: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)
        self.causal = bool(causal)
        self.num_court_tokens = int(num_court_tokens)

        if self.hidden_dim % int(num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative.")
        if num_query2ball_layers < 0:
            raise ValueError("num_query2ball_layers must be non-negative.")

        head_dim = self.hidden_dim // int(num_heads)
        rope_dim_value = head_dim if rope_dim is None else int(rope_dim)
        if rope_dim_value % 2 != 0:
            raise ValueError("rope_dim must be even.")
        if rope_dim_value > head_dim:
            raise ValueError("rope_dim must be <= head_dim.")

        mlp_inter_dim_value = (
            int(mlp_inter_dim) if mlp_inter_dim is not None else int((8 * self.hidden_dim) / 3)
        )
        if use_moe and moe_config is None:
            raise ValueError("use_moe=True requires moe_config.")

        self.use_rope_stage1_cross = bool(use_rope_stage1_cross)
        self.use_rope_stage1_self = bool(use_rope_stage1_self)
        self.use_rope_stage2_cross = bool(use_rope_stage2_cross)
        self.use_rope_stage2_self = bool(use_rope_stage2_self)

        self.invisible_token = InvisibleTokenEmbedding(
            dim=self.hidden_dim,
            init_std=invisible_init_std,
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
        self.court_id_embed = nn.Embedding(self.num_court_tokens, self.hidden_dim)
        self.query_base = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * float(query_init_std))

        stage1_cross_rope_dim = rope_dim_value if self.use_rope_stage1_cross else 0
        stage1_self_rope_dim = rope_dim_value if self.use_rope_stage1_self else 0
        stage2_cross_rope_dim = rope_dim_value if self.use_rope_stage2_cross else 0
        stage2_self_rope_dim = rope_dim_value if self.use_rope_stage2_self else 0

        self.ball_to_court_cross_layers = nn.ModuleList(
            [
                CrossAttnBlock(
                    CrossAttnBlockConfig(
                        dim=self.hidden_dim,
                        n_heads=int(num_heads),
                        mlp_inter_dim=mlp_inter_dim_value,
                        head_dim=head_dim,
                        rope_dim=stage1_cross_rope_dim,
                        attn_dropout=float(dropout),
                        use_moe=bool(use_moe),
                        moe_config=moe_config,
                    )
                )
                for _ in range(int(num_layers))
            ]
        )
        self.ball_temporal_layers = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=self.hidden_dim,
                        n_heads=int(num_heads),
                        mlp_inter_dim=mlp_inter_dim_value,
                        head_dim=head_dim,
                        rope_dim=stage1_self_rope_dim,
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
        self.query_to_ball_cross_layers = nn.ModuleList(
            [
                CrossAttnBlock(
                    CrossAttnBlockConfig(
                        dim=self.hidden_dim,
                        n_heads=int(num_heads),
                        mlp_inter_dim=mlp_inter_dim_value,
                        head_dim=head_dim,
                        rope_dim=stage2_cross_rope_dim,
                        attn_dropout=float(dropout),
                        use_moe=bool(use_moe),
                        moe_config=moe_config,
                    )
                )
                for _ in range(int(num_query2ball_layers))
            ]
        )
        self.query_temporal_layers = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=self.hidden_dim,
                        n_heads=int(num_heads),
                        mlp_inter_dim=mlp_inter_dim_value,
                        head_dim=head_dim,
                        rope_dim=stage2_self_rope_dim,
                        attn_dropout=float(dropout),
                        rope_base=float(rope_theta),
                        yarn=yarn,
                        use_moe=bool(use_moe),
                        moe_config=moe_config,
                    )
                )
                for _ in range(int(num_query2ball_layers))
            ]
        )

        # Keep this alias for existing training code that references len(self.model.blocks).
        self.blocks = self.ball_temporal_layers

        self.final_norm = RMSNorm(self.hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 2),
        )

        freqs_ball_tokens = precompute_freqs_cis(
            dim=rope_dim_value,
            seqlen=self.max_seq_len,
            base=float(rope_theta),
            yarn=yarn,
            device=None,
        )
        freqs_ball_raw = precompute_freqs_cis(
            dim=rope_dim_value,
            seqlen=self.max_seq_len,
            base=float(rope_theta),
            yarn=yarn,
            device=None,
        )
        freqs_query = precompute_freqs_cis(
            dim=rope_dim_value,
            seqlen=self.max_seq_len,
            base=float(rope_theta),
            yarn=yarn,
            device=None,
        )
        self.register_buffer("freqs_ball_tokens", freqs_ball_tokens, persistent=False)
        self.register_buffer("freqs_ball_raw", freqs_ball_raw, persistent=False)
        self.register_buffer("freqs_query", freqs_query, persistent=False)

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
        mlp_inter_dim = model_cfg.get("mlp_inter_dim", model_cfg.get("ffn_dim"))
        mlp_inter_dim_value = int(mlp_inter_dim) if mlp_inter_dim is not None else None

        use_moe = bool(model_cfg.get("use_moe", False))
        moe_config = None
        if use_moe:
            moe_config = cls._build_moe_config(
                model_cfg,
                dim=hidden_dim,
                mlp_inter_dim=int(mlp_inter_dim_value or (8 * hidden_dim / 3)),
            )

        num_ball_layers = model_cfg.get("num_ball_layers", model_cfg.get("num_layers", 6))

        return cls(
            hidden_dim=hidden_dim,
            num_layers=int(num_ball_layers),
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
            num_query2ball_layers=int(model_cfg.get("num_query2ball_layers", 2)),
            rope_dim=model_cfg.get("rope_dim", None),
            query_init_std=float(model_cfg.get("query_init_std", 0.02)),
            num_court_tokens=int(model_cfg.get("num_court_tokens", NUM_COURT_KP)),
            use_rope_stage1_cross=bool(model_cfg.get("use_rope_stage1_cross", False)),
            use_rope_stage1_self=bool(model_cfg.get("use_rope_stage1_self", True)),
            use_rope_stage2_cross=bool(model_cfg.get("use_rope_stage2_cross", True)),
            use_rope_stage2_self=bool(model_cfg.get("use_rope_stage2_self", True)),
        )

    @staticmethod
    def _build_self_attn_mask(valid: Tensor) -> tuple[Tensor, Tensor]:
        valid_fixed = valid.bool()
        fully_masked = ~valid_fixed.any(dim=1)
        if fully_masked.any():
            valid_fixed = valid_fixed.clone()
            valid_fixed[fully_masked, 0] = True
        attn_mask = valid_fixed[:, None, :].expand(
            valid_fixed.shape[0], valid_fixed.shape[1], valid_fixed.shape[1]
        )
        return attn_mask, valid_fixed

    def forward(
        self,
        *,
        ball_uv_in: Tensor,
        ball_obs_mask: Tensor,
        court_kp: Tensor,
        court_vis: Tensor | None = None,
        seq_len: Tensor | None = None,
        ball_mask: Tensor | None = None,
        return_intermediate_ball_hidden: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Forward.

        Args:
            ball_uv_in: (B, T, 2) corrupted inputs (missing frames should be 0).
            ball_obs_mask: (B, T) observed mask (1=observed, 0=missing).
            court_kp: (B, 20, 2) court keypoints.
            court_vis: (B, 20) court visibility mask.
            seq_len: (B,) valid sequence lengths.
            ball_mask: (B, T) padding mask (1=valid).
            return_intermediate_ball_hidden: If True, also return stage-1 ball token
                hidden states after each temporal layer.

        Returns:
            Completed UV predictions: (B, T, 2). If
            ``return_intermediate_ball_hidden=True``, returns
            ``(pred, intermediate_ball_hidden_list)``.
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

        court_tok = self.court_embed(court_kp, court_vis)
        if court_tok.shape[1] != self.num_court_tokens:
            raise ValueError(
                f"Expected {self.num_court_tokens} court tokens, got {court_tok.shape[1]}"
            )
        ball_tok = self.ball_embed(ball_uv_in, ball_obs_mask)

        court_ids = torch.arange(self.num_court_tokens, device=ball_uv_in.device, dtype=torch.long)
        court_tokens = court_tok + self.court_id_embed(court_ids).unsqueeze(0)

        if ball_mask is None and seq_len is not None:
            t = torch.arange(T, device=ball_uv_in.device)[None, :]
            ball_mask = t < seq_len.to(torch.long).view(B, 1)

        ball_valid = torch.ones(B, T, device=ball_uv_in.device, dtype=torch.bool)
        if ball_mask is not None:
            ball_valid = ball_mask > 0

        ball_attn_mask, ball_valid = self._build_self_attn_mask(ball_valid)
        court_valid = torch.ones(B, self.num_court_tokens, device=ball_uv_in.device, dtype=torch.bool)

        freqs_ball_tokens = self.freqs_ball_tokens[:T]
        if freqs_ball_tokens.device != ball_uv_in.device:
            freqs_ball_tokens = freqs_ball_tokens.to(ball_uv_in.device)

        freqs_ball_raw = self.freqs_ball_raw[:T]
        if freqs_ball_raw.device != ball_uv_in.device:
            freqs_ball_raw = freqs_ball_raw.to(ball_uv_in.device)

        freqs_query = self.freqs_query[:T]
        if freqs_query.device != ball_uv_in.device:
            freqs_query = freqs_query.to(ball_uv_in.device)

        # Stage 1: ball tokens attend to court, then temporal self-attention.
        ball_tokens = ball_tok
        intermediate_ball_hidden: list[Tensor] = []
        for cross_layer, self_layer in zip(
            self.ball_to_court_cross_layers,
            self.ball_temporal_layers,
            strict=True,
        ):
            ball_tokens = cross_layer(
                ball_tokens,
                court_tokens,
                key_valid=court_valid,
                freqs_q_cis=freqs_ball_tokens if self.use_rope_stage1_cross else None,
                freqs_k_cis=None,
            )
            ball_tokens, _ = self_layer(
                ball_tokens,
                residual=None,
                start_pos=0,
                freqs_cis=freqs_ball_tokens if self.use_rope_stage1_self else None,
                attn_mask=ball_attn_mask,
                is_causal=self.causal,
            )
            if return_intermediate_ball_hidden:
                intermediate_ball_hidden.append(ball_tokens)

        # Preserve raw token memory to anchor observed-frame representation.
        ball_memory = torch.cat([ball_tokens, ball_tok], dim=1)
        ball_memory_valid = torch.cat([ball_valid, ball_valid], dim=1)

        # Independent RoPE streams for query and memory branches.
        freqs_ball_memory = torch.cat([freqs_ball_tokens, freqs_ball_raw], dim=0)

        # Stage 2: learned query attends to processed+raw memories, then temporal self-attention.
        query = self.query_base.expand(B, T, -1)
        for cross_layer, self_layer in zip(
            self.query_to_ball_cross_layers,
            self.query_temporal_layers,
            strict=True,
        ):
            query = cross_layer(
                query,
                ball_memory,
                key_valid=ball_memory_valid,
                freqs_q_cis=freqs_query if self.use_rope_stage2_cross else None,
                freqs_k_cis=freqs_ball_memory if self.use_rope_stage2_cross else None,
            )
            query, _ = self_layer(
                query,
                residual=None,
                start_pos=0,
                freqs_cis=freqs_query if self.use_rope_stage2_self else None,
                attn_mask=ball_attn_mask,
                is_causal=self.causal,
            )

        query = self.final_norm(query)
        pred = self.head(query)

        if return_intermediate_ball_hidden:
            return pred, intermediate_ball_hidden
        return pred


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
