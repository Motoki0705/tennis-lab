"""UV trajectory completion model without court inputs."""

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

if TYPE_CHECKING:
    from omegaconf import DictConfig


class UVTrajectoryCompletionNoCourtModel(nn.Module):
    """Complete ball UV trajectory from noisy/masked inputs without court keypoints."""

    uses_court_context = False

    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        rope_theta: float = 10000.0,
        rope_dim: int | None = None,
        yarn: YaRNConfig | None = None,
        causal: bool = False,
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
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative.")

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

        self.invisible_token = InvisibleTokenEmbedding(
            dim=self.hidden_dim,
            init_std=invisible_init_std,
        )
        self.ball_embed = BallUVEmbedding(
            dim=self.hidden_dim,
            dropout=float(dropout),
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
                        rope_dim=rope_dim_value,
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
        self.in_frame_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )

        freqs_cis = precompute_freqs_cis(
            dim=rope_dim_value,
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
    def from_config(cls, config: DictConfig) -> UVTrajectoryCompletionNoCourtModel:
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

        return cls(
            hidden_dim=hidden_dim,
            num_layers=int(model_cfg.get("num_layers", 6)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            max_seq_len=int(model_cfg.get("max_seq_len", data_cfg.get("max_seq_len", 256))),
            rope_theta=float(model_cfg.get("rope_theta", 10000.0)),
            rope_dim=model_cfg.get("rope_dim", None),
            yarn=cls._parse_yarn_config(model_cfg),
            causal=bool(model_cfg.get("causal", False)),
            mlp_inter_dim=mlp_inter_dim_value,
            use_moe=use_moe,
            moe_config=moe_config,
            invisible_init_std=float(model_cfg.get("invisible_init_std", 0.02)),
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
        ball_uv: Tensor,
        ball_vis: Tensor | None = None,
        ball_mask: Tensor | None = None,
        *,
        return_intermediate_ball_hidden: bool = False,
        return_in_frame_logits: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]] | tuple[Tensor, Tensor] | tuple[Tensor, list[Tensor], Tensor]:
        """Forward pass for no-court trajectory completion."""
        _, T, _ = ball_uv.shape
        if T > self.max_seq_len:
            ball_uv = ball_uv[:, : self.max_seq_len]
            if ball_vis is not None:
                ball_vis = ball_vis[:, : self.max_seq_len]
            if ball_mask is not None:
                ball_mask = ball_mask[:, : self.max_seq_len]
            T = self.max_seq_len

        ball_tokens = self.ball_embed(ball_uv, ball_vis)

        ball_valid = torch.ones(ball_uv.shape[0], T, device=ball_uv.device, dtype=torch.bool)
        if ball_mask is not None:
            ball_valid = ball_mask > 0
        ball_attn_mask, _ = self._build_self_attn_mask(ball_valid)

        freqs = self.freqs_cis[:T]
        if freqs.device != ball_uv.device:
            freqs = freqs.to(ball_uv.device)

        residual = None
        intermediate_ball_hidden: list[Tensor] = []
        for block in self.blocks:
            ball_tokens, residual = block(
                ball_tokens,
                residual,
                start_pos=0,
                freqs_cis=freqs,
                attn_mask=ball_attn_mask,
                is_causal=self.causal,
            )
            if return_intermediate_ball_hidden:
                intermediate_ball_hidden.append(ball_tokens)

        if residual is None:
            h = self.final_norm(ball_tokens)
        else:
            h, _ = self.final_norm(ball_tokens, residual)

        pred = self.head(h)
        in_frame_logits = self.in_frame_head(h).squeeze(-1)

        if return_intermediate_ball_hidden and return_in_frame_logits:
            return pred, intermediate_ball_hidden, in_frame_logits
        if return_intermediate_ball_hidden:
            return pred, intermediate_ball_hidden
        if return_in_frame_logits:
            return pred, in_frame_logits
        return pred

