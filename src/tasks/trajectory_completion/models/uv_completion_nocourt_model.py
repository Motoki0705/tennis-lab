"""UV trajectory completion model without court inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from src.utils.models import (
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
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
        ffn_dim: int | None = None,
        ffn_type: str = "swiglu",
        invisible_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)

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
                        ffn_dim=ffn_dim,
                        head_dim=head_dim,
                        rope_dim=rope_dim_value,
                        attn_dropout=float(dropout),
                        rope_base=float(rope_theta),
                        ffn_type=ffn_type,
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
            device=None,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    @classmethod
    def from_config(cls, config: DictConfig) -> UVTrajectoryCompletionNoCourtModel:
        model_cfg = config.get("model", {}) or {}
        data_cfg = config.get("data", {}) or {}

        hidden_dim = int(model_cfg.get("hidden_dim", 256))

        return cls(
            hidden_dim=hidden_dim,
            num_layers=int(model_cfg.get("num_layers", 6)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            max_seq_len=int(model_cfg.get("max_seq_len", data_cfg.get("max_seq_len", 256))),
            rope_theta=float(model_cfg.get("rope_theta", 10000.0)),
            rope_dim=model_cfg.get("rope_dim", None),
            ffn_dim=model_cfg.get("ffn_dim"),
            ffn_type=str(model_cfg.get("ffn_type", "swiglu")),
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

        intermediate_ball_hidden: list[Tensor] = []
        for block in self.blocks:
            ball_tokens = block(
                ball_tokens,
                freqs_cis=freqs,
                attn_mask=ball_attn_mask,
            )
            if return_intermediate_ball_hidden:
                intermediate_ball_hidden.append(ball_tokens)

        h = self.final_norm(ball_tokens)

        pred = self.head(h)
        in_frame_logits = self.in_frame_head(h).squeeze(-1)

        if return_intermediate_ball_hidden and return_in_frame_logits:
            return pred, intermediate_ball_hidden, in_frame_logits
        if return_intermediate_ball_hidden:
            return pred, intermediate_ball_hidden
        if return_in_frame_logits:
            return pred, in_frame_logits
        return pred
