"""Shared backbone for ball multi-task learning."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from src.ball_multitask.models.adapters.ball_3d_adapter import Ball3DTokenAdapter
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


class BallMultitaskBackbone(nn.Module):
    """Shared Transformer backbone for UV/3D ball tasks.

    Token layout: [REG*, COURT(20), BALL(T)]
    Returns per-frame ball token features.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        rope_dim: int | None = None,
        rope_theta: float = 10000.0,
        yarn: YaRNConfig | None = None,
        use_moe: bool = False,
        moe_config: MoEConfig | None = None,
        causal: bool = False,
        max_seq_len: int = 256,
        num_register_tokens: int = 0,
        invisible_init_std: float = 0.02,
    ) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if use_moe and moe_config is None:
            raise ValueError("use_moe=True requires moe_config.")

        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)
        self.num_register_tokens = int(num_register_tokens)
        self.causal = bool(causal)

        head_dim = int(hidden_dim // num_heads)
        self.rope_dim = int(rope_dim) if rope_dim is not None else head_dim
        self.rope_theta = float(rope_theta)
        self.yarn = yarn
        self.max_tokens = int(NUM_COURT_KP + self.max_seq_len)

        if ffn_dim is None:
            ffn_dim = int((8 * hidden_dim) / 3)
            ffn_dim = (ffn_dim + 63) // 64 * 64

        if moe_config is not None and moe_config.dim != hidden_dim:
            raise ValueError("moe_config.dim must match hidden_dim.")
        if self.num_register_tokens < 0:
            raise ValueError("num_register_tokens must be >= 0.")

        self.invisible_token = InvisibleTokenEmbedding(
            dim=self.hidden_dim, init_std=float(invisible_init_std)
        )
        self.court_embed = CourtKPUVEmbedding(
            dim=self.hidden_dim,
            dropout=float(dropout),
            invisible_token=self.invisible_token,
        )
        self.ball_uv_embed = BallUVEmbedding(
            dim=self.hidden_dim,
            dropout=float(dropout),
            invisible_token=self.invisible_token,
        )
        self.ball_3d_adapter = Ball3DTokenAdapter(
            dim=self.hidden_dim,
            dropout=float(dropout),
            invisible_token=self.invisible_token,
        )

        self.type_embed = nn.Embedding(2, self.hidden_dim)

        if self.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, self.num_register_tokens, self.hidden_dim)
            )
            nn.init.trunc_normal_(self.register_tokens, std=0.02)

        self.learned_court_tokens = nn.Parameter(
            torch.zeros(1, NUM_COURT_KP, self.hidden_dim)
        )
        nn.init.trunc_normal_(self.learned_court_tokens, std=0.02)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=self.hidden_dim,
                        n_heads=int(num_heads),
                        mlp_inter_dim=int(ffn_dim),
                        head_dim=head_dim,
                        rope_dim=self.rope_dim,
                        attn_dropout=float(dropout),
                        rope_base=self.rope_theta,
                        yarn=self.yarn,
                        use_moe=bool(use_moe),
                        moe_config=moe_config,
                    )
                )
                for _ in range(int(num_layers))
            ]
        )
        self.final_norm = RMSNorm(self.hidden_dim)

        freqs_cis = precompute_freqs_cis(
            dim=self.rope_dim,
            seqlen=self.max_tokens,
            base=self.rope_theta,
            yarn=self.yarn,
            device=None,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    @classmethod
    def from_config(cls, config: DictConfig) -> "BallMultitaskBackbone":
        """Create backbone from configuration."""
        model_cfg = config.get("model", {}) or {}
        yarn_cfg = model_cfg.get("yarn")
        yarn: YaRNConfig | None = None
        if yarn_cfg is not None:
            yarn_cfg = dict(yarn_cfg)
            if yarn_cfg.get("original_seq_len", None) is not None:
                yarn = YaRNConfig(**yarn_cfg)

        use_moe = bool(model_cfg.get("use_moe", False))
        moe_cfg = model_cfg.get("moe_config", None) or model_cfg.get("moe", None)
        moe_config: MoEConfig | None = None
        if use_moe and moe_cfg is not None:
            moe_config = MoEConfig(dim=int(model_cfg.get("hidden_dim", 256)), **dict(moe_cfg))

        return cls(
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            num_layers=int(model_cfg.get("num_layers", 6)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            ffn_dim=model_cfg.get("ffn_dim", None),
            dropout=float(model_cfg.get("dropout", 0.1)),
            rope_dim=model_cfg.get("rope_dim", None),
            rope_theta=float(model_cfg.get("rope_theta", 10000.0)),
            yarn=yarn,
            use_moe=use_moe,
            moe_config=moe_config,
            causal=bool(model_cfg.get("causal", False)),
            max_seq_len=int(model_cfg.get("max_seq_len", 256)),
            num_register_tokens=int(model_cfg.get("num_register_tokens", 0)),
            invisible_init_std=float(model_cfg.get("invisible_init_std", 0.02)),
        )

    def forward_uv(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        *,
        ball_vis: Tensor | None = None,
        ball_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
        seq_len: Tensor | None = None,
    ) -> Tensor:
        """Encode UV input into ball token features."""
        ball_uv, ball_vis, ball_mask, seq_len = self._clip_sequence(
            ball_uv, ball_vis, ball_mask, seq_len
        )
        B, T, _ = ball_uv.shape

        court_tokens = self.court_embed(court_kp, court_vis)
        ball_tokens = self.ball_uv_embed(ball_uv, ball_vis)
        return self._forward_tokens(
            ball_tokens=ball_tokens,
            court_tokens=court_tokens,
            ball_mask=ball_mask,
            seq_len=seq_len,
            batch_size=B,
            seq_len_t=T,
        )

    def forward_3d(
        self,
        ball_pos: Tensor,
        *,
        ball_vis: Tensor | None = None,
        ball_mask: Tensor | None = None,
        seq_len: Tensor | None = None,
    ) -> Tensor:
        """Encode 3D input into ball token features."""
        ball_pos, ball_vis, ball_mask, seq_len = self._clip_sequence(
            ball_pos, ball_vis, ball_mask, seq_len
        )
        B, T, _ = ball_pos.shape

        court_tokens = self.learned_court_tokens.expand(B, -1, -1)
        ball_tokens = self.ball_3d_adapter(ball_pos, ball_vis)
        return self._forward_tokens(
            ball_tokens=ball_tokens,
            court_tokens=court_tokens,
            ball_mask=ball_mask,
            seq_len=seq_len,
            batch_size=B,
            seq_len_t=T,
        )

    def _forward_tokens(
        self,
        *,
        ball_tokens: Tensor,
        court_tokens: Tensor,
        ball_mask: Tensor | None,
        seq_len: Tensor | None,
        batch_size: int,
        seq_len_t: int,
    ) -> Tensor:
        court_type = self.type_embed(
            torch.zeros(NUM_COURT_KP, device=ball_tokens.device, dtype=torch.long)
        )[None, :, :]
        ball_type = self.type_embed(
            torch.ones(seq_len_t, device=ball_tokens.device, dtype=torch.long)
        )[None, :, :]

        token_body = torch.cat(
            [court_tokens + court_type, ball_tokens + ball_type], dim=1
        )
        if self.num_register_tokens > 0:
            reg = self.register_tokens.expand(batch_size, -1, -1)
            x = torch.cat([reg, token_body], dim=1)
        else:
            x = token_body

        freqs_cis = self._build_freqs(token_body, x.device)
        attn_mask = self._build_attn_mask(
            batch_size=batch_size,
            seq_len_t=seq_len_t,
            ball_mask=ball_mask,
            seq_len=seq_len,
            device=x.device,
        )

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

        ball_start = self.num_register_tokens + NUM_COURT_KP
        return x[:, ball_start:, :]

    def _build_freqs(self, token_body: Tensor, device: torch.device) -> Tensor:
        S_body = token_body.shape[1]
        if S_body > self.freqs_cis.shape[0]:
            raise ValueError(
                "Sequence length exceeds cached freqs_cis length. Increase max_seq_len."
            )
        freqs_cis_body = self.freqs_cis[:S_body]
        if freqs_cis_body.device != device:
            freqs_cis_body = freqs_cis_body.to(device)
        if self.num_register_tokens <= 0:
            return freqs_cis_body

        prefix_freqs = torch.ones(
            self.num_register_tokens,
            freqs_cis_body.shape[1],
            device=device,
            dtype=freqs_cis_body.dtype,
        )
        return torch.cat([prefix_freqs, freqs_cis_body], dim=0)

    def _build_attn_mask(
        self,
        *,
        batch_size: int,
        seq_len_t: int,
        ball_mask: Tensor | None,
        seq_len: Tensor | None,
        device: torch.device,
    ) -> Tensor | None:
        if ball_mask is None and seq_len is not None:
            t = torch.arange(seq_len_t, device=device)[None, :]
            ball_mask = t < seq_len.to(torch.long).view(batch_size, 1)

        if ball_mask is None:
            return None

        court_valid = torch.ones(batch_size, NUM_COURT_KP, device=device, dtype=torch.bool)
        ball_valid = ball_mask > 0
        if self.num_register_tokens > 0:
            reg_valid = torch.ones(
                batch_size, self.num_register_tokens, device=device, dtype=torch.bool
            )
            key_padding_mask = torch.cat([reg_valid, court_valid, ball_valid], dim=1)
        else:
            key_padding_mask = torch.cat([court_valid, ball_valid], dim=1)

        S = key_padding_mask.shape[1]
        return key_padding_mask[:, None, :].expand(batch_size, S, S)

    def _clip_sequence(
        self,
        ball_seq: Tensor,
        ball_vis: Tensor | None,
        ball_mask: Tensor | None,
        seq_len: Tensor | None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
        if ball_seq.shape[1] <= self.max_seq_len:
            return ball_seq, ball_vis, ball_mask, seq_len

        ball_seq = ball_seq[:, : self.max_seq_len]
        if ball_vis is not None:
            ball_vis = ball_vis[:, : self.max_seq_len]
        if ball_mask is not None:
            ball_mask = ball_mask[:, : self.max_seq_len]
        if seq_len is not None:
            seq_len = torch.clamp(seq_len, max=self.max_seq_len)
        return ball_seq, ball_vis, ball_mask, seq_len


if __name__ == "__main__":
    torch.manual_seed(0)
    model = BallMultitaskBackbone(hidden_dim=64, num_layers=2, num_heads=4, max_seq_len=16)
    ball_uv = torch.randn(2, 8, 2)
    court_kp = torch.randn(2, NUM_COURT_KP, 2)
    out = model.forward_uv(ball_uv, court_kp)
    assert out.shape == (2, 8, 64)
    ball_pos = torch.randn(2, 8, 3)
    out3 = model.forward_3d(ball_pos)
    assert out3.shape == (2, 8, 64)
    print("backbone smoke ok")
