"""BLCS multi-view early-fusion model.

Per-camera fused tokens are built at each frame by concatenating:
    [ball_uv(2), court_kp_flat(20*2), court_vis(20)]
and applying an MLP embedding.

Then:
1) Replace per-camera tokens with an invisible token where ball is invisible.
2) Fuse tokens across cameras into one temporal token per frame.
3) Apply temporal Transformer blocks and regress 3D trajectory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from src.tasks.blcs.models.components.heads import Trajectory3DHead, VelocityHead
from src.utils.models import (
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    precompute_freqs_cis,
)
from src.utils.models.embeddings import InvisibleTokenEmbedding
from src.utils.schema.court import NUM_COURT_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BLCSMultiViewEarlyFusionModel(nn.Module):
    """BLCS multi-view model with early fusion before camera fusion."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        rope_dim: int | None = None,
        rope_theta: float = 10000.0,
        ffn_type: str = "swiglu",
        predict_velocity: bool = False,
        max_seq_len: int = 120,
        max_num_cameras: int = 8,
        invisible_init_std: float = 0.02,
        num_court_tokens: int = NUM_COURT_KP,
    ) -> None:
        """Initialize multi-view early-fusion BLCS model."""
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)
        self.max_num_cameras = int(max_num_cameras)
        self.predict_velocity = bool(predict_velocity)
        self.num_court_tokens = int(num_court_tokens)

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        if self.max_num_cameras <= 0:
            raise ValueError(
                f"max_num_cameras must be positive, got {self.max_num_cameras}"
            )
        head_dim = hidden_dim // num_heads

        rope_dim = head_dim if rope_dim is None else int(rope_dim)
        self.rope_dim = int(rope_dim)
        self.rope_theta = float(rope_theta)

        if ffn_dim is None:
            ffn_dim = int((8 * hidden_dim) / 3)
            ffn_dim = (ffn_dim + 63) // 64 * 64

        self.fusion_input_dim = int(2 + self.num_court_tokens * 2 + self.num_court_tokens)
        self.invisible_token = InvisibleTokenEmbedding(
            dim=hidden_dim,
            init_std=float(invisible_init_std),
        )
        self.fusion_embed = nn.Sequential(
            nn.Linear(self.fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(float(dropout)),
        )
        self.camera_fusion_mlp = nn.Sequential(
            nn.Linear(self.max_num_cameras * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(float(dropout)),
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=hidden_dim,
                        n_heads=num_heads,
                        ffn_dim=ffn_dim,
                        head_dim=head_dim,
                        rope_dim=self.rope_dim,
                        attn_dropout=dropout,
                        rope_base=self.rope_theta,
                        ffn_type=ffn_type,
                    )
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(hidden_dim)

        self.position_head = Trajectory3DHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            output_dim=3,
            num_layers=2,
            dropout=dropout,
        )
        self.velocity_head = None
        if predict_velocity:
            self.velocity_head = VelocityHead(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim // 2,
                output_dim=3,
                num_layers=2,
                dropout=dropout,
            )

        freqs_cis = precompute_freqs_cis(
            dim=self.rope_dim,
            seqlen=self.max_seq_len,
            base=self.rope_theta,
            device=None,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    @classmethod
    def from_config(cls, config: DictConfig) -> BLCSMultiViewEarlyFusionModel:
        """Create model from configuration."""
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})

        return cls(
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            num_layers=int(model_cfg.get("num_layers", 6)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            ffn_dim=model_cfg.get("ffn_dim", None),
            dropout=float(model_cfg.get("dropout", 0.1)),
            rope_dim=model_cfg.get("rope_dim", None),
            rope_theta=float(model_cfg.get("rope_theta", 10000.0)),
            ffn_type=str(model_cfg.get("ffn_type", "swiglu")),
            predict_velocity=bool(model_cfg.get("predict_velocity", False)),
            max_seq_len=int(model_cfg.get("max_seq_len", data_cfg.get("max_seq_len", 120))),
            max_num_cameras=int(model_cfg.get("max_num_cameras", model_cfg.get("max_views", 8))),
            invisible_init_std=float(model_cfg.get("invisible_init_std", 0.02)),
            num_court_tokens=int(data_cfg.get("num_court_kp", NUM_COURT_KP)),
        )

    def _flatten_court_kp_multiview(self, court_kp: Tensor, seq_len: int) -> Tensor:
        """Return court keypoints as (B, N, T, K*2)."""
        if court_kp.dim() == 4:
            court_kp = court_kp.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
        if court_kp.dim() != 5:
            raise ValueError(
                "court_kp must have shape (B, N, T, K, 2) or (B, N, K, 2), "
                f"got {tuple(court_kp.shape)}"
            )
        bsz, n_cam, seq_len_in, n_kp, _ = court_kp.shape
        return court_kp.reshape(bsz, n_cam, seq_len_in, n_kp * 2)

    def _prepare_court_vis_multiview(
        self,
        court_vis: Tensor | None,
        batch_size: int,
        n_cams: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Return court visibility as (B, N, T, K)."""
        n_kp = self.num_court_tokens
        if court_vis is None:
            return torch.ones(
                batch_size,
                n_cams,
                seq_len,
                n_kp,
                device=device,
                dtype=dtype,
            )
        if court_vis.dim() == 3:
            court_vis = court_vis.unsqueeze(2).expand(-1, -1, seq_len, -1)
        return court_vis.to(device=device, dtype=dtype)

    @staticmethod
    def _build_time_attn_mask(valid: Tensor) -> tuple[Tensor, Tensor]:
        """Build temporal self-attention mask from valid mask (B, T)."""
        valid_fixed = valid.bool()
        fully_masked = ~valid_fixed.any(dim=1)
        if fully_masked.any():
            valid_fixed = valid_fixed.clone()
            valid_fixed[fully_masked, 0] = True
        attn_mask = valid_fixed[:, None, :].expand(valid_fixed.shape[0], valid_fixed.shape[1], valid_fixed.shape[1])
        return attn_mask, valid_fixed

    def _fuse_across_cameras(self, tokens: Tensor, cam_valid: Tensor) -> tuple[Tensor, Tensor]:
        """Fuse camera tokens by MLP on flattened camera dimension.

        Args:
            tokens: (B, N, T, H)
            cam_valid: (B, N, T) bool

        Returns:
            tuple:
              - temporal_tokens: (B, T, H)
              - time_valid: (B, T) bool
        """
        batch_size, n_cams, seq_len, hidden_dim = tokens.shape
        valid = cam_valid.bool()
        time_valid = valid.any(dim=1)

        inv = self.invisible_token().to(dtype=tokens.dtype, device=tokens.device)
        inv = inv.view(1, 1, 1, hidden_dim)
        tokens = torch.where(valid.unsqueeze(-1), tokens, inv.expand_as(tokens))

        if n_cams < self.max_num_cameras:
            pad_n = self.max_num_cameras - n_cams
            pad_tokens = inv.expand(batch_size, pad_n, seq_len, hidden_dim)
            tokens = torch.cat([tokens, pad_tokens], dim=1)

        tokens_bt = tokens.permute(0, 2, 1, 3).reshape(
            batch_size * seq_len,
            self.max_num_cameras * hidden_dim,
        )
        fused_bt = self.camera_fusion_mlp(tokens_bt)
        temporal_tokens = fused_bt.reshape(batch_size, seq_len, hidden_dim)
        return temporal_tokens, time_valid

    def forward(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_vis: Tensor | None = None,
        ball_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            ball_uv: Ball 2D positions, shape (B, N, T, 2).
            court_kp: Court keypoints, shape (B, N, T, 20, 2) or (B, N, 20, 2).
            ball_vis: Ball visibility flags, shape (B, N, T). Optional.
            ball_mask: Ball validity mask, shape (B, N, T). Optional.
            court_vis: Court visibility mask, shape (B, N, T, 20) or (B, N, 20). Optional.
        """
        if ball_uv.dim() != 4:
            raise ValueError(f"ball_uv must have shape (B, N, T, 2), got {tuple(ball_uv.shape)}")

        batch_size, n_cams, seq_len, input_dim = ball_uv.shape
        if input_dim != 2:
            raise ValueError(f"ball_uv last dim must be 2, got {input_dim}")
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}. "
                "Increase model.max_seq_len."
            )
        if n_cams > self.max_num_cameras:
            raise ValueError(
                f"n_cams={n_cams} exceeds max_num_cameras={self.max_num_cameras}."
            )

        court_flat = self._flatten_court_kp_multiview(court_kp=court_kp, seq_len=seq_len)
        court_vis_feat = self._prepare_court_vis_multiview(
            court_vis=court_vis,
            batch_size=batch_size,
            n_cams=n_cams,
            seq_len=seq_len,
            device=ball_uv.device,
            dtype=ball_uv.dtype,
        )

        fused_input = torch.cat([ball_uv, court_flat, court_vis_feat], dim=-1)
        x = self.fusion_embed(fused_input)

        if ball_vis is not None:
            if ball_vis.shape != (batch_size, n_cams, seq_len):
                raise ValueError(
                    f"ball_vis must have shape {(batch_size, n_cams, seq_len)}, "
                    f"got {tuple(ball_vis.shape)}"
                )
            visible = (ball_vis > 0).unsqueeze(-1)
            inv = self.invisible_token().to(dtype=x.dtype, device=x.device)
            x = torch.where(visible, x, inv.view(1, 1, 1, -1).expand_as(x))

        if ball_mask is not None:
            if ball_mask.shape != (batch_size, n_cams, seq_len):
                raise ValueError(
                    f"ball_mask must have shape {(batch_size, n_cams, seq_len)}, "
                    f"got {tuple(ball_mask.shape)}"
                )
            cam_valid = (ball_mask > 0).bool()
        elif ball_vis is not None:
            cam_valid = (ball_vis > 0).bool()
        else:
            cam_valid = torch.ones(
                batch_size, n_cams, seq_len, device=ball_uv.device, dtype=torch.bool
            )

        x, time_valid = self._fuse_across_cameras(tokens=x, cam_valid=cam_valid)

        freqs_cis = self.freqs_cis[:seq_len]
        if freqs_cis.device != x.device:
            freqs_cis = freqs_cis.to(x.device)

        attn_mask, time_valid_fixed = self._build_time_attn_mask(time_valid)
        x = x * time_valid_fixed.unsqueeze(-1).to(dtype=x.dtype)

        for blk in self.blocks:
            x = blk(
                x,
                freqs_cis=freqs_cis,
                attn_mask=attn_mask,
            )

        x = self.final_norm(x)

        out: dict[str, Tensor] = {"position": self.position_head(x)}
        if self.predict_velocity and self.velocity_head is not None:
            out["velocity"] = self.velocity_head(x)
        return out

    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
