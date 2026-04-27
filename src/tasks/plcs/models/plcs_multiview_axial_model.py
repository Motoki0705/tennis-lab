"""Axial multi-view PLCS model with one token per camera/time element."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import torch
import torch.nn as nn
from torch import Tensor

from src.tasks.plcs.models.components.heads import PositionHead, RotationHead
from src.utils.models import (
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    precompute_freqs_cis_nd,
)
from src.utils.models.embeddings import CourtPlayerGroupEmbedding, InvisibleTokenEmbedding
from src.utils.schema.court import NUM_COURT_KP
from src.utils.schema.player import NUM_HUMAN_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig


class PLCSMultiViewAxialModel(nn.Module):
    """PLCS multiview model with alternating camera/time self-attention."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        rope_dim: int | None = None,
        rope_theta: float = 10000.0,
        rope_theta_time: float | None = None,
        rope_theta_camera: float | None = None,
        ffn_type: Literal["swiglu", "mlp"] = "swiglu",
        max_views: int = 8,
        max_seq_len: int = 120,
        invisible_init_std: float = 0.02,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.max_views = int(max_views)
        self.max_seq_len = int(max_seq_len)

        if self.hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} must be divisible by num_heads={num_heads}"
            )
        if num_layers < 0:
            raise ValueError(f"num_layers must be non-negative, got {num_layers}")
        if self.max_views <= 0:
            raise ValueError(f"max_views must be positive, got {max_views}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

        head_dim = self.hidden_dim // num_heads
        rope_dim = head_dim if rope_dim is None else int(rope_dim)
        if rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be even, got {rope_dim}")
        if rope_dim > head_dim:
            raise ValueError(f"rope_dim={rope_dim} cannot exceed head_dim={head_dim}")

        self.rope_dim = int(rope_dim)
        self.rope_bases = (
            float(rope_theta if rope_theta_time is None else rope_theta_time),
            float(rope_theta if rope_theta_camera is None else rope_theta_camera),
        )

        if ffn_dim is None:
            ffn_dim = int((8 * self.hidden_dim) / 3)
            ffn_dim = (ffn_dim + 63) // 64 * 64

        self.invisible_token = InvisibleTokenEmbedding(
            dim=self.hidden_dim,
            init_std=invisible_init_std,
        )
        self.group_embed = CourtPlayerGroupEmbedding(
            dim=self.hidden_dim,
            invisible_token=self.invisible_token,
        )

        self.camera_layers = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=self.hidden_dim,
                        n_heads=num_heads,
                        ffn_dim=ffn_dim,
                        head_dim=head_dim,
                        rope_dim=self.rope_dim,
                        attn_dropout=dropout,
                        rope_base=self.rope_bases[1],
                        ffn_type=ffn_type,
                    )
                )
                for _ in range(num_layers)
            ]
        )
        self.time_layers = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=self.hidden_dim,
                        n_heads=num_heads,
                        ffn_dim=ffn_dim,
                        head_dim=head_dim,
                        rope_dim=self.rope_dim,
                        attn_dropout=dropout,
                        rope_base=self.rope_bases[0],
                        ffn_type=ffn_type,
                    )
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm(self.hidden_dim)

        self.position_head = PositionHead(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim // 2,
            output_dim=3,
            num_layers=2,
            dropout=dropout,
        )
        self.rotation_head = RotationHead(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim // 2,
            num_layers=2,
            dropout=dropout,
        )

        token_freqs = precompute_freqs_cis_nd(
            dim=self.rope_dim,
            pos=self._build_token_positions(
                seq_len=self.max_seq_len,
                n_cams=self.max_views,
            ),
            base=self.rope_bases,
        )
        self.register_buffer("token_freqs_cis", token_freqs, persistent=False)

    @classmethod
    def from_config(cls, config: DictConfig) -> PLCSMultiViewAxialModel:
        """Create model from hydra config."""
        model_cfg = config.get("model", {})

        return cls(
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            num_layers=int(model_cfg.get("num_layers", 6)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            ffn_dim=model_cfg.get("ffn_dim", None),
            dropout=float(model_cfg.get("dropout", 0.1)),
            rope_dim=model_cfg.get("rope_dim", None),
            rope_theta=float(model_cfg.get("rope_theta", 10000.0)),
            rope_theta_time=model_cfg.get("rope_theta_time", None),
            rope_theta_camera=model_cfg.get("rope_theta_camera", None),
            ffn_type=cast(Literal["swiglu", "mlp"], str(model_cfg.get("ffn_type", "swiglu"))),
            max_views=int(model_cfg.get("max_views", 8)),
            max_seq_len=int(model_cfg.get("max_seq_len", 120)),
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
            valid_fixed.shape[0],
            valid_fixed.shape[1],
            valid_fixed.shape[1],
        )
        return attn_mask, valid_fixed

    @staticmethod
    def _build_token_positions(*, seq_len: int, n_cams: int) -> Tensor:
        time_idx = torch.arange(seq_len, dtype=torch.long) + 1
        camera_idx = torch.arange(n_cams, dtype=torch.long)
        return torch.stack(
            [
                time_idx[:, None].expand(seq_len, n_cams),
                camera_idx[None, :].expand(seq_len, n_cams),
            ],
            dim=-1,
        )

    def _camera_freqs(self, *, batch_size: int, seq_len: int, n_cams: int) -> Tensor:
        freqs = self.token_freqs_cis[:seq_len, :n_cams]
        return freqs.unsqueeze(0).expand(
            batch_size,
            seq_len,
            n_cams,
            self.rope_dim // 2,
        ).reshape(batch_size * seq_len, n_cams, self.rope_dim // 2)

    def _time_freqs(self, *, batch_size: int, seq_len: int, n_cams: int) -> Tensor:
        freqs = self.token_freqs_cis[:seq_len, :n_cams].permute(1, 0, 2)
        return freqs.unsqueeze(0).expand(
            batch_size,
            n_cams,
            seq_len,
            self.rope_dim // 2,
        ).reshape(batch_size * n_cams, seq_len, self.rope_dim // 2)

    @staticmethod
    def _masked_camera_mean(x: Tensor, valid: Tensor) -> Tensor:
        weights = valid.unsqueeze(-1).to(dtype=x.dtype)
        summed = (x * weights).sum(dim=2)
        counts = weights.sum(dim=2).clamp_min(1.0)
        pooled = summed / counts
        time_valid = valid.any(dim=2, keepdim=True)
        return pooled * time_valid.to(dtype=x.dtype)

    def forward(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None = None,
        human_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass for multiview PLCS inputs."""
        if human_kp.dim() != 5:
            raise ValueError(
                "PLCSMultiViewAxialModel expects human_kp as (B,N,T,17,2), "
                f"got shape {tuple(human_kp.shape)}"
            )
        if court_kp.dim() != 5:
            raise ValueError(
                "PLCSMultiViewAxialModel expects court_kp as (B,N,T,20,2), "
                f"got shape {tuple(court_kp.shape)}"
            )
        if human_vis is not None and human_vis.dim() != 4:
            raise ValueError(
                "PLCSMultiViewAxialModel expects human_vis as (B,N,T,17), "
                f"got shape {tuple(human_vis.shape)}"
            )
        if court_vis is not None and court_vis.dim() != 4:
            raise ValueError(
                "PLCSMultiViewAxialModel expects court_vis as (B,N,T,20), "
                f"got shape {tuple(court_vis.shape)}"
            )

        batch_size, n_cams, seq_len_in = human_kp.shape[:3]
        if n_cams > self.max_views:
            raise ValueError(f"Number of views N={n_cams} exceeds max_views={self.max_views}.")
        if seq_len_in > self.max_seq_len:
            raise ValueError(
                f"Sequence length T={seq_len_in} exceeds max_seq_len={self.max_seq_len}."
            )

        if human_vis is not None:
            human_kp = human_kp * (human_vis > 0).unsqueeze(-1).to(dtype=human_kp.dtype)
        if court_vis is not None:
            court_kp = court_kp * (court_vis > 0).unsqueeze(-1).to(dtype=court_kp.dtype)

        if human_mask is not None:
            if human_mask.dim() != 3 or human_mask.shape != (batch_size, n_cams, seq_len_in):
                raise ValueError(
                    "human_mask for multiview models must be (B,N,T), "
                    f"got {tuple(human_mask.shape)}"
                )
            token_valid = human_mask > 0
        else:
            token_valid = torch.ones(
                batch_size,
                n_cams,
                seq_len_in,
                dtype=torch.bool,
                device=human_kp.device,
            )

        court_flat = court_kp.reshape(batch_size * n_cams * seq_len_in, NUM_COURT_KP, 2)
        human_flat = human_kp.reshape(batch_size * n_cams * seq_len_in, NUM_HUMAN_KP, 2)
        group_vis = token_valid.reshape(batch_size * n_cams * seq_len_in)
        x = self.group_embed(court_flat, human_flat, group_vis).reshape(
            batch_size,
            n_cams,
            seq_len_in,
            self.hidden_dim,
        ).permute(0, 2, 1, 3)

        token_valid_t = token_valid.permute(0, 2, 1)
        x = x * token_valid_t.unsqueeze(-1).to(dtype=x.dtype)

        camera_valid = token_valid_t.reshape(batch_size * seq_len_in, n_cams)
        time_valid = token_valid_t.permute(0, 2, 1).reshape(batch_size * n_cams, seq_len_in)
        camera_mask, _ = self._build_self_attn_mask(camera_valid)
        time_mask, _ = self._build_self_attn_mask(time_valid)
        camera_freqs = self._camera_freqs(
            batch_size=batch_size,
            seq_len=seq_len_in,
            n_cams=n_cams,
        )
        time_freqs = self._time_freqs(
            batch_size=batch_size,
            seq_len=seq_len_in,
            n_cams=n_cams,
        )

        for camera_layer, time_layer in zip(
            self.camera_layers,
            self.time_layers,
            strict=True,
        ):
            x_camera = x.reshape(batch_size * seq_len_in, n_cams, self.hidden_dim)
            x_camera = camera_layer(
                x_camera,
                freqs_cis=camera_freqs,
                attn_mask=camera_mask,
            )
            x_camera = x_camera * camera_valid.unsqueeze(-1).to(dtype=x_camera.dtype)
            x = x_camera.reshape(batch_size, seq_len_in, n_cams, self.hidden_dim)

            x_time = x.permute(0, 2, 1, 3).reshape(batch_size * n_cams, seq_len_in, self.hidden_dim)
            x_time = time_layer(
                x_time,
                freqs_cis=time_freqs,
                attn_mask=time_mask,
            )
            x_time = x_time * time_valid.unsqueeze(-1).to(dtype=x_time.dtype)
            x = x_time.reshape(batch_size, n_cams, seq_len_in, self.hidden_dim).permute(0, 2, 1, 3)

        x = self._masked_camera_mean(x, token_valid_t)
        x = self.final_norm(x)

        return {
            "position": self.position_head(x),
            "rotation": self.rotation_head(x),
        }