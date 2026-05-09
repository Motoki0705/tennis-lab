"""Axial multi-view BLCS model with one token per camera/time element."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import torch
from torch import Tensor, nn

from src.tasks.blcs.models.components.heads import Trajectory3DHead, VelocityHead
from src.utils.models import (
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    precompute_freqs_cis_nd,
)
from src.utils.models.embeddings import CourtBallGroupEmbedding, InvisibleTokenEmbedding
from src.utils.schema.court import NUM_COURT_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BLCSMultiViewAxialModel(nn.Module):
    """BLCS multi-view model with alternating camera/time self-attention."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        attention_type: Literal["mha", "gqa"] = "mha",
        num_kv_heads: int | None = None,
        ffn_dim: int | None = None,
        ffn_type: Literal["swiglu", "mlp"] = "swiglu",
        dropout: float = 0.1,
        rope_dim: int | None = None,
        rope_theta: float = 10000.0,
        rope_theta_time: float | None = None,
        rope_theta_camera: float | None = None,
        num_layers: int = 4,
        predict_velocity: bool = False,
        max_seq_len: int = 120,
        max_num_cameras: int = 8,
        invisible_init_std: float = 0.02,
        num_court_tokens: int = NUM_COURT_KP,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        if self.hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} must be divisible by num_heads={num_heads}"
            )
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if max_num_cameras <= 0:
            raise ValueError(f"max_num_cameras must be positive, got {max_num_cameras}")
        if num_layers < 0:
            raise ValueError(f"num_layers must be non-negative, got {num_layers}")

        self.max_seq_len = int(max_seq_len)
        self.max_num_cameras = int(max_num_cameras)
        self.predict_velocity = bool(predict_velocity)
        self.num_court_tokens = int(num_court_tokens)

        head_dim = self.hidden_dim // num_heads
        rope_dim = head_dim if rope_dim is None else int(rope_dim)
        if rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be even, got {rope_dim}")
        if rope_dim > head_dim:
            raise ValueError(f"rope_dim={rope_dim} cannot exceed head_dim={head_dim}")

        self.rope_dim = int(rope_dim)
        self.rope_bases = (
            float(self._coalesce_theta(rope_theta_time, rope_theta)),
            float(self._coalesce_theta(rope_theta_camera, rope_theta)),
        )

        if ffn_dim is None:
            ffn_dim = int((8 * self.hidden_dim) / 3)
            ffn_dim = (ffn_dim + 63) // 64 * 64

        self.invisible_token = InvisibleTokenEmbedding(
            dim=self.hidden_dim,
            init_std=invisible_init_std,
        )
        self.group_embed = CourtBallGroupEmbedding(
            dim=self.hidden_dim,
            invisible_token=self.invisible_token,
            num_court_tokens=self.num_court_tokens,
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
                        attention_type=attention_type,
                        n_kv_heads=num_kv_heads,
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
                        attention_type=attention_type,
                        n_kv_heads=num_kv_heads,
                        rope_base=self.rope_bases[0],
                        ffn_type=ffn_type,
                    )
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm(self.hidden_dim)

        self.position_head = Trajectory3DHead(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim // 2,
            output_dim=3,
            num_layers=2,
            dropout=dropout,
        )
        self.velocity_head = None
        if self.predict_velocity:
            self.velocity_head = VelocityHead(
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim // 2,
                output_dim=3,
                num_layers=2,
                dropout=dropout,
            )

        token_freqs = precompute_freqs_cis_nd(
            dim=self.rope_dim,
            pos=self._build_token_positions(
                seq_len=self.max_seq_len,
                n_cams=self.max_num_cameras,
            ),
            base=self.rope_bases,
        )
        self.register_buffer("token_freqs_cis", token_freqs, persistent=False)

    @classmethod
    def from_config(cls, config: DictConfig) -> BLCSMultiViewAxialModel:
        """Create model from Hydra/OmegaConf config."""
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})

        return cls(
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            attention_type=str(model_cfg.get("attention_type", "mha")),
            num_kv_heads=model_cfg.get("num_kv_heads", None),
            ffn_dim=model_cfg.get("ffn_dim", None),
            ffn_type=str(model_cfg.get("ffn_type", "swiglu")),
            dropout=float(model_cfg.get("dropout", 0.1)),
            rope_dim=model_cfg.get("rope_dim", None),
            rope_theta=float(model_cfg.get("rope_theta", 10000.0)),
            rope_theta_time=model_cfg.get("rope_theta_time", None),
            rope_theta_camera=model_cfg.get("rope_theta_camera", None),
            num_layers=int(model_cfg.get("num_layers", 4)),
            predict_velocity=bool(model_cfg.get("predict_velocity", False)),
            max_seq_len=int(
                model_cfg.get("max_seq_len", data_cfg.get("max_seq_len", 120))
            ),
            max_num_cameras=int(
                model_cfg.get("max_num_cameras", model_cfg.get("max_views", 8))
            ),
            invisible_init_std=float(model_cfg.get("invisible_init_std", 0.02)),
            num_court_tokens=int(
                model_cfg.get(
                    "num_court_tokens", data_cfg.get("num_court_kp", NUM_COURT_KP)
                )
            ),
        )

    @staticmethod
    def _coalesce_theta(theta: float | None, fallback: float) -> float:
        return fallback if theta is None else float(theta)

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
        return (
            freqs.unsqueeze(0)
            .expand(
                batch_size,
                seq_len,
                n_cams,
                self.rope_dim // 2,
            )
            .reshape(batch_size * seq_len, n_cams, self.rope_dim // 2)
        )

    def _time_freqs(self, *, batch_size: int, seq_len: int, n_cams: int) -> Tensor:
        freqs = self.token_freqs_cis[:seq_len, :n_cams].permute(1, 0, 2)
        return (
            freqs.unsqueeze(0)
            .expand(
                batch_size,
                n_cams,
                seq_len,
                self.rope_dim // 2,
            )
            .reshape(batch_size * n_cams, seq_len, self.rope_dim // 2)
        )

    def forward(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_vis: Tensor | None = None,
        ball_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass for multi-view BLCS inputs."""
        if ball_uv.dim() != 4:
            raise ValueError(
                f"ball_uv must have shape (B, N, T, 2), got {tuple(ball_uv.shape)}"
            )

        batch_size, n_cams, seq_len_in, _ = ball_uv.shape
        if seq_len_in > self.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len_in} exceeds max_seq_len={self.max_seq_len}. "
                "Increase model.max_seq_len."
            )
        if n_cams > self.max_num_cameras:
            raise ValueError(
                f"n_cams={n_cams} exceeds max_num_cameras={self.max_num_cameras}."
            )

        if court_kp.dim() == 4:
            court_kp = court_kp.unsqueeze(2).expand(-1, -1, seq_len_in, -1, -1)
        if court_kp.dim() != 5:
            raise ValueError(
                "court_kp must have shape "
                f"(B, N, T, {self.num_court_tokens}, 2) or "
                f"(B, N, {self.num_court_tokens}, 2), "
                f"got {tuple(court_kp.shape)}"
            )
        if court_kp.shape[-2] != self.num_court_tokens:
            raise ValueError(
                f"Expected court_kp with K={self.num_court_tokens}, got K={court_kp.shape[-2]}."
            )

        if court_vis is not None:
            if court_vis.dim() == 3:
                court_vis = court_vis.unsqueeze(2).expand(-1, -1, seq_len_in, -1)
            if court_vis.dim() != 4:
                raise ValueError(
                    "court_vis must have shape "
                    f"(B, N, T, {self.num_court_tokens}) or "
                    f"(B, N, {self.num_court_tokens}), "
                    f"got {tuple(court_vis.shape)}"
                )
            if court_vis.shape[-1] != self.num_court_tokens:
                raise ValueError(
                    f"Expected court_vis with K={self.num_court_tokens}, got K={court_vis.shape[-1]}."
                )

        if ball_vis is None:
            raise ValueError(
                "ball_vis is required for BLCSMultiViewAxialModel forward."
            )
        if ball_mask is None:
            raise ValueError(
                "ball_mask is required for BLCSMultiViewAxialModel forward."
            )
        if ball_vis.shape != (batch_size, n_cams, seq_len_in):
            raise ValueError(
                f"ball_vis must have shape {(batch_size, n_cams, seq_len_in)}, "
                f"got {tuple(ball_vis.shape)}"
            )
        if ball_mask.shape != (batch_size, n_cams, seq_len_in):
            raise ValueError(
                f"ball_mask must have shape {(batch_size, n_cams, seq_len_in)}, "
                f"got {tuple(ball_mask.shape)}"
            )

        court_flat = court_kp.reshape(
            batch_size * n_cams * seq_len_in, self.num_court_tokens, 2
        )
        ball_flat = ball_uv.reshape(batch_size * n_cams * seq_len_in, 2)
        group_vis = ball_vis.reshape(batch_size * n_cams * seq_len_in)
        x = (
            self.group_embed(court_flat, ball_flat, group_vis)
            .reshape(
                batch_size,
                n_cams,
                seq_len_in,
                self.hidden_dim,
            )
            .permute(0, 2, 1, 3)
        )

        token_valid = (ball_mask > 0).permute(0, 2, 1)

        camera_valid = token_valid.reshape(batch_size * seq_len_in, n_cams)
        time_valid = token_valid.permute(0, 2, 1).reshape(
            batch_size * n_cams, seq_len_in
        )
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
            x = x_camera.reshape(batch_size, seq_len_in, n_cams, self.hidden_dim)

            x_time = x.permute(0, 2, 1, 3).reshape(
                batch_size * n_cams, seq_len_in, self.hidden_dim
            )
            x_time = time_layer(
                x_time,
                freqs_cis=time_freqs,
                attn_mask=time_mask,
            )
            x = x_time.reshape(batch_size, n_cams, seq_len_in, self.hidden_dim).permute(
                0, 2, 1, 3
            )

        x = x[:, :, 0, :]
        x = self.final_norm(x)

        out: dict[str, Tensor] = {"position": self.position_head(x)}
        if self.predict_velocity and self.velocity_head is not None:
            out["velocity"] = self.velocity_head(x)
        return out
