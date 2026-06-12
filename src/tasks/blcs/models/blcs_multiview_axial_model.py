"""Axial multi-view BLCS model with one token per camera/time element."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor, nn

from src.tasks.blcs.models.components.heads import Trajectory3DHead, VelocityHead
from src.utils.models import (
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    build_self_attn_mask,
    default_ffn_dim,
    precompute_freqs_cis_nd,
    validate_rope_dim,
)
from src.utils.models.components.ops.time_local import build_local_attention_keep_mask
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
        time_window_radius: int = 16,
        camera_layers_per_stage: Sequence[int] | None = None,
        time_layers_per_stage: Sequence[int] | None = None,
        time_global_stage_mask: Sequence[bool] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        num_layers = int(num_layers)
        camera_layers_per_stage = self._normalize_stage_ints(
            camera_layers_per_stage,
            num_layers=num_layers,
            default_value=1,
        )
        time_layers_per_stage = self._normalize_stage_ints(
            time_layers_per_stage,
            num_layers=num_layers,
            default_value=1,
        )
        time_global_stage_mask = self._normalize_stage_mask(
            time_global_stage_mask,
            num_layers=num_layers,
        )

        self._validate_init_args(
            hidden_dim=self.hidden_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            max_num_cameras=max_num_cameras,
            num_layers=num_layers,
            camera_layers_per_stage=camera_layers_per_stage,
            time_layers_per_stage=time_layers_per_stage,
            time_global_stage_mask=time_global_stage_mask,
        )

        self.num_layers = num_layers
        self.max_seq_len = int(max_seq_len)
        self.max_num_cameras = int(max_num_cameras)
        self.predict_velocity = bool(predict_velocity)
        self.num_court_tokens = int(num_court_tokens)
        self.time_window_radius = int(time_window_radius)
        self.camera_layers_per_stage = camera_layers_per_stage
        self.time_layers_per_stage = time_layers_per_stage
        self.time_global_stage_mask = time_global_stage_mask
        if self.time_window_radius < 0:
            raise ValueError(
                f"time_window_radius must be non-negative, got {self.time_window_radius}"
            )

        head_dim = self.hidden_dim // num_heads
        rope_dim = head_dim if rope_dim is None else int(rope_dim)
        self._validate_rope_dim(rope_dim=rope_dim, head_dim=head_dim)

        self.rope_dim = int(rope_dim)
        self.rope_bases = (
            float(self._coalesce_theta(rope_theta_time, rope_theta)),
            float(self._coalesce_theta(rope_theta_camera, rope_theta)),
        )

        if ffn_dim is None:
            ffn_dim = default_ffn_dim(self.hidden_dim)

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
                nn.ModuleList(
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
                        for _ in range(camera_layer_count)
                    ]
                )
                for camera_layer_count in self.camera_layers_per_stage
            ]
        )
        self.time_layers = nn.ModuleList(
            [
                nn.ModuleList(
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
                        for _ in range(time_layer_count)
                    ]
                )
                for time_layer_count in self.time_layers_per_stage
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

    @staticmethod
    def _validate_init_args(
        *,
        hidden_dim: int,
        num_heads: int,
        max_seq_len: int,
        max_num_cameras: int,
        num_layers: int,
        camera_layers_per_stage: tuple[int, ...],
        time_layers_per_stage: tuple[int, ...],
        time_global_stage_mask: tuple[bool, ...],
    ) -> None:
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if max_num_cameras <= 0:
            raise ValueError(f"max_num_cameras must be positive, got {max_num_cameras}")
        if num_layers < 0:
            raise ValueError(f"num_layers must be non-negative, got {num_layers}")
        if len(camera_layers_per_stage) != num_layers:
            raise ValueError(
                "camera_layers_per_stage length must equal num_layers, got "
                f"{len(camera_layers_per_stage)} and {num_layers}"
            )
        if len(time_layers_per_stage) != num_layers:
            raise ValueError(
                "time_layers_per_stage length must equal num_layers, got "
                f"{len(time_layers_per_stage)} and {num_layers}"
            )
        if len(time_global_stage_mask) != num_layers:
            raise ValueError(
                "time_global_stage_mask length must equal num_layers, got "
                f"{len(time_global_stage_mask)} and {num_layers}"
            )
        if any(layer_count <= 0 for layer_count in camera_layers_per_stage):
            raise ValueError("camera_layers_per_stage must contain only positive integers")
        if any(layer_count <= 0 for layer_count in time_layers_per_stage):
            raise ValueError("time_layers_per_stage must contain only positive integers")

    @staticmethod
    def _normalize_stage_ints(
        values: Sequence[int] | None,
        *,
        num_layers: int,
        default_value: int,
    ) -> tuple[int, ...]:
        if values is None:
            return tuple(default_value for _ in range(num_layers))
        return tuple(int(value) for value in values)

    @staticmethod
    def _normalize_stage_mask(
        values: Sequence[bool] | None,
        *,
        num_layers: int,
    ) -> tuple[bool, ...]:
        if values is None:
            return tuple(False for _ in range(num_layers))
        return tuple(bool(value) for value in values)

    @staticmethod
    def _validate_rope_dim(*, rope_dim: int, head_dim: int) -> None:
        validate_rope_dim(rope_dim=rope_dim, head_dim=head_dim)

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
            max_num_cameras=int(model_cfg.get("max_num_cameras", 8)),
            invisible_init_std=float(model_cfg.get("invisible_init_std", 0.02)),
            num_court_tokens=int(
                model_cfg.get(
                    "num_court_tokens", data_cfg.get("num_court_kp", NUM_COURT_KP)
                )
            ),
            time_window_radius=int(model_cfg.get("time_window_radius", 16)),
            camera_layers_per_stage=model_cfg.get("camera_layers_per_stage", None),
            time_layers_per_stage=model_cfg.get("time_layers_per_stage", None),
            time_global_stage_mask=model_cfg.get("time_global_stage_mask", None),
        )

    @staticmethod
    def _coalesce_theta(theta: float | None, fallback: float) -> float:
        return fallback if theta is None else float(theta)

    @staticmethod
    def _build_self_attn_mask(valid: Tensor) -> tuple[Tensor, Tensor]:
        """Build self-attention mask from valid mask.

        Delegates to :func:`src.utils.models.build_self_attn_mask`.
        See that function for full documentation.
        """
        return build_self_attn_mask(valid)

    @staticmethod
    def _build_sliding_attn_mask(valid: Tensor, radius: int) -> Tensor:
        return build_local_attention_keep_mask(valid, radius)

    def _use_global_time_attention(
        self,
        *,
        stage_index: int,
        time_layer_index: int,
        time_layers_in_stage: int,
    ) -> bool:
        return self.time_global_stage_mask[stage_index] and (
            time_layer_index == time_layers_in_stage - 1
        )

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

    def _apply_time_attention_layer(
        self,
        x: Tensor,
        *,
        time_layer: TransformerBlock,
        use_global_attention: bool,
        time_full_mask: Tensor,
        sliding_mask: Tensor,
        time_freqs: Tensor,
    ) -> Tensor:
        batch_size, seq_len, n_cams, _ = x.shape
        x_time = x.permute(0, 2, 1, 3).reshape(
            batch_size * n_cams, seq_len, self.hidden_dim
        )
        attn_mask = time_full_mask if use_global_attention else sliding_mask
        x_time = time_layer(
            x_time,
            freqs_cis=time_freqs,
            attn_mask=attn_mask,
        )
        return x_time.reshape(batch_size, n_cams, seq_len, self.hidden_dim).permute(
            0, 2, 1, 3
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
        (
            court_kp,
            ball_vis,
            ball_mask,
            court_vis,
            batch_size,
            n_cams,
            seq_len_in,
        ) = self._prepare_forward_inputs(
            ball_uv=ball_uv,
            court_kp=court_kp,
            ball_vis=ball_vis,
            ball_mask=ball_mask,
            court_vis=court_vis,
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
        time_mask, time_valid = self._build_self_attn_mask(time_valid)
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
        sliding_mask = self._build_sliding_attn_mask(
            time_valid,
            radius=self.time_window_radius,
        )

        for stage_index, (camera_stage_layers, time_stage_layers) in enumerate(zip(
            self.camera_layers,
            self.time_layers,
            strict=True,
        )):
            for camera_layer in camera_stage_layers:
                x_camera = x.reshape(batch_size * seq_len_in, n_cams, self.hidden_dim)
                x_camera = camera_layer(
                    x_camera,
                    freqs_cis=camera_freqs,
                    attn_mask=camera_mask,
                )
                x = x_camera.reshape(batch_size, seq_len_in, n_cams, self.hidden_dim)

            time_layers_in_stage = len(time_stage_layers)
            for time_layer_index, time_layer in enumerate(time_stage_layers):
                x = self._apply_time_attention_layer(
                    x,
                    time_layer=time_layer,
                    use_global_attention=self._use_global_time_attention(
                        stage_index=stage_index,
                        time_layer_index=time_layer_index,
                        time_layers_in_stage=time_layers_in_stage,
                    ),
                    time_full_mask=time_mask,
                    sliding_mask=sliding_mask,
                    time_freqs=time_freqs,
                )

        x = x[:, :, 0, :]
        x = self.final_norm(x)

        out: dict[str, Tensor] = {"position": self.position_head(x)}
        if self.predict_velocity and self.velocity_head is not None:
            out["velocity"] = self.velocity_head(x)
        return out

    def _prepare_forward_inputs(
        self,
        *,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_vis: Tensor | None,
        ball_mask: Tensor | None,
        court_vis: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, int, int, int]:
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
        return court_kp, ball_vis, ball_mask, court_vis, batch_size, n_cams, seq_len_in
