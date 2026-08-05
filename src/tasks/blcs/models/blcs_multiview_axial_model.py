"""Axial multi-view BLCS model with one token per camera/time element."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

from torch import Tensor, nn

from src.tasks.blcs.configuration import AxialModelConfig
from src.tasks.blcs.models.components.heads import Trajectory3DHead, VelocityHead
from src.utils.models import (
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    precompute_freqs_cis_nd,
    resolve_axial_rope_bases,
)
from src.utils.models.axial_multiview_mixin import AxialMultiViewMixin
from src.utils.models.components.ops.time_local import build_local_attention_keep_mask
from src.utils.models.embeddings import CourtBallGroupEmbedding, InvisibleTokenEmbedding


class BLCSMultiViewAxialModel(AxialMultiViewMixin, nn.Module):
    """BLCS multi-view model with alternating camera/time self-attention."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        attention_type: Literal["mha", "gqa"],
        num_kv_heads: int | None,
        ffn_dim: int,
        ffn_type: Literal["swiglu", "mlp"],
        dropout: float,
        rope_dim: int,
        rope_theta_time: float,
        rope_theta_camera: float,
        num_layers: int,
        predict_velocity: bool,
        max_seq_len: int,
        max_num_cameras: int,
        invisible_init_std: float,
        num_court_tokens: int,
        time_window_radius: int,
        camera_layers_per_stage: Sequence[int],
        time_layers_per_stage: Sequence[int],
        time_global_stage_mask: Sequence[bool],
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        num_layers = int(num_layers)
        camera_layers_per_stage = self._normalize_stage_ints(
            camera_layers_per_stage,
        )
        time_layers_per_stage = self._normalize_stage_ints(
            time_layers_per_stage,
        )
        time_global_stage_mask = self._normalize_stage_mask(
            time_global_stage_mask,
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
        self._validate_rope_dim(rope_dim=rope_dim, head_dim=head_dim)

        self.rope_dim = int(rope_dim)
        self.rope_bases = resolve_axial_rope_bases(
            rope_theta_time=rope_theta_time,
            rope_theta_camera=rope_theta_camera,
        )

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
            raise ValueError(
                "camera_layers_per_stage must contain only positive integers"
            )
        if any(layer_count <= 0 for layer_count in time_layers_per_stage):
            raise ValueError(
                "time_layers_per_stage must contain only positive integers"
            )

    @staticmethod
    def _normalize_stage_ints(
        values: Sequence[int],
    ) -> tuple[int, ...]:
        return tuple(int(value) for value in values)

    @staticmethod
    def _normalize_stage_mask(
        values: Sequence[bool],
    ) -> tuple[bool, ...]:
        return tuple(bool(value) for value in values)

    @classmethod
    def from_config(cls, config: AxialModelConfig) -> BLCSMultiViewAxialModel:
        """Create model from Hydra/OmegaConf config."""
        raw_attention_type = config.attention_type
        if raw_attention_type == "mha":
            attention_type: Literal["mha", "gqa"] = "mha"
        elif raw_attention_type == "gqa":
            attention_type = "gqa"
        else:
            raise ValueError(f"Unsupported attention_type={raw_attention_type!r}")
        raw_ffn_type = config.ffn_type
        if raw_ffn_type == "swiglu":
            ffn_type: Literal["swiglu", "mlp"] = "swiglu"
        elif raw_ffn_type == "mlp":
            ffn_type = "mlp"
        else:
            raise ValueError(f"Unsupported ffn_type={raw_ffn_type!r}")

        return cls(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            attention_type=attention_type,
            num_kv_heads=config.num_kv_heads,
            ffn_dim=config.ffn_dim,
            ffn_type=ffn_type,
            dropout=config.dropout,
            rope_dim=config.rope_dim,
            rope_theta_time=config.rope_theta_time,
            rope_theta_camera=config.rope_theta_camera,
            num_layers=config.num_layers,
            predict_velocity=config.predict_velocity,
            max_seq_len=config.max_seq_len,
            max_num_cameras=config.max_num_cameras,
            invisible_init_std=config.invisible_init_std,
            num_court_tokens=config.num_court_tokens,
            time_window_radius=config.time_window_radius,
            camera_layers_per_stage=config.camera_layers_per_stage,
            time_layers_per_stage=config.time_layers_per_stage,
            time_global_stage_mask=config.time_global_stage_mask,
        )

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
        return cast(
            Tensor,
            x_time.reshape(batch_size, n_cams, seq_len, self.hidden_dim).permute(
                0, 2, 1, 3
            ),
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

        if court_vis is not None:
            court_visible = court_vis > 0
            court_kp = court_kp.masked_fill(~court_visible.unsqueeze(-1), 0.0)
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

        for stage_index, (camera_stage_layers, time_stage_layers) in enumerate(
            zip(
                self.camera_layers,
                self.time_layers,
                strict=True,
            )
        ):
            if not isinstance(camera_stage_layers, nn.ModuleList) or not isinstance(
                time_stage_layers, nn.ModuleList
            ):
                raise TypeError("Axial stages must contain ModuleList instances.")
            for camera_layer in camera_stage_layers:
                if not isinstance(camera_layer, TransformerBlock):
                    raise TypeError("Axial camera layers must be TransformerBlock.")
                x_camera = x.reshape(batch_size * seq_len_in, n_cams, self.hidden_dim)
                x_camera = camera_layer(
                    x_camera,
                    freqs_cis=camera_freqs,
                    attn_mask=camera_mask,
                )
                x = x_camera.reshape(batch_size, seq_len_in, n_cams, self.hidden_dim)

            time_layers_in_stage = len(time_stage_layers)
            for time_layer_index, time_layer in enumerate(time_stage_layers):
                if not isinstance(time_layer, TransformerBlock):
                    raise TypeError("Axial time layers must be TransformerBlock.")
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
