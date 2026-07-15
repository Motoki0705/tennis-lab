"""Axial multi-view BLCS model with configurable line-map court tokens."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, cast

from torch import Tensor, nn

from src.tasks.blcs.models.components.heads import Trajectory3DHead, VelocityHead
from src.utils.models import (
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    default_ffn_dim,
    precompute_freqs_cis_nd,
    resolve_rope_bases,
)
from src.utils.models.axial_multiview_mixin import AxialMultiViewMixin
from src.utils.models.components.ops.time_local import build_local_attention_keep_mask
from src.utils.models.embeddings import (
    CourtBallGroupEmbedding,
    CourtLineMapBallGroupEmbedding,
    InvisibleTokenEmbedding,
)
from src.utils.schema.court import NUM_COURT_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BLCSMultiViewAxialModel(AxialMultiViewMixin, nn.Module):
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
        rope_theta_type: float | None = None,
        num_layers: int = 4,
        predict_velocity: bool = False,
        max_seq_len: int = 120,
        max_num_cameras: int = 8,
        invisible_init_std: float = 0.02,
        num_court_tokens: int = NUM_COURT_KP,
        court_input_type: Literal["kp", "line"] = "kp",
        line_map_channels: Sequence[int] = (16, 32, 64),
        num_line_map_tokens: int = 1,
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
        if court_input_type not in {"kp", "line"}:
            raise ValueError(
                f"court_input_type must be 'kp' or 'line', got {court_input_type!r}."
            )
        self.court_input_type = court_input_type
        self.line_map_channels = tuple(int(value) for value in line_map_channels)
        if not self.line_map_channels or any(
            value <= 0 for value in self.line_map_channels
        ):
            raise ValueError("line_map_channels must contain positive integers.")
        self.num_line_map_tokens = int(num_line_map_tokens)
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
        if self.court_input_type == "line":
            self._validate_rope_axis_capacity(rope_dim=rope_dim, n_axes=3)

        self.rope_dim = int(rope_dim)
        self.rope_bases = resolve_rope_bases(
            rope_theta,
            rope_theta_time,
            rope_theta_camera,
            self._coalesce_theta(rope_theta_type, rope_theta)
            if self.court_input_type == "line"
            else None,
        )

        if ffn_dim is None:
            ffn_dim = default_ffn_dim(self.hidden_dim)

        self.invisible_token = InvisibleTokenEmbedding(
            dim=self.hidden_dim,
            init_std=invisible_init_std,
        )
        if self.court_input_type == "kp":
            self.group_embed: CourtBallGroupEmbedding | None = CourtBallGroupEmbedding(
                dim=self.hidden_dim,
                invisible_token=self.invisible_token,
                num_court_tokens=self.num_court_tokens,
            )
            self.line_group_embed: CourtLineMapBallGroupEmbedding | None = None
        else:
            self.group_embed = None
            self.line_group_embed = CourtLineMapBallGroupEmbedding(
                dim=self.hidden_dim,
                line_map_channels=self.line_map_channels,
                num_line_map_tokens=self.num_line_map_tokens,
                invisible_token=self.invisible_token,
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

        token_type_ids = (
            self._build_line_token_type_ids(self.num_line_map_tokens)
            if self.court_input_type == "line"
            else None
        )
        token_freqs = precompute_freqs_cis_nd(
            dim=self.rope_dim,
            pos=self._build_token_positions(
                seq_len=self.max_seq_len,
                n_cams=self.max_num_cameras,
                token_type_ids=token_type_ids,
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

    @classmethod
    def from_config(cls, config: DictConfig) -> BLCSMultiViewAxialModel:
        """Create model from Hydra/OmegaConf config."""
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})

        return cls(
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            attention_type=cast(
                Literal["mha", "gqa"], str(model_cfg.get("attention_type", "mha"))
            ),
            num_kv_heads=model_cfg.get("num_kv_heads", None),
            ffn_dim=model_cfg.get("ffn_dim", None),
            ffn_type=cast(
                Literal["swiglu", "mlp"], str(model_cfg.get("ffn_type", "swiglu"))
            ),
            dropout=float(model_cfg.get("dropout", 0.1)),
            rope_dim=model_cfg.get("rope_dim", None),
            rope_theta=float(model_cfg.get("rope_theta", 10000.0)),
            rope_theta_time=model_cfg.get("rope_theta_time", None),
            rope_theta_camera=model_cfg.get("rope_theta_camera", None),
            rope_theta_type=model_cfg.get("rope_theta_type", None),
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
            court_input_type=cast(
                Literal["kp", "line"], str(model_cfg.get("court_input_type", "kp"))
            ),
            line_map_channels=model_cfg.get("line_map_channels", [16, 32, 64]),
            num_line_map_tokens=int(model_cfg.get("num_line_map_tokens", 1)),
            time_window_radius=int(model_cfg.get("time_window_radius", 16)),
            camera_layers_per_stage=model_cfg.get("camera_layers_per_stage", None),
            time_layers_per_stage=model_cfg.get("time_layers_per_stage", None),
            time_global_stage_mask=model_cfg.get("time_global_stage_mask", None),
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
        court_kp: Tensor | None = None,
        ball_vis: Tensor | None = None,
        ball_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
        court_line_map: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass for multi-view BLCS inputs."""
        if self.court_input_type == "kp":
            if court_line_map is not None:
                raise ValueError(
                    "court_line_map must not be provided in court_input_type='kp'."
                )
            (
                prepared_court_kp,
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
            court_flat = prepared_court_kp.reshape(
                batch_size * n_cams * seq_len_in, self.num_court_tokens, 2
            )
            ball_flat = ball_uv.reshape(batch_size * n_cams * seq_len_in, 2)
            group_vis = ball_vis.reshape(batch_size * n_cams * seq_len_in)
            if self.group_embed is None:
                raise RuntimeError("KP group embedding is not initialized.")
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
            tokens_per_camera = 1
        else:
            if court_kp is not None or court_vis is not None:
                raise ValueError(
                    "court_kp and court_vis must not be provided in court_input_type='line'."
                )
            (
                prepared_line_map,
                ball_vis,
                ball_mask,
                batch_size,
                n_cams,
                seq_len_in,
            ) = self._prepare_line_forward_inputs(
                ball_uv=ball_uv,
                court_line_map=court_line_map,
                ball_vis=ball_vis,
                ball_mask=ball_mask,
            )
            line_map_flat = prepared_line_map.reshape(
                batch_size * n_cams * seq_len_in,
                1,
                prepared_line_map.shape[-2],
                prepared_line_map.shape[-1],
            )
            ball_flat = ball_uv.reshape(batch_size * n_cams * seq_len_in, 2)
            group_vis = ball_vis.reshape(batch_size * n_cams * seq_len_in)
            if self.line_group_embed is None:
                raise RuntimeError("Line group embedding is not initialized.")
            x = (
                self.line_group_embed(line_map_flat, ball_flat, group_vis)
                .reshape(
                    batch_size,
                    n_cams,
                    seq_len_in,
                    self.num_line_map_tokens + 1,
                    self.hidden_dim,
                )
                .permute(0, 2, 1, 3, 4)
                .reshape(
                    batch_size,
                    seq_len_in,
                    n_cams * (self.num_line_map_tokens + 1),
                    self.hidden_dim,
                )
            )
            tokens_per_camera = self.num_line_map_tokens + 1
            token_valid = (ball_mask > 0).permute(0, 2, 1).repeat_interleave(
                tokens_per_camera,
                dim=2,
            )

        axis_tokens = n_cams * tokens_per_camera
        camera_valid = token_valid.reshape(batch_size * seq_len_in, axis_tokens)
        time_valid = token_valid.permute(0, 2, 1).reshape(
            batch_size * axis_tokens, seq_len_in
        )
        camera_mask, _ = self._build_self_attn_mask(camera_valid)
        time_mask, time_valid = self._build_self_attn_mask(time_valid)
        camera_freqs = self._camera_freqs(
            batch_size=batch_size,
            seq_len=seq_len_in,
            n_cams=n_cams,
            tokens_per_camera=tokens_per_camera,
        )
        time_freqs = self._time_freqs(
            batch_size=batch_size,
            seq_len=seq_len_in,
            n_cams=n_cams,
            tokens_per_camera=tokens_per_camera,
        )
        sliding_mask = self._build_sliding_attn_mask(
            time_valid,
            radius=self.time_window_radius,
        )

        for stage_index, (camera_stage_module, time_stage_module) in enumerate(
            zip(
                self.camera_layers,
                self.time_layers,
                strict=True,
            )
        ):
            camera_stage_layers = cast(nn.ModuleList, camera_stage_module)
            time_stage_layers = cast(nn.ModuleList, time_stage_module)
            for camera_layer_module in camera_stage_layers:
                camera_layer = cast(TransformerBlock, camera_layer_module)
                x_camera = x.reshape(
                    batch_size * seq_len_in, axis_tokens, self.hidden_dim
                )
                x_camera = camera_layer(
                    x_camera,
                    freqs_cis=camera_freqs,
                    attn_mask=camera_mask,
                )
                x = x_camera.reshape(
                    batch_size, seq_len_in, axis_tokens, self.hidden_dim
                )

            time_layers_in_stage = len(time_stage_layers)
            for time_layer_index, time_layer_module in enumerate(time_stage_layers):
                time_layer = cast(TransformerBlock, time_layer_module)
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
        court_kp: Tensor | None,
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

        if court_kp is None:
            raise ValueError("court_kp is required for court_input_type='kp'.")
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

    def _prepare_line_forward_inputs(
        self,
        *,
        ball_uv: Tensor,
        court_line_map: Tensor | None,
        ball_vis: Tensor | None,
        ball_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, int, int, int]:
        if ball_uv.dim() != 4:
            raise ValueError(
                f"ball_uv must have shape (B, N, T, 2), got {tuple(ball_uv.shape)}"
            )
        batch_size, n_cams, seq_len_in, _ = ball_uv.shape
        if seq_len_in > self.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len_in} exceeds max_seq_len={self.max_seq_len}."
            )
        if n_cams > self.max_num_cameras:
            raise ValueError(
                f"n_cams={n_cams} exceeds max_num_cameras={self.max_num_cameras}."
            )
        if court_line_map is None:
            raise ValueError("court_line_map is required for court_input_type='line'.")
        if court_line_map.dim() == 5:
            court_line_map = court_line_map.unsqueeze(2).expand(
                -1, -1, seq_len_in, -1, -1, -1
            )
        if court_line_map.dim() != 6 or court_line_map.shape[:3] != (
            batch_size,
            n_cams,
            seq_len_in,
        ) or court_line_map.shape[3] != 1:
            raise ValueError(
                "court_line_map must have shape (B,N,T,1,H,W), got "
                f"{tuple(court_line_map.shape)}."
            )
        if ball_vis is None or tuple(ball_vis.shape) != (
            batch_size,
            n_cams,
            seq_len_in,
        ):
            raise ValueError(
                "ball_vis is required with shape "
                f"{(batch_size, n_cams, seq_len_in)} for line mode."
            )
        if ball_mask is None or tuple(ball_mask.shape) != (
            batch_size,
            n_cams,
            seq_len_in,
        ):
            raise ValueError(
                "ball_mask is required with shape "
                f"{(batch_size, n_cams, seq_len_in)} for line mode."
            )
        return court_line_map, ball_vis, ball_mask, batch_size, n_cams, seq_len_in
