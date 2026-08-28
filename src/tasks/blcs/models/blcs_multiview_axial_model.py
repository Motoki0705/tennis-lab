"""Axial multi-view BLCS model with one token per camera/time element."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

from torch import Tensor, nn

from src.tasks.blcs.configuration import AxialModelConfig
from src.tasks.blcs.models.components.heads import build_trajectory_output
from src.tasks.blcs.models.components.padding import (
    build_axial_padding_masks,
    mask_trajectory_outputs,
)
from src.utils.models import (
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    precompute_freqs_cis_nd,
    resolve_axial_rope_bases,
    validate_rope_dim,
)
from src.utils.models.axial_multiview_mixin import AxialMultiViewMixin
from src.utils.models.components.ffn_layers import FFNType
from src.utils.models.embeddings import CourtBallGroupEmbedding, InvisibleTokenEmbedding


class _GlobalTimeAttention(nn.Module):
    """Time attention implementation fixed to the full-sequence mask."""

    def __init__(self, block: TransformerBlock, hidden_dim: int) -> None:
        super().__init__()
        self.block = block
        self.hidden_dim = hidden_dim

    def forward(
        self,
        x: Tensor,
        full_attention_keep_mask: Tensor,
        sliding_attention_keep_mask: Tensor,
        frequencies: Tensor,
    ) -> Tensor:
        """Apply full-sequence time attention."""
        del sliding_attention_keep_mask
        batch_size, seq_len, num_cameras = x.shape[:3]
        values = x.permute(0, 2, 1, 3).reshape(
            batch_size * num_cameras,
            seq_len,
            self.hidden_dim,
        )
        values = cast(
            "Tensor",
            self.block(
                values,
                freqs_cis=frequencies,
                attn_mask=full_attention_keep_mask,
            ),
        )
        return values.reshape(
            batch_size,
            num_cameras,
            seq_len,
            self.hidden_dim,
        ).permute(0, 2, 1, 3)


class _SlidingTimeAttention(nn.Module):
    """Time attention implementation fixed to the configured local mask."""

    def __init__(self, block: TransformerBlock, hidden_dim: int) -> None:
        super().__init__()
        self.block = block
        self.hidden_dim = hidden_dim

    def forward(
        self,
        x: Tensor,
        full_attention_keep_mask: Tensor,
        sliding_attention_keep_mask: Tensor,
        frequencies: Tensor,
    ) -> Tensor:
        """Apply sliding-window time attention."""
        del full_attention_keep_mask
        batch_size, seq_len, num_cameras = x.shape[:3]
        values = x.permute(0, 2, 1, 3).reshape(
            batch_size * num_cameras,
            seq_len,
            self.hidden_dim,
        )
        values = cast(
            "Tensor",
            self.block(
                values,
                freqs_cis=frequencies,
                attn_mask=sliding_attention_keep_mask,
            ),
        )
        return values.reshape(
            batch_size,
            num_cameras,
            seq_len,
            self.hidden_dim,
        ).permute(0, 2, 1, 3)


class _AxialAttentionStage(nn.Module):
    """One preconstructed camera/time stage without runtime implementation selection."""

    def __init__(
        self,
        *,
        camera_layers: list[TransformerBlock],
        time_layers: list[nn.Module],
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.camera_layers = nn.ModuleList(camera_layers)
        self.time_layers = nn.ModuleList(time_layers)
        self.hidden_dim = hidden_dim

    def forward(
        self,
        x: Tensor,
        camera_attention_keep_mask: Tensor,
        time_attention_keep_mask: Tensor,
        sliding_attention_keep_mask: Tensor,
        camera_frequencies: Tensor,
        time_frequencies: Tensor,
    ) -> Tensor:
        """Apply the fixed camera layers followed by fixed time layers."""
        batch_size, seq_len, num_cameras = x.shape[:3]
        for layer in self.camera_layers:
            camera_values = x.reshape(
                batch_size * seq_len,
                num_cameras,
                self.hidden_dim,
            )
            camera_values = layer(
                camera_values,
                freqs_cis=camera_frequencies,
                attn_mask=camera_attention_keep_mask,
            )
            x = camera_values.reshape(
                batch_size,
                seq_len,
                num_cameras,
                self.hidden_dim,
            )
        for layer in self.time_layers:
            x = layer(
                x,
                time_attention_keep_mask,
                sliding_attention_keep_mask,
                time_frequencies,
            )
        return x


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
        ffn_type: FFNType,
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
        validate_rope_dim(rope_dim=rope_dim, head_dim=head_dim)

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

        camera_block_config = TransformerBlockConfig(
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
        time_block_config = TransformerBlockConfig(
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
        stages: list[_AxialAttentionStage] = []
        for camera_count, time_count, global_last in zip(
            self.camera_layers_per_stage,
            self.time_layers_per_stage,
            self.time_global_stage_mask,
            strict=True,
        ):
            time_implementations: list[nn.Module] = []
            for time_index in range(time_count):
                block = TransformerBlock(time_block_config)
                if global_last and time_index == time_count - 1:
                    time_implementations.append(
                        _GlobalTimeAttention(block, self.hidden_dim)
                    )
                else:
                    time_implementations.append(
                        _SlidingTimeAttention(block, self.hidden_dim)
                    )
            stages.append(
                _AxialAttentionStage(
                    camera_layers=[
                        TransformerBlock(camera_block_config)
                        for _ in range(camera_count)
                    ],
                    time_layers=time_implementations,
                    hidden_dim=self.hidden_dim,
                )
            )
        self.stages = nn.ModuleList(stages)

        self.final_norm = RMSNorm(self.hidden_dim)

        self.output_head = build_trajectory_output(
            input_dim=self.hidden_dim,
            dropout=dropout,
            predict_velocity=predict_velocity,
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
        ffn_type = config.ffn_type

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

    def forward(
        self,
        ball_uv: Tensor,
        ball_vis: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        padding_mask: Tensor,
    ) -> dict[str, Tensor]:
        """Forward pass for multi-view BLCS inputs."""
        batch_size, n_cams, seq_len_in = ball_uv.shape[:3]
        masks = build_axial_padding_masks(
            padding_mask,
            time_window_radius=self.time_window_radius,
        )
        court_kp = court_kp.masked_fill(~court_vis.unsqueeze(-1), 0.0)
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
        x = x * masks.context_valid.permute(0, 2, 1).unsqueeze(-1)

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
        for stage in self.stages:
            x = stage(
                x,
                masks.camera_attention_keep_mask,
                masks.time_attention_keep_mask,
                masks.sliding_attention_keep_mask,
                camera_freqs,
                time_freqs,
            )

        x = x[:, :, 0, :]
        x = self.final_norm(x)

        outputs = cast("dict[str, Tensor]", self.output_head(x))
        return mask_trajectory_outputs(outputs, masks.frame_valid)

    token_freqs_cis: Tensor
