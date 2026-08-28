"""Fixed-width mHC and hybrid-CSWA BLCS track-query model."""

from __future__ import annotations

from src.utils.models.components.ffn_layers import FFNType

from typing import Literal, cast

import torch
from torch import Tensor, nn

from src.tasks.blcs.configuration import TrackQueryModelConfig
from src.tasks.blcs.data.tracking_types import BLCSTrackingPrediction
from src.tasks.blcs.models.components.observation_fusion import (
    LinearTrackObservationFusion,
)
from src.utils.models import (
    CSWAConfig,
    FixedQueryTrackStage,
    RMSNorm,
    RotaryFrequencyComputer,
    TransformerBlock,
    TransformerBlockConfig,
)
from src.utils.models.components.mhc import (
    ManifoldConstrainedHyperConnection,
    MHCConfig,
)
from src.utils.models.multiview_padding import build_fixed_query_padding_masks


class BLCSTrackQueryModel(nn.Module):
    """Predict persistent queries with a constructor-fixed ``C,C,C,G`` cycle."""

    def __init__(self, config: TrackQueryModelConfig) -> None:
        super().__init__()
        self.hidden_dim = int(config.hidden_dim)
        self.num_heads = int(config.num_heads)
        self.num_queries = int(config.num_queries)
        self.num_stages = int(config.num_stages)
        self.role_rope_scale = int(config.role_rope_enabled)
        if self.num_stages <= 0 or self.num_stages % 4 != 0:
            raise ValueError("num_stages must be a positive multiple of 4.")
        if self.num_queries <= 0:
            raise ValueError("num_queries must be positive.")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        head_dim = self.hidden_dim // self.num_heads
        self.rope_dim = config.rope_dim
        if self.rope_dim > head_dim or self.rope_dim % 2:
            raise ValueError(
                "rope_dim must be even and no larger than the attention head dim."
            )

        self.num_court_tokens = 14
        self.observation_encoder = LinearTrackObservationFusion(
            hidden_dim=self.hidden_dim,
            num_court_tokens=self.num_court_tokens,
            invisible_init_std=config.invisible_init_std,
        )

        self.slot_embeddings = nn.Parameter(
            torch.randn(self.num_queries, self.hidden_dim) * 0.02
        )
        self.stages = nn.ModuleList(
            [
                self._build_stage(
                    stage_index=stage_index,
                    config=config,
                    head_dim=head_dim,
                )
                for stage_index in range(self.num_stages)
            ]
        )
        self.spatial_frequency_computer = RotaryFrequencyComputer(
            dim=self.rope_dim,
            base=10000.0,
            n_axes=3,
        )
        self.temporal_frequency_computer = RotaryFrequencyComputer(
            dim=self.rope_dim,
            base=10000.0,
            n_axes=1,
        )
        self.output_norm = RMSNorm(self.hidden_dim)
        self.position_head = nn.Linear(self.hidden_dim, 3)
        self.presence_head = nn.Linear(self.hidden_dim, 1)
        self.register_forward_pre_hook(
            self._validate_forward_inputs,
            with_kwargs=True,
        )

    def _validate_forward_inputs(
        self,
        _module: nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        ball_uv = cast(Tensor, args[0] if args else kwargs["ball_uv"])
        ball_vis = cast(
            Tensor,
            args[1] if len(args) > 1 else kwargs["ball_vis"],
        )
        court_kp = cast(Tensor, args[2] if len(args) > 2 else kwargs["court_kp"])
        court_vis = cast(Tensor, args[3] if len(args) > 3 else kwargs["court_vis"])
        padding_mask = cast(
            Tensor, args[4] if len(args) > 4 else kwargs["padding_mask"]
        )
        if ball_uv.ndim != 5:
            raise ValueError("ball_uv must have shape (B,V,T,Q,2).")
        if ball_uv.shape[3] != self.num_queries:
            raise ValueError("ball_uv candidate width must equal model.num_queries.")
        if ball_vis.shape != ball_uv.shape[:-1]:
            raise ValueError("ball_vis must match ball_uv query axes.")
        batch_size, num_views, num_frames = ball_uv.shape[:3]
        if court_kp.shape != (batch_size, num_views, num_frames, 14, 2):
            raise ValueError("court_kp must have shape (B,V,T,14,2).")
        if court_vis.shape != court_kp.shape[:-1]:
            raise ValueError("court_vis must match court_kp without UV.")
        if padding_mask.shape != (batch_size, num_views, num_frames):
            raise ValueError("padding_mask must have shape (B,V,T).")
        if any(
            tensor.dtype != torch.bool for tensor in (ball_vis, court_vis, padding_mask)
        ):
            raise TypeError("ball_vis, court_vis, and padding_mask must be boolean.")

    def _block_config(
        self,
        *,
        config: TrackQueryModelConfig,
        head_dim: int,
        temporal_cswa: bool,
            ffn_type: FFNType = "swiglu",
) -> TransformerBlockConfig:
        attention_type: Literal["mha", "cswa"] = "cswa" if temporal_cswa else "mha"
        cswa_config = (
            CSWAConfig(
                dim=self.hidden_dim,
                n_heads=self.num_heads,
                head_dim=head_dim,
                rope_dim=self.rope_dim,
                attn_dropout=config.dropout,
                compression_ratio=config.cswa.compression_ratio,
                window_radius=config.cswa.window_radius,
                backend=config.cswa.backend,
            )
            if temporal_cswa
            else None
        )
        return TransformerBlockConfig(
            dim=self.hidden_dim,
            n_heads=self.num_heads,
            ffn_dim=config.ffn_dim,
            head_dim=head_dim,
            rope_dim=self.rope_dim,
            attn_dropout=config.dropout,
            attention_type=attention_type,
            n_kv_heads=None,
            rope_base=10000.0,
            ffn_type=config.ffn_type,
            cswa=cswa_config,
        )

    def _build_stage(
        self,
        *,
        stage_index: int,
        config: TrackQueryModelConfig,
        head_dim: int,
            ffn_type: FFNType = "swiglu",
) -> FixedQueryTrackStage:
        temporal_cswa = stage_index % 4 < 3
        temporal_config = self._block_config(
            config=config,
            head_dim=head_dim,
            temporal_cswa=temporal_cswa,
                    ffn_type=ffn_type,
)
        spatial_config = self._block_config(
            config=config,
            head_dim=head_dim,
            temporal_cswa=False,
                    ffn_type=ffn_type,
)
        return FixedQueryTrackStage(
            stage_index=stage_index,
            mhc=ManifoldConstrainedHyperConnection(
                MHCConfig(
                    dim=self.hidden_dim,
                    num_streams=self.num_queries,
                    coefficient_dim=config.mhc.coefficient_dim,
                    sinkhorn_iters=config.mhc.sinkhorn_iters,
                    eps=config.mhc.eps,
                    residual_identity_bias=config.mhc.residual_identity_bias,
                    update_scale_init=config.mhc.update_scale_init,
                )
            ),
            object_temporal_block=TransformerBlock(temporal_config),
            spatial_block=TransformerBlock(spatial_config),
            query_temporal_block=TransformerBlock(temporal_config),
            hidden_dim=self.hidden_dim,
            num_queries=self.num_queries,
        )

    @staticmethod
    def build_spatial_coordinates(
        *,
        batch_size: int,
        num_frames: int,
        num_views: int,
        num_detections: int,
        num_queries: int,
        device: torch.device,
    ) -> Tensor:
        """Return ``(B*T,Q+V*Q,3)`` time/camera/role coordinates."""
        if num_detections != num_queries:
            raise ValueError("num_detections must equal num_queries.")
        return BLCSTrackQueryModel._build_spatial_coordinates(
            batch_size=batch_size,
            num_frames=num_frames,
            num_views=num_views,
            num_queries=num_queries,
            device=device,
        )

    @staticmethod
    def _build_spatial_coordinates(
        *,
        batch_size: int,
        num_frames: int,
        num_views: int,
        num_queries: int,
        device: torch.device,
    ) -> Tensor:
        """Compute validated ``(B*T,Q+V*Q,3)`` spatial coordinates."""
        time = torch.arange(num_frames, device=device).view(1, num_frames, 1)
        slot = torch.zeros(
            batch_size, num_frames, num_queries, 3, device=device, dtype=torch.long
        )
        slot[..., 0] = time
        camera_tokens = torch.zeros(
            batch_size,
            num_frames,
            num_views,
            num_queries,
            3,
            device=device,
            dtype=torch.long,
        )
        camera_tokens[..., 0] = time.view(1, num_frames, 1, 1)
        camera = torch.arange(1, num_views + 1, device=device).view(1, 1, num_views, 1)
        camera_tokens[..., 1] = camera
        camera_tokens[..., 2] = 1
        return torch.cat([slot, camera_tokens.flatten(2, 3)], dim=2).flatten(0, 1)

    def forward(
        self,
        ball_uv: Tensor,
        ball_vis: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        padding_mask: Tensor,
    ) -> BLCSTrackingPrediction:
        """Predict fixed-width tracks; only padding controls attention validity."""
        batch_size, num_views, num_frames, _, _ = ball_uv.shape
        coordinates = self._build_spatial_coordinates(
            batch_size=batch_size,
            num_frames=num_frames,
            num_views=num_views,
            num_queries=self.num_queries,
            device=ball_uv.device,
        )
        rope_coordinates = coordinates.clone()
        rope_coordinates[..., 2] *= self.role_rope_scale
        return self._forward_with_spatial_coordinates(
            ball_uv,
            ball_vis,
            court_kp,
            court_vis,
            padding_mask,
            spatial_coordinates=rope_coordinates,
        )

    def _forward_with_spatial_coordinates(
        self,
        ball_uv: Tensor,
        ball_vis: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        padding_mask: Tensor,
        *,
        spatial_coordinates: Tensor,
    ) -> BLCSTrackingPrediction:
        """Execute the shared architecture with validated spatial coordinates."""
        batch_size, _num_views, num_frames, _, _ = ball_uv.shape
        masks = build_fixed_query_padding_masks(
            padding_mask,
            num_queries=self.num_queries,
        )
        context_valid = masks.context_valid
        effective_ball_uv = ball_uv.masked_fill(
            ~context_valid.unsqueeze(-1).unsqueeze(-1),
            0.0,
        )
        effective_ball_vis = ball_vis & context_valid.unsqueeze(-1)
        effective_court_kp = court_kp.masked_fill(
            ~context_valid.unsqueeze(-1).unsqueeze(-1),
            0.0,
        )
        effective_court_vis = court_vis & context_valid.unsqueeze(-1)

        time_major_tokens = self.observation_encoder(
            effective_court_kp,
            effective_court_vis,
            effective_ball_uv,
            effective_ball_vis,
        )
        camera_tokens = time_major_tokens.permute(0, 2, 1, 3, 4)
        camera_tokens = camera_tokens * masks.object_state_valid.unsqueeze(-1)

        slots = self.slot_embeddings.view(
            1, 1, self.num_queries, self.hidden_dim
        ).expand(batch_size, num_frames, -1, -1)
        slots = slots * masks.frame_valid[:, :, None, None]
        spatial_freqs = self.spatial_frequency_computer(spatial_coordinates)
        time_freqs = self.temporal_frequency_computer(
            torch.arange(num_frames, device=ball_uv.device).unsqueeze(-1)
        )

        for stage in self.stages:
            camera_tokens, slots = stage(
                camera_tokens,
                slots,
                object_state_valid=masks.object_state_valid,
                frame_valid=masks.frame_valid,
                spatial_attention_keep_mask=masks.spatial_attention_keep_mask,
                object_temporal_state_valid=masks.object_temporal_state_valid,
                object_temporal_attention_keep_mask=(
                    masks.object_temporal_attention_keep_mask
                ),
                query_temporal_state_valid=masks.query_temporal_state_valid,
                query_temporal_attention_keep_mask=(
                    masks.query_temporal_attention_keep_mask
                ),
                spatial_freqs=spatial_freqs,
                time_freqs=time_freqs,
            )

        slots = self.output_norm(slots)
        position = self.position_head(slots) * masks.frame_valid[:, :, None, None]
        presence_logits = self.presence_head(slots).squeeze(-1)
        presence_logits = presence_logits * masks.frame_valid[:, :, None]
        return {
            "position": position,
            "presence_logits": presence_logits,
        }


__all__ = ["BLCSTrackQueryModel"]
