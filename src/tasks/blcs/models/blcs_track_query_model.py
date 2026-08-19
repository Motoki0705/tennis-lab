"""Fixed-width mHC and hybrid-CSWA BLCS track-query model."""

from __future__ import annotations

from typing import Literal, cast

import torch
from torch import Tensor, nn

from src.tasks.blcs.configuration import TrackQueryModelConfig
from src.tasks.blcs.data.tracking_types import BLCSTrackingPrediction
from src.tasks.blcs.models.components.observation_fusion import (
    LinearTrackObservationFusion,
    PointAttentionTrackObservationFusion,
)
from src.tasks.blcs.models.components.track_query_stage import BLCSTrackQueryStage
from src.utils.models import (
    CSWAConfig,
    RMSNorm,
    RotaryFrequencyComputer,
    TransformerBlock,
    TransformerBlockConfig,
)
from src.utils.models.components.mhc import (
    ManifoldConstrainedHyperConnection,
    MHCConfig,
)


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
        self.observation_encoder: (
            LinearTrackObservationFusion | PointAttentionTrackObservationFusion
        )
        if config.observation_fusion == "linear":
            self.observation_encoder = LinearTrackObservationFusion(
                hidden_dim=self.hidden_dim,
                num_court_tokens=self.num_court_tokens,
                invisible_init_std=config.invisible_init_std,
            )
        elif config.observation_fusion == "point_attention":
            if config.point_fusion is None:
                raise ValueError(
                    "model.point_fusion is required when observation_fusion="
                    "'point_attention'."
                )
            self.observation_encoder = PointAttentionTrackObservationFusion(
                hidden_dim=self.hidden_dim,
                num_court_tokens=self.num_court_tokens,
                config=config.point_fusion,
                invisible_init_std=config.invisible_init_std,
            )
        else:
            raise ValueError(
                "observation_fusion must be 'linear' or 'point_attention', got "
                f"{config.observation_fusion!r}."
            )
        self.observation_encoder.register_forward_hook(
            self._validate_observation_output,
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
        ball_visible = cast(
            Tensor,
            args[1] if len(args) > 1 else kwargs["ball_visible"],
        )
        candidate_mask = cast(
            Tensor,
            args[2] if len(args) > 2 else kwargs["candidate_mask"],
        )
        if ball_uv.ndim != 5:
            raise ValueError("ball_uv must have shape (B,V,T,Q,2).")
        if ball_uv.shape[3] != self.num_queries:
            raise ValueError("ball_uv candidate width must equal model.num_queries.")
        if ball_visible.shape != ball_uv.shape[:-1]:
            raise ValueError("ball_visible must match ball_uv candidate axes.")
        if candidate_mask.shape != ball_visible.shape:
            raise ValueError("candidate_mask must match ball_visible.")
        if bool((ball_visible & ~candidate_mask).any()):
            raise ValueError("ball_visible implies candidate_mask.")

    @staticmethod
    def _validate_observation_output(
        _module: nn.Module,
        args: tuple[object, ...],
        output: object,
    ) -> None:
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError(
                "observation encoder must return candidate tokens and a camera mask."
            )
        state_valid = cast(Tensor, args[4])
        time_major_valid = cast(Tensor, output[1])
        expected_time_major = state_valid.permute(0, 2, 1, 3)
        if time_major_valid.shape != expected_time_major.shape:
            raise RuntimeError("observation encoder returned an invalid camera mask shape.")

    def _block_config(
        self,
        *,
        config: TrackQueryModelConfig,
        head_dim: int,
        temporal_cswa: bool,
    ) -> TransformerBlockConfig:
        attention_type: Literal["mha", "cswa"] = (
            "cswa" if temporal_cswa else "mha"
        )
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
            ffn_type="swiglu",
            cswa=cswa_config,
        )

    def _build_stage(
        self,
        *,
        stage_index: int,
        config: TrackQueryModelConfig,
        head_dim: int,
    ) -> BLCSTrackQueryStage:
        temporal_cswa = stage_index % 4 < 3
        temporal_config = self._block_config(
            config=config,
            head_dim=head_dim,
            temporal_cswa=temporal_cswa,
        )
        spatial_config = self._block_config(
            config=config,
            head_dim=head_dim,
            temporal_cswa=False,
        )
        return BLCSTrackQueryStage(
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
        camera = torch.arange(1, num_views + 1, device=device).view(
            1, 1, num_views, 1
        )
        camera_tokens[..., 1] = camera
        camera_tokens[..., 2] = 1
        return torch.cat([slot, camera_tokens.flatten(2, 3)], dim=2).flatten(0, 1)

    def forward(
        self,
        ball_uv: Tensor,
        ball_visible: Tensor,
        candidate_mask: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        frame_mask: Tensor,
        camera_state_valid: Tensor,
        spatial_attention_mask: Tensor,
        object_temporal_state_valid: Tensor,
        object_temporal_attention_mask: Tensor,
        query_temporal_state_valid: Tensor,
        query_temporal_attention_mask: Tensor,
        point_attention_mask: Tensor,
    ) -> BLCSTrackingPrediction:
        """Predict fixed-width clip-local ball tracks from prepared contracts."""
        batch_size, num_views, num_frames, _, _ = ball_uv.shape

        time_major_tokens, time_major_valid = self.observation_encoder(
            court_kp,
            court_vis,
            ball_uv,
            ball_visible,
            camera_state_valid,
            point_attention_mask,
        )
        del time_major_valid
        camera_tokens = time_major_tokens.permute(0, 2, 1, 3, 4)
        camera_tokens = camera_tokens * camera_state_valid.unsqueeze(-1)

        slots = self.slot_embeddings.view(
            1, 1, self.num_queries, self.hidden_dim
        ).expand(batch_size, num_frames, -1, -1)
        slots = slots * frame_mask[:, :, None, None]
        coordinates = self._build_spatial_coordinates(
            batch_size=batch_size,
            num_frames=num_frames,
            num_views=num_views,
            num_queries=self.num_queries,
            device=ball_uv.device,
        )
        rope_coordinates = coordinates.clone()
        rope_coordinates[..., 2] *= self.role_rope_scale
        spatial_freqs = self.spatial_frequency_computer(rope_coordinates)
        time_freqs = self.temporal_frequency_computer(
            torch.arange(num_frames, device=ball_uv.device).unsqueeze(-1)
        )

        for stage in self.stages:
            camera_tokens, slots = stage(
                camera_tokens,
                slots,
                camera_state_valid=camera_state_valid,
                frame_mask=frame_mask,
                spatial_attention_mask=spatial_attention_mask,
                object_temporal_state_valid=object_temporal_state_valid,
                object_temporal_attention_mask=object_temporal_attention_mask,
                query_temporal_state_valid=query_temporal_state_valid,
                query_temporal_attention_mask=query_temporal_attention_mask,
                spatial_freqs=spatial_freqs,
                time_freqs=time_freqs,
            )

        slots = self.output_norm(slots)
        position = self.position_head(slots) * frame_mask[:, :, None, None]
        presence_logits = self.presence_head(slots).squeeze(-1)
        presence_logits = presence_logits * frame_mask[:, :, None]
        return {
            "position": position,
            "presence_logits": presence_logits,
        }


__all__ = ["BLCSTrackQueryModel"]
