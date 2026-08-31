"""Fixed-width mHC and hybrid-CSWA PLCS track-query model."""

from __future__ import annotations

from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.tasks.plcs.configuration import PLCSModelConfig
from src.tasks.plcs.data.tracking_types import PLCSTrackingPrediction
from src.tasks.plcs.models.components.heads import CanonicalPoseHead
from src.tasks.plcs.models.components.presence_competition import (
    DeepSetsPresenceResidual,
    build_presence_competition,
    decode_presence_logits,
)
from src.utils.models import (
    CSWAConfig,
    RMSNorm,
    RotaryFrequencyComputer,
    TransformerBlock,
    TransformerBlockConfig,
)
from src.utils.models.components.ffn_layers import FFNType
from src.utils.models.components.fixed_query_track_stage import FixedQueryTrackStage
from src.utils.models.components.mhc import (
    ManifoldConstrainedHyperConnection,
    MHCConfig,
)
from src.utils.models.embeddings import (
    CourtPlayerGroupEmbedding,
    InvisibleTokenEmbedding,
)
from src.utils.models.multiview_padding import build_fixed_query_padding_masks


def build_track_query_canonical_pose_head(
    config: PLCSModelConfig,
    *,
    hidden_dim: int,
    num_joints: int,
) -> CanonicalPoseHead | None:
    """Build the optional shared per-query canonical-pose readout."""
    if not bool(config.values.get("predict_canonical_pose", False)):
        return None
    return CanonicalPoseHead(
        input_dim=hidden_dim,
        hidden_dim=hidden_dim // 2,
        num_layers=2,
        dropout=config.number("dropout"),
        num_keypoints=num_joints,
    )


def decode_track_query_canonical_pose(
    head: CanonicalPoseHead | None,
    query_tokens: Tensor,
    *,
    frame_valid: Tensor,
) -> Tensor | None:
    """Decode and mask ``(B,T,Q,J,3)`` pose from shared query features."""
    if head is None:
        return None
    canonical_pose = head(query_tokens)
    return cast(
        "Tensor",
        canonical_pose * frame_valid[:, :, None, None, None],
    )


class PLCSTrackQueryModel(nn.Module):
    """Predict persistent player queries with a fixed ``C,C,C,G`` cycle."""

    def __init__(self, config: PLCSModelConfig) -> None:
        super().__init__()
        if config.name != "plcs_track_query":
            raise ValueError("PLCSTrackQueryModel requires plcs_track_query config.")
        mhc_config = config.track_query_mhc
        cswa_config = config.track_query_cswa
        if mhc_config is None or cswa_config is None:
            raise ValueError("PLCS track-query mhc and cswa config must be validated.")

        self.hidden_dim = config.integer("hidden_dim")
        self.num_heads = config.integer("num_heads")
        self.num_queries = config.integer("num_queries")
        self.num_stages = config.integer("num_stages")
        self.num_joints = config.integer("num_joints")
        self.role_rope_scale = int(config.boolean("role_rope_enabled"))
        if self.num_stages <= 0 or self.num_stages % 4 != 0:
            raise ValueError("num_stages must be a positive multiple of 4.")
        if self.num_queries <= 0:
            raise ValueError("num_queries must be positive.")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        head_dim = self.hidden_dim // self.num_heads
        self.rope_dim = config.integer("rope_dim")
        if self.rope_dim > head_dim or self.rope_dim % 2:
            raise ValueError(
                "rope_dim must be even and no larger than the attention head dim."
            )

        self.num_court_tokens = 14
        self.invisible_token = InvisibleTokenEmbedding(
            dim=self.hidden_dim,
            init_std=config.number("invisible_init_std"),
        )
        self.group_embed = CourtPlayerGroupEmbedding(
            dim=self.hidden_dim,
            invisible_token=self.invisible_token,
            num_court_tokens=self.num_court_tokens,
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
        rope_theta = config.number("rope_theta")
        self.spatial_frequency_computer = RotaryFrequencyComputer(
            dim=self.rope_dim,
            base=rope_theta,
            n_axes=3,
        )
        self.temporal_frequency_computer = RotaryFrequencyComputer(
            dim=self.rope_dim,
            base=rope_theta,
            n_axes=1,
        )
        self.output_norm = RMSNorm(self.hidden_dim)
        self.position_head = nn.Linear(self.hidden_dim, 3)
        self.rotation_head = nn.Linear(self.hidden_dim, 2)
        self.presence_head = nn.Linear(self.hidden_dim, 1)
        self.canonical_pose_head = build_track_query_canonical_pose_head(
            config,
            hidden_dim=self.hidden_dim,
            num_joints=self.num_joints,
        )
        self.presence_competition_mode = config.track_query_presence_competition
        if self.presence_competition_mode is None:
            raise ValueError(
                "PLCS track-query presence competition config must be validated."
            )
        presence_competition = build_presence_competition(
            self.presence_competition_mode,
            hidden_dim=self.hidden_dim,
        )
        if presence_competition is not None:
            self.presence_competition = presence_competition
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
        human_kp = cast(Tensor, args[0] if args else kwargs["human_kp"])
        human_vis = cast(
            Tensor,
            args[1] if len(args) > 1 else kwargs["human_vis"],
        )
        court_kp = cast(Tensor, args[2] if len(args) > 2 else kwargs["court_kp"])
        court_vis = cast(Tensor, args[3] if len(args) > 3 else kwargs["court_vis"])
        padding_mask = cast(
            Tensor, args[4] if len(args) > 4 else kwargs["padding_mask"]
        )
        if human_kp.ndim != 6:
            raise ValueError("human_kp must have shape (B,V,T,Q,J,2).")
        if human_kp.shape[3:5] != (self.num_queries, self.num_joints):
            raise ValueError(
                "human_kp query and joint axes must equal model.num_queries "
                "and model.num_joints."
            )
        if human_kp.shape[-1] != 2:
            raise ValueError("human_kp coordinates must have width 2.")
        if human_vis.shape != human_kp.shape[:-1]:
            raise ValueError("human_vis must match human_kp without UV.")
        batch_size, num_views, num_frames = human_kp.shape[:3]
        if court_kp.shape != (
            batch_size,
            num_views,
            num_frames,
            self.num_court_tokens,
            2,
        ):
            raise ValueError("court_kp must have shape (B,V,T,14,2).")
        if court_vis.shape != court_kp.shape[:-1]:
            raise ValueError("court_vis must match court_kp without UV.")
        if padding_mask.shape != (batch_size, num_views, num_frames):
            raise ValueError("padding_mask must have shape (B,V,T).")
        if any(
            tensor.dtype != torch.bool
            for tensor in (human_vis, court_vis, padding_mask)
        ):
            raise TypeError("human_vis, court_vis, and padding_mask must be boolean.")

    def _block_config(
        self,
        *,
        config: PLCSModelConfig,
        head_dim: int,
        temporal_cswa: bool,
    ) -> TransformerBlockConfig:
        attention_type: Literal["mha", "cswa"] = "cswa" if temporal_cswa else "mha"
        track_cswa = config.track_query_cswa
        if track_cswa is None:
            raise ValueError("PLCS track-query cswa config must be validated.")
        cswa = (
            CSWAConfig(
                dim=self.hidden_dim,
                n_heads=self.num_heads,
                head_dim=head_dim,
                rope_dim=self.rope_dim,
                attn_dropout=config.number("dropout"),
                compression_ratio=track_cswa.compression_ratio,
                window_radius=track_cswa.window_radius,
                backend=track_cswa.backend,
            )
            if temporal_cswa
            else None
        )
        return TransformerBlockConfig(
            dim=self.hidden_dim,
            n_heads=self.num_heads,
            ffn_dim=config.integer("ffn_dim"),
            head_dim=head_dim,
            rope_dim=self.rope_dim,
            attn_dropout=config.number("dropout"),
            attention_type=attention_type,
            n_kv_heads=None,
            rope_base=config.number("rope_theta"),
            ffn_type=cast(FFNType, config.string("ffn_type")),
            cswa=cswa,
        )

    def _build_stage(
        self,
        *,
        stage_index: int,
        config: PLCSModelConfig,
        head_dim: int,
    ) -> FixedQueryTrackStage:
        mhc_config = config.track_query_mhc
        if mhc_config is None:
            raise ValueError("PLCS track-query mhc config must be validated.")
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
        return FixedQueryTrackStage(
            stage_index=stage_index,
            mhc=ManifoldConstrainedHyperConnection(
                MHCConfig(
                    dim=self.hidden_dim,
                    num_streams=self.num_queries,
                    coefficient_dim=mhc_config.coefficient_dim,
                    sinkhorn_iters=mhc_config.sinkhorn_iters,
                    eps=mhc_config.eps,
                    residual_identity_bias=mhc_config.residual_identity_bias,
                    update_scale_init=mhc_config.update_scale_init,
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
        time = torch.arange(num_frames, device=device).view(1, num_frames, 1)
        slots = torch.zeros(
            batch_size, num_frames, num_queries, 3, device=device, dtype=torch.long
        )
        slots[..., 0] = time
        objects = torch.zeros(
            batch_size,
            num_frames,
            num_views,
            num_queries,
            3,
            device=device,
            dtype=torch.long,
        )
        objects[..., 0] = time.view(1, num_frames, 1, 1)
        objects[..., 1] = torch.arange(1, num_views + 1, device=device).view(
            1, 1, num_views, 1
        )
        objects[..., 2] = 1
        return torch.cat((slots, objects.flatten(2, 3)), dim=2).flatten(0, 1)

    def forward(
        self,
        human_kp: Tensor,
        human_vis: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        padding_mask: Tensor,
    ) -> PLCSTrackingPrediction:
        """Predict fixed-width tracks; only padding controls attention validity."""
        batch_size, num_views, num_frames = human_kp.shape[:3]
        coordinates = self.build_spatial_coordinates(
            batch_size=batch_size,
            num_frames=num_frames,
            num_views=num_views,
            num_detections=self.num_queries,
            num_queries=self.num_queries,
            device=human_kp.device,
        )
        rope_coordinates = coordinates.clone()
        rope_coordinates[..., 2] *= self.role_rope_scale
        return self._forward_with_spatial_coordinates(
            human_kp,
            human_vis,
            court_kp,
            court_vis,
            padding_mask,
            spatial_coordinates=rope_coordinates,
        )

    def _forward_with_spatial_coordinates(
        self,
        human_kp: Tensor,
        human_vis: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        padding_mask: Tensor,
        *,
        spatial_coordinates: Tensor,
    ) -> PLCSTrackingPrediction:
        """Execute the shared architecture with already validated coordinates."""
        batch_size, _num_views, num_frames = human_kp.shape[:3]
        masks = build_fixed_query_padding_masks(
            padding_mask,
            num_queries=self.num_queries,
        )

        effective_human_vis = human_vis & masks.context_valid[..., None, None]
        effective_court_vis = court_vis & masks.context_valid[..., None]
        observation_visible = effective_human_vis.any(dim=-1)
        masked_court = court_kp.masked_fill(~effective_court_vis.unsqueeze(-1), 0.0)
        court_for_queries = masked_court.unsqueeze(3).expand(
            -1, -1, -1, self.num_queries, -1, -1
        )
        masked_human = human_kp.masked_fill(
            ~effective_human_vis.unsqueeze(-1),
            0.0,
        )
        object_tokens = self.group_embed(
            court_for_queries,
            masked_human,
            observation_visible,
        )
        object_tokens = object_tokens * masks.object_state_valid.unsqueeze(-1)

        query_tokens = self.slot_embeddings.view(
            1, 1, self.num_queries, self.hidden_dim
        ).expand(batch_size, num_frames, -1, -1)
        query_tokens = query_tokens * masks.frame_valid[:, :, None, None]
        spatial_freqs = self.spatial_frequency_computer(spatial_coordinates)
        time_freqs = self.temporal_frequency_computer(
            torch.arange(num_frames, device=human_kp.device).unsqueeze(-1)
        )

        for stage in self.stages:
            object_tokens, query_tokens = stage(
                object_tokens,
                query_tokens,
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

        query_tokens = self.output_norm(query_tokens)
        output_valid = masks.frame_valid[:, :, None]
        position = self.position_head(query_tokens) * output_valid.unsqueeze(-1)
        rotation = F.normalize(self.rotation_head(query_tokens), dim=-1)
        rotation = rotation * output_valid.unsqueeze(-1)
        presence_competition = cast(
            "DeepSetsPresenceResidual | None",
            getattr(self, "presence_competition", None),
        )
        presence_logits = decode_presence_logits(
            self.presence_head,
            presence_competition,
            query_tokens,
            frame_valid=masks.frame_valid,
        )
        canonical_pose = decode_track_query_canonical_pose(
            self.canonical_pose_head,
            query_tokens,
            frame_valid=masks.frame_valid,
        )
        prediction: PLCSTrackingPrediction = {
            "position": position,
            "rotation": rotation,
            "presence_logits": presence_logits,
        }
        if canonical_pose is not None:
            prediction["canonical_pose"] = canonical_pose
        return prediction


__all__ = [
    "PLCSTrackQueryModel",
    "build_track_query_canonical_pose_head",
    "decode_track_query_canonical_pose",
]
