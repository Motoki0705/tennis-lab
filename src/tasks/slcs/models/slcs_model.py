"""SLCS multimodal temporal fusion model.

Architecture
------------
Per frame ``t`` the model builds ``E = P + 1`` entity tokens from the 2D
observations (each conditioned on the same court keypoints):

- ``P`` player tokens: court + player-pose group embedding
  (:class:`CourtPlayerGroupEmbedding`), invisible frames replaced by a learned
  invisible token,
- 1 ball token: court + ball-UV group embedding
  (:class:`CourtBallGroupEmbedding`).

The token grid ``(T, E)`` is processed by alternating axial self-attention:
entity-axis attention (players and the ball exchange information within a
frame) and time-axis attention (each entity aggregates its own history), with
interleaved 2-axis RoPE (time, entity) — the same mechanism as the
BLCS/PLCS ``*MultiViewAxialModel`` family with the camera axis reinterpreted
as an entity axis.

DINOv3 patch tokens arrive only for sparsely sampled frames. Their spatial grid
can be bilinearly reduced before channel projection, then they are fused by
explicit cross-attention every ``dino_cross_attn_every`` axial blocks: queries
are all entity tokens (RoPE time position = window frame index), keys/values
are all patch tokens of all sampled frames (RoPE time position = the *actual*
sampled frame index from ``dino_frame_idx``, entity-axis position = a
dedicated visual-stream slot ``E``). Temporal propagation of visual evidence
is therefore handled by attention with true time offsets — never by implicit
interpolation. Windows without any valid patch token skip the visual pathway
explicitly.

The axial depth is configurable as shared, position-specific, and
rotation-specific layers. Setting both task-specific depths to zero is the
original shared architecture. Setting the shared depth to zero isolates the
position branch (player + ball position) from the player-rotation branch after
the common observation embeddings.

Outputs (normalized court coordinates, ``(cos, sin)`` yaw) with per-frame
Laplace log-scales as aleatoric uncertainty:

- ``player_position (B, P, T, 3)``, ``player_position_log_b (B, P, T)``
- ``player_rotation (B, P, T, 2)``, ``player_rotation_log_b (B, P, T)``
- ``ball_position (B, T, 3)``, ``ball_position_log_b (B, T)``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor, nn

from src.tasks.slcs.models.components.dino_adapter import DinoTokenEncoder
from src.tasks.slcs.models.components.heads import (
    BallPositionHead,
    LogScaleHead,
    PlayerPositionHead,
    PlayerRotationHead,
)
from src.utils.models import (
    CrossAttnBlock,
    CrossAttnBlockConfig,
    RMSNorm,
    RotaryFrequencyComputer,
    TransformerBlock,
    TransformerBlockConfig,
    validate_rope_dim,
)
from src.utils.models.axial_multiview_mixin import AxialMultiViewMixin
from src.utils.models.embeddings import (
    CourtBallGroupEmbedding,
    CourtPlayerGroupEmbedding,
    InvisibleTokenEmbedding,
)
from src.utils.models.transformer_utils import resolve_axial_rope_bases
from src.utils.schema.player import NUM_HUMAN_KP

if TYPE_CHECKING:
    from src.tasks.slcs.configuration import SLCSDataRuntimeConfig, SLCSModelConfig


class _DinoCrossUpdate(nn.Module):
    """Apply one configured visual update and mask samples without evidence."""

    def __init__(self, block: CrossAttnBlock) -> None:
        super().__init__()
        self.block = block

    def forward(
        self,
        q: Tensor,
        kv: Tensor,
        *,
        attn_mask: Tensor,
        freqs_q_cis: Tensor,
        freqs_k_cis: Tensor,
        batch_has_dino: Tensor,
    ) -> Tensor:
        """Compute cross-attention for already validated tensor inputs."""
        updated = self.block(
            q,
            kv,
            attn_mask=attn_mask,
            freqs_q_cis=freqs_q_cis,
            freqs_k_cis=freqs_k_cis,
        )
        return torch.where(batch_has_dino[:, None, None], updated, q)


class _NoDinoCrossUpdate(nn.Module):
    """Resolved axial stage that has no configured visual update."""

    def forward(
        self,
        q: Tensor,
        kv: Tensor,
        *,
        attn_mask: Tensor,
        freqs_q_cis: Tensor,
        freqs_k_cis: Tensor,
        batch_has_dino: Tensor,
    ) -> Tensor:
        """Return the axial result unchanged for this configured stage."""
        return q


class SLCSFusionModel(AxialMultiViewMixin, nn.Module):
    """Entity x time axial transformer with sparse DINOv3 cross-attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_shared_layers: int,
        num_position_layers: int,
        num_rotation_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        rope_dim: int,
        rope_theta_time: float,
        rope_theta_entity: float,
        attention_type: Literal["mha"],
        ffn_type: Literal["swiglu", "mlp"],
        num_players: int,
        num_court_kp: int,
        max_seq_len: int,
        invisible_init_std: float,
        dino_embed_dim: int,
        dino_grid_h: int,
        dino_grid_w: int,
        dino_patch_downsample_factor: int,
        dino_cross_attn_every: int,
        log_b_min: float,
        log_b_max: float,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.num_players = int(num_players)
        self.num_court_kp = int(num_court_kp)
        self.max_seq_len = int(max_seq_len)
        self.num_entities = self.num_players + 1
        self.num_shared_layers = int(num_shared_layers)
        self.num_position_layers = int(num_position_layers)
        self.num_rotation_layers = int(num_rotation_layers)
        self.dino_cross_attn_every = int(dino_cross_attn_every)
        self.log_b_min = float(log_b_min)
        self.log_b_max = float(log_b_max)

        if self.num_players <= 0:
            raise ValueError(f"num_players must be positive, got {num_players}.")
        if self.dino_cross_attn_every <= 0:
            raise ValueError(
                f"dino_cross_attn_every must be positive, got {dino_cross_attn_every}."
            )
        if self.log_b_min >= self.log_b_max:
            raise ValueError(
                f"log_b_min={log_b_min} must be smaller than log_b_max={log_b_max}."
            )
        self._validate_init_args(
            hidden_dim=self.hidden_dim,
            num_heads=num_heads,
            num_shared_layers=self.num_shared_layers,
            num_position_layers=self.num_position_layers,
            num_rotation_layers=self.num_rotation_layers,
            max_seq_len=self.max_seq_len,
        )

        if ffn_dim <= 0:
            raise ValueError(f"ffn_dim must be positive, got {ffn_dim}.")
        if attention_type != "mha":
            raise ValueError(
                "SLCSFusionModel supports only the canonical MHA architecture; "
                f"got {attention_type!r}."
            )

        head_dim = self.hidden_dim // num_heads
        validate_rope_dim(rope_dim=rope_dim, head_dim=head_dim)
        self.head_dim = int(head_dim)
        self.rope_dim = int(rope_dim)
        self.rope_bases = resolve_axial_rope_bases(
            rope_theta_time=rope_theta_time,
            rope_theta_camera=rope_theta_entity,
        )

        # ---- Observation embeddings -----------------------------------
        self.invisible_token = InvisibleTokenEmbedding(
            dim=self.hidden_dim, init_std=invisible_init_std
        )
        self.player_embed = CourtPlayerGroupEmbedding(
            dim=self.hidden_dim,
            invisible_token=self.invisible_token,
            num_court_tokens=self.num_court_kp,
        )
        self.ball_embed = CourtBallGroupEmbedding(
            dim=self.hidden_dim,
            invisible_token=self.invisible_token,
            num_court_tokens=self.num_court_kp,
        )
        self.entity_embed = nn.Embedding(self.num_entities, self.hidden_dim)

        # ---- DINOv3 visual stream --------------------------------------
        self.dino_encoder = DinoTokenEncoder(
            input_dim=int(dino_embed_dim),
            dim=self.hidden_dim,
            grid_h=int(dino_grid_h),
            grid_w=int(dino_grid_w),
            downsample_factor=int(dino_patch_downsample_factor),
        )

        # ---- Axial trunk with interleaved cross-attention --------------
        def block(*, rope_base: float) -> TransformerBlock:
            return TransformerBlock(
                TransformerBlockConfig(
                    dim=self.hidden_dim,
                    n_heads=num_heads,
                    ffn_dim=ffn_dim,
                    head_dim=head_dim,
                    rope_dim=self.rope_dim,
                    attn_dropout=dropout,
                    attention_type=attention_type,
                    n_kv_heads=None,
                    rope_base=rope_base,
                    ffn_type=ffn_type,
                )
            )

        def cross_layers(depth: int) -> nn.ModuleList:
            return nn.ModuleList(
                [
                    _DinoCrossUpdate(
                        CrossAttnBlock(
                            CrossAttnBlockConfig(
                                dim=self.hidden_dim,
                                n_heads=num_heads,
                                ffn_dim=ffn_dim,
                                head_dim=head_dim,
                                rope_dim=self.rope_dim,
                                attn_dropout=dropout,
                                ffn_type=ffn_type,
                            )
                        )
                    )
                    if (layer_idx + 1) % self.dino_cross_attn_every == 0
                    else _NoDinoCrossUpdate()
                    for layer_idx in range(depth)
                ]
            )

        self.entity_layers = nn.ModuleList(
            [block(rope_base=self.rope_bases[1]) for _ in range(self.num_shared_layers)]
        )
        self.time_layers = nn.ModuleList(
            [block(rope_base=self.rope_bases[0]) for _ in range(self.num_shared_layers)]
        )
        self.dino_cross_layers = cross_layers(self.num_shared_layers)
        self.position_entity_layers = nn.ModuleList(
            [
                block(rope_base=self.rope_bases[1])
                for _ in range(self.num_position_layers)
            ]
        )
        self.position_time_layers = nn.ModuleList(
            [
                block(rope_base=self.rope_bases[0])
                for _ in range(self.num_position_layers)
            ]
        )
        self.position_dino_cross_layers = cross_layers(self.num_position_layers)
        self.rotation_entity_layers = nn.ModuleList(
            [
                block(rope_base=self.rope_bases[1])
                for _ in range(self.num_rotation_layers)
            ]
        )
        self.rotation_time_layers = nn.ModuleList(
            [
                block(rope_base=self.rope_bases[0])
                for _ in range(self.num_rotation_layers)
            ]
        )
        self.rotation_dino_cross_layers = cross_layers(self.num_rotation_layers)

        all_shared = self.num_position_layers == 0 and self.num_rotation_layers == 0
        self.final_norm: nn.Module = (
            RMSNorm(self.hidden_dim) if all_shared else nn.Identity()
        )
        self.position_final_norm: nn.Module = (
            nn.Identity() if all_shared else RMSNorm(self.hidden_dim)
        )
        self.rotation_final_norm: nn.Module = (
            nn.Identity() if all_shared else RMSNorm(self.hidden_dim)
        )

        # ---- Heads ------------------------------------------------------
        head_hidden = self.hidden_dim // 2
        self.player_position_head = PlayerPositionHead(
            input_dim=self.hidden_dim,
            hidden_dim=head_hidden,
            num_layers=2,
            dropout=dropout,
        )
        self.player_rotation_head = PlayerRotationHead(
            input_dim=self.hidden_dim,
            hidden_dim=head_hidden,
            num_layers=2,
            dropout=dropout,
        )
        self.ball_position_head = BallPositionHead(
            input_dim=self.hidden_dim,
            hidden_dim=head_hidden,
            num_layers=2,
            dropout=dropout,
        )
        self.player_position_scale_head = LogScaleHead(
            input_dim=self.hidden_dim,
            hidden_dim=head_hidden // 2,
            num_layers=1,
            dropout=dropout,
        )
        self.player_rotation_scale_head = LogScaleHead(
            input_dim=self.hidden_dim,
            hidden_dim=head_hidden // 2,
            num_layers=1,
            dropout=dropout,
        )
        self.ball_position_scale_head = LogScaleHead(
            input_dim=self.hidden_dim,
            hidden_dim=head_hidden // 2,
            num_layers=1,
            dropout=dropout,
        )

        # RoPE table over (time, entity-or-visual-stream) positions. Time axis
        # holds positions 1..max_seq_len (mixin convention); the extra entity
        # slot ``num_entities`` is reserved for DINOv3 keys.
        self.token_rope = RotaryFrequencyComputer(
            dim=self.rope_dim,
            base=self.rope_bases,
            n_axes=2,
        )
        token_freqs = self.token_rope(
            self._build_token_positions(
                seq_len=self.max_seq_len,
                n_cams=self.num_entities + 1,
            )
        )
        self.register_buffer("token_freqs_cis", token_freqs, persistent=False)

    @staticmethod
    def _validate_init_args(
        *,
        hidden_dim: int,
        num_heads: int,
        num_shared_layers: int,
        num_position_layers: int,
        num_rotation_layers: int,
        max_seq_len: int,
    ) -> None:
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        depths = (num_shared_layers, num_position_layers, num_rotation_layers)
        if any(depth < 0 for depth in depths):
            raise ValueError(f"trunk layer counts must be non-negative, got {depths}.")
        if num_shared_layers + num_position_layers <= 0:
            raise ValueError("position path must contain at least one trunk layer.")
        if num_shared_layers + num_rotation_layers <= 0:
            raise ValueError("rotation path must contain at least one trunk layer.")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

    @classmethod
    def from_config(
        cls, model: SLCSModelConfig, data: SLCSDataRuntimeConfig
    ) -> SLCSFusionModel:
        """Create the model from the validated canonical model/data contract."""
        dino = data.pipeline.dino_spec
        return cls(
            hidden_dim=model.hidden_dim,
            num_shared_layers=model.num_shared_layers,
            num_position_layers=model.num_position_layers,
            num_rotation_layers=model.num_rotation_layers,
            num_heads=model.num_heads,
            ffn_dim=model.ffn_dim,
            dropout=model.dropout,
            rope_dim=model.rope_dim,
            rope_theta_time=model.rope_theta_time,
            rope_theta_entity=model.rope_theta_entity,
            attention_type=model.attention_type,
            ffn_type=model.ffn_type,
            num_players=data.pipeline.num_players,
            num_court_kp=data.pipeline.num_court_kp,
            max_seq_len=data.pipeline.window_size,
            invisible_init_std=model.invisible_init_std,
            dino_embed_dim=dino.embed_dim,
            dino_grid_h=dino.grid_h,
            dino_grid_w=dino.grid_w,
            dino_patch_downsample_factor=model.dino_patch_downsample_factor,
            dino_cross_attn_every=model.dino_cross_attn_every,
            log_b_min=model.log_b_min,
            log_b_max=model.log_b_max,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        *,
        player_kp: Tensor,
        player_kp_vis: Tensor,
        player_valid: Tensor,
        ball_uv: Tensor,
        ball_vis: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        frame_mask: Tensor,
        entity_attn_mask: Tensor,
        time_attn_mask: Tensor,
        dino_tokens: Tensor,
        dino_frame_idx: Tensor,
        dino_attn_mask: Tensor,
        dino_batch_has_evidence: Tensor,
    ) -> dict[str, Tensor]:
        """Run the fusion model on one batch of single-camera windows.

        Shapes are documented in :class:`src.tasks.slcs.data.types.SLCSBatch`
        (without the leading batch axis for per-sample tensors).
        """
        batch_size, _, seq_len = player_kp.shape[:3]
        num_players = self.num_players
        num_entities = self.num_entities

        # Zero-out invisible observations before embedding (PLCS convention).
        player_kp = player_kp * (player_kp_vis > 0).unsqueeze(-1).to(player_kp.dtype)
        ball_uv = ball_uv * (ball_vis > 0).unsqueeze(-1).to(ball_uv.dtype)
        court_kp = court_kp * (court_vis > 0).unsqueeze(-1).to(court_kp.dtype)

        frame_valid = frame_mask > 0

        # ---- Entity tokens (B, T, E, D) --------------------------------
        court_for_players = (
            court_kp.unsqueeze(1)
            .expand(batch_size, num_players, seq_len, self.num_court_kp, 2)
            .reshape(batch_size * num_players * seq_len, self.num_court_kp, 2)
        )
        player_flat = player_kp.reshape(
            batch_size * num_players * seq_len, NUM_HUMAN_KP, 2
        )
        player_token_valid = (player_valid > 0) & frame_valid.unsqueeze(1)  # (B, P, T)
        player_tokens = self.player_embed(
            court_for_players,
            player_flat,
            player_token_valid.reshape(batch_size * num_players * seq_len),
        ).reshape(batch_size, num_players, seq_len, self.hidden_dim)

        ball_token_valid = (ball_vis > 0) & frame_valid  # (B, T)
        ball_tokens = self.ball_embed(
            court_kp.reshape(batch_size * seq_len, self.num_court_kp, 2),
            ball_uv.reshape(batch_size * seq_len, 2),
            ball_token_valid.reshape(batch_size * seq_len),
        ).reshape(batch_size, 1, seq_len, self.hidden_dim)

        x = torch.cat([player_tokens, ball_tokens], dim=1)  # (B, E, T, D)
        entity_ids = torch.arange(num_entities, device=x.device)
        x = x + self.entity_embed(entity_ids)[None, :, None, :]
        x = x.permute(0, 2, 1, 3)  # (B, T, E, D)

        entity_freqs = self._camera_freqs(
            batch_size=batch_size, seq_len=seq_len, n_cams=num_entities
        )
        time_freqs = self._time_freqs(
            batch_size=batch_size, seq_len=seq_len, n_cams=num_entities
        )

        dino_ctx = self._encode_dino(
            dino_tokens=dino_tokens,
            dino_frame_idx=dino_frame_idx,
            dino_attn_mask=dino_attn_mask,
            dino_batch_has_evidence=dino_batch_has_evidence,
            batch_size=batch_size,
        )

        x = self._run_axial_trunk(
            x,
            self.entity_layers,
            self.time_layers,
            self.dino_cross_layers,
            batch_size=batch_size,
            seq_len=seq_len,
            num_entities=num_entities,
            entity_freqs=entity_freqs,
            time_freqs=time_freqs,
            entity_mask=entity_attn_mask,
            time_mask=time_attn_mask,
            dino_ctx=dino_ctx,
        )
        position_x = self._run_axial_trunk(
            x,
            self.position_entity_layers,
            self.position_time_layers,
            self.position_dino_cross_layers,
            batch_size=batch_size,
            seq_len=seq_len,
            num_entities=num_entities,
            entity_freqs=entity_freqs,
            time_freqs=time_freqs,
            entity_mask=entity_attn_mask,
            time_mask=time_attn_mask,
            dino_ctx=dino_ctx,
        )
        rotation_x = self._run_axial_trunk(
            x,
            self.rotation_entity_layers,
            self.rotation_time_layers,
            self.rotation_dino_cross_layers,
            batch_size=batch_size,
            seq_len=seq_len,
            num_entities=num_entities,
            entity_freqs=entity_freqs,
            time_freqs=time_freqs,
            entity_mask=entity_attn_mask,
            time_mask=time_attn_mask,
            dino_ctx=dino_ctx,
        )

        position_x = self.position_final_norm(self.final_norm(position_x))
        rotation_x = self.rotation_final_norm(self.final_norm(rotation_x))

        position_player_feat = position_x[:, :, :num_players, :].permute(0, 2, 1, 3)
        rotation_player_feat = rotation_x[:, :, :num_players, :].permute(0, 2, 1, 3)
        ball_feat = position_x[:, :, num_players, :]

        return {
            "player_position": self.player_position_head(position_player_feat),
            "player_rotation": self.player_rotation_head(rotation_player_feat),
            "player_position_log_b": self._clamp_log_b(
                self.player_position_scale_head(position_player_feat).squeeze(-1)
            ),
            "player_rotation_log_b": self._clamp_log_b(
                self.player_rotation_scale_head(rotation_player_feat).squeeze(-1)
            ),
            "ball_position": self.ball_position_head(ball_feat),
            "ball_position_log_b": self._clamp_log_b(
                self.ball_position_scale_head(ball_feat).squeeze(-1)
            ),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_axial_trunk(
        self,
        x: Tensor,
        entity_layers: nn.ModuleList,
        time_layers: nn.ModuleList,
        dino_cross_layers: nn.ModuleList,
        *,
        batch_size: int,
        seq_len: int,
        num_entities: int,
        entity_freqs: Tensor,
        time_freqs: Tensor,
        entity_mask: Tensor,
        time_mask: Tensor,
        dino_ctx: tuple[Tensor, Tensor, Tensor, Tensor],
    ) -> Tensor:
        """Run one shared or task-specific axial trunk."""
        kv, dino_attn_mask, k_freqs, batch_has_dino = dino_ctx
        for entity_layer, time_layer, dino_cross_layer in zip(
            entity_layers, time_layers, dino_cross_layers, strict=True
        ):
            x_entity = x.reshape(batch_size * seq_len, num_entities, self.hidden_dim)
            x_entity = entity_layer(
                x_entity, freqs_cis=entity_freqs, attn_mask=entity_mask
            )
            x = x_entity.reshape(batch_size, seq_len, num_entities, self.hidden_dim)

            x_time = x.permute(0, 2, 1, 3).reshape(
                batch_size * num_entities, seq_len, self.hidden_dim
            )
            x_time = time_layer(x_time, freqs_cis=time_freqs, attn_mask=time_mask)
            x = x_time.reshape(
                batch_size, num_entities, seq_len, self.hidden_dim
            ).permute(0, 2, 1, 3)

            q = x.reshape(batch_size, seq_len * num_entities, self.hidden_dim)
            q = dino_cross_layer(
                q,
                kv,
                attn_mask=dino_attn_mask,
                freqs_q_cis=self._query_freqs(
                    batch_size=batch_size, seq_len=seq_len
                ),
                freqs_k_cis=k_freqs,
                batch_has_dino=batch_has_dino,
            )
            x = q.reshape(batch_size, seq_len, num_entities, self.hidden_dim)
        return x

    def _clamp_log_b(self, log_b: Tensor) -> Tensor:
        return log_b.clamp(min=self.log_b_min, max=self.log_b_max)

    def _query_freqs(self, *, batch_size: int, seq_len: int) -> Tensor:
        """Return cross-attention query frequencies with a head broadcast axis."""
        freqs = self.token_freqs_cis[:seq_len, : self.num_entities]
        expanded: Tensor = (
            freqs.reshape(seq_len * self.num_entities, 1, self.rope_dim // 2)
            .unsqueeze(0)
            .expand(
                batch_size,
                seq_len * self.num_entities,
                1,
                self.rope_dim // 2,
            )
        )
        return expanded

    def _encode_dino(
        self,
        *,
        dino_tokens: Tensor,
        dino_frame_idx: Tensor,
        dino_attn_mask: Tensor,
        dino_batch_has_evidence: Tensor,
        batch_size: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Encode validated patch tokens and build tensor-only validity state."""

        num_samples = dino_tokens.shape[1]
        num_patches = self.dino_encoder.num_tokens
        encoded = self.dino_encoder(dino_tokens)  # (B, T_d, S, D)
        kv = encoded.reshape(batch_size, num_samples * num_patches, self.hidden_dim)

        kv = kv.masked_fill(~dino_batch_has_evidence[:, None, None], 0.0)

        # Time-axis RoPE with the *actual* sampled frame indices; the entity
        # axis uses the reserved visual-stream slot. Buffer row f holds the
        # RoPE phase of window frame f (time position f+1, mixin convention),
        # so keys index rows by the raw window-relative frame index.
        time_row = dino_frame_idx.clamp(min=0)
        freqs = self.token_freqs_cis[time_row, self.num_entities]
        k_freqs = (
            freqs.unsqueeze(2)
            .expand(
                batch_size,
                num_samples,
                num_patches,
                1,
                self.rope_dim // 2,
            )
            .reshape(
                batch_size,
                num_samples * num_patches,
                1,
                self.rope_dim // 2,
            )
        )
        return kv, dino_attn_mask, k_freqs, dino_batch_has_evidence


__all__ = ["SLCSFusionModel"]
