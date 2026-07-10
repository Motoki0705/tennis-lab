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

DINOv3 patch tokens arrive only for sparsely sampled frames. They are fused by
explicit cross-attention every ``dino_cross_attn_every`` axial blocks: queries
are all entity tokens (RoPE time position = window frame index), keys/values
are all patch tokens of all sampled frames (RoPE time position = the *actual*
sampled frame index from ``dino_frame_idx``, entity-axis position = a
dedicated visual-stream slot ``E``). Temporal propagation of visual evidence
is therefore handled by attention with true time offsets — never by implicit
interpolation. Windows without any valid patch token skip the visual pathway
explicitly.

Outputs (normalized court coordinates, ``(cos, sin)`` yaw) with per-frame
Laplace log-scales as aleatoric uncertainty:

- ``player_position (B, P, T, 3)``, ``player_position_log_b (B, P, T)``
- ``player_rotation (B, P, T, 2)``, ``player_rotation_log_b (B, P, T)``
- ``ball_position (B, T, 3)``, ``ball_position_log_b (B, T)``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

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
    TransformerBlock,
    TransformerBlockConfig,
    default_ffn_dim,
    precompute_freqs_cis_nd,
)
from src.utils.models.axial_multiview_mixin import AxialMultiViewMixin
from src.utils.models.embeddings import (
    CourtBallGroupEmbedding,
    CourtPlayerGroupEmbedding,
    InvisibleTokenEmbedding,
)
from src.utils.schema.player import NUM_HUMAN_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig


class SLCSFusionModel(AxialMultiViewMixin, nn.Module):
    """Entity x time axial transformer with sparse DINOv3 cross-attention."""

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
        rope_theta_entity: float | None = None,
        ffn_type: Literal["swiglu", "mlp"] = "swiglu",
        num_players: int = 2,
        num_court_kp: int = 14,
        max_seq_len: int = 120,
        invisible_init_std: float = 0.02,
        dino_embed_dim: int = 768,
        dino_grid_h: int = 16,
        dino_grid_w: int = 28,
        dino_cross_attn_every: int = 2,
        log_b_min: float = -6.0,
        log_b_max: float = 3.0,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.num_players = int(num_players)
        self.num_court_kp = int(num_court_kp)
        self.max_seq_len = int(max_seq_len)
        self.num_entities = self.num_players + 1
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
            num_layers=num_layers,
            max_seq_len=self.max_seq_len,
        )

        head_dim = self.hidden_dim // num_heads
        rope_dim = head_dim if rope_dim is None else int(rope_dim)
        self._validate_rope_dim(rope_dim=rope_dim, head_dim=head_dim)
        self.head_dim = int(head_dim)
        self.rope_dim = int(rope_dim)
        self.rope_bases = (
            self._coalesce_theta(rope_theta_time, rope_theta),
            self._coalesce_theta(rope_theta_entity, rope_theta),
        )

        if ffn_dim is None:
            ffn_dim = default_ffn_dim(self.hidden_dim)

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
        )

        # ---- Axial trunk with interleaved cross-attention --------------
        def block() -> TransformerBlock:
            return TransformerBlock(
                TransformerBlockConfig(
                    dim=self.hidden_dim,
                    n_heads=num_heads,
                    ffn_dim=ffn_dim,
                    head_dim=head_dim,
                    rope_dim=self.rope_dim,
                    attn_dropout=dropout,
                    rope_base=self.rope_bases,
                    ffn_type=ffn_type,
                )
            )

        self.entity_layers = nn.ModuleList([block() for _ in range(num_layers)])
        self.time_layers = nn.ModuleList([block() for _ in range(num_layers)])
        self.dino_cross_layers = nn.ModuleDict(
            {
                str(layer_idx): CrossAttnBlock(
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
                for layer_idx in range(num_layers)
                if (layer_idx + 1) % self.dino_cross_attn_every == 0
            }
        )

        self.final_norm = RMSNorm(self.hidden_dim)

        # ---- Heads ------------------------------------------------------
        head_hidden = self.hidden_dim // 2
        self.player_position_head = PlayerPositionHead(
            input_dim=self.hidden_dim, hidden_dim=head_hidden, dropout=dropout
        )
        self.player_rotation_head = PlayerRotationHead(
            input_dim=self.hidden_dim, hidden_dim=head_hidden, dropout=dropout
        )
        self.ball_position_head = BallPositionHead(
            input_dim=self.hidden_dim, hidden_dim=head_hidden, dropout=dropout
        )
        self.player_position_scale_head = LogScaleHead(
            input_dim=self.hidden_dim, hidden_dim=head_hidden // 2, dropout=dropout
        )
        self.player_rotation_scale_head = LogScaleHead(
            input_dim=self.hidden_dim, hidden_dim=head_hidden // 2, dropout=dropout
        )
        self.ball_position_scale_head = LogScaleHead(
            input_dim=self.hidden_dim, hidden_dim=head_hidden // 2, dropout=dropout
        )

        # RoPE table over (time, entity-or-visual-stream) positions. Time axis
        # holds positions 1..max_seq_len (mixin convention); the extra entity
        # slot ``num_entities`` is reserved for DINOv3 keys.
        token_freqs = precompute_freqs_cis_nd(
            dim=self.rope_dim,
            pos=self._build_token_positions(
                seq_len=self.max_seq_len,
                n_cams=self.num_entities + 1,
            ),
            base=self.rope_bases,
        )
        self.register_buffer("token_freqs_cis", token_freqs, persistent=False)

    @staticmethod
    def _validate_init_args(
        *, hidden_dim: int, num_heads: int, num_layers: int, max_seq_len: int
    ) -> None:
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

    @classmethod
    def from_config(cls, config: DictConfig) -> SLCSFusionModel:
        """Create the model from a Hydra config (``model`` + ``data`` sections)."""
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})
        dino_cfg = data_cfg.get("dino", {})

        def dino_dim(key: str, default: int) -> int:
            value = model_cfg.get(f"dino_{key}", dino_cfg.get(key))
            return int(value) if value is not None else int(default)

        patch = int(dino_cfg.get("patch_size", 16))
        grid_h_default = int(dino_cfg.get("image_height", 256)) // patch
        grid_w_default = int(dino_cfg.get("image_width", 448)) // patch
        return cls(
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            num_layers=int(model_cfg.get("num_layers", 6)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            ffn_dim=model_cfg.get("ffn_dim", None),
            dropout=float(model_cfg.get("dropout", 0.1)),
            rope_dim=model_cfg.get("rope_dim", None),
            rope_theta=float(model_cfg.get("rope_theta", 10000.0)),
            rope_theta_time=model_cfg.get("rope_theta_time", None),
            rope_theta_entity=model_cfg.get("rope_theta_entity", None),
            ffn_type=cast(
                Literal["swiglu", "mlp"], str(model_cfg.get("ffn_type", "swiglu"))
            ),
            num_players=int(
                model_cfg.get("num_players", data_cfg.get("num_players", 2))
            ),
            num_court_kp=int(
                model_cfg.get("num_court_kp", data_cfg.get("num_court_kp", 14))
            ),
            max_seq_len=int(
                model_cfg.get("max_seq_len", data_cfg.get("window_size", 120))
            ),
            invisible_init_std=float(model_cfg.get("invisible_init_std", 0.02)),
            dino_embed_dim=dino_dim("embed_dim", 768),
            dino_grid_h=int(model_cfg.get("dino_grid_h", grid_h_default)),
            dino_grid_w=int(model_cfg.get("dino_grid_w", grid_w_default)),
            dino_cross_attn_every=int(model_cfg.get("dino_cross_attn_every", 2)),
            log_b_min=float(model_cfg.get("log_b_min", -6.0)),
            log_b_max=float(model_cfg.get("log_b_max", 3.0)),
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
        dino_tokens: Tensor | None = None,
        dino_frame_idx: Tensor | None = None,
        dino_valid: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Run the fusion model on one batch of single-camera windows.

        Shapes are documented in :class:`src.tasks.slcs.data.types.SLCSBatch`
        (without the leading batch axis for per-sample tensors).
        """
        batch_size, seq_len = self._validate_forward_inputs(
            player_kp=player_kp,
            player_kp_vis=player_kp_vis,
            player_valid=player_valid,
            ball_uv=ball_uv,
            ball_vis=ball_vis,
            court_kp=court_kp,
            court_vis=court_vis,
            frame_mask=frame_mask,
            dino_tokens=dino_tokens,
            dino_frame_idx=dino_frame_idx,
            dino_valid=dino_valid,
        )
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
        player_token_valid = (
            (player_valid > 0) & frame_valid.unsqueeze(1)
        )  # (B, P, T)
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

        # Attention validity: padded frames are masked on the time axis; all
        # entity slots of a real frame participate (invisible observations are
        # already represented by the invisible token).
        token_valid = frame_valid.unsqueeze(-1).expand(batch_size, seq_len, num_entities)
        entity_axis_valid = token_valid.reshape(batch_size * seq_len, num_entities)
        time_axis_valid = token_valid.permute(0, 2, 1).reshape(
            batch_size * num_entities, seq_len
        )
        entity_mask, _ = self._build_self_attn_mask(entity_axis_valid)
        time_mask, _ = self._build_self_attn_mask(time_axis_valid)
        entity_freqs = self._camera_freqs(
            batch_size=batch_size, seq_len=seq_len, n_cams=num_entities
        )
        time_freqs = self._time_freqs(
            batch_size=batch_size, seq_len=seq_len, n_cams=num_entities
        )

        dino_ctx = self._encode_dino(
            dino_tokens=dino_tokens,
            dino_frame_idx=dino_frame_idx,
            dino_valid=dino_valid,
            batch_size=batch_size,
        )

        for layer_idx, (entity_layer, time_layer) in enumerate(
            zip(self.entity_layers, self.time_layers, strict=True)
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
            x = x_time.reshape(batch_size, num_entities, seq_len, self.hidden_dim).permute(
                0, 2, 1, 3
            )

            layer_key = str(layer_idx)
            if layer_key in self.dino_cross_layers and dino_ctx is not None:
                cross_layer = self.dino_cross_layers[layer_key]
                kv, key_valid, k_freqs = dino_ctx
                q = x.reshape(batch_size, seq_len * num_entities, self.hidden_dim)
                q_freqs = self._query_freqs(batch_size=batch_size, seq_len=seq_len)
                q = cross_layer(
                    q,
                    kv,
                    key_valid=key_valid,
                    freqs_q_cis=q_freqs,
                    freqs_k_cis=k_freqs,
                )
                x = q.reshape(batch_size, seq_len, num_entities, self.hidden_dim)

        x = self.final_norm(x)  # (B, T, E, D)
        player_feat = x[:, :, :num_players, :].permute(0, 2, 1, 3)  # (B, P, T, D)
        ball_feat = x[:, :, num_players, :]  # (B, T, D)

        return {
            "player_position": self.player_position_head(player_feat),
            "player_rotation": self.player_rotation_head(player_feat),
            "player_position_log_b": self._clamp_log_b(
                self.player_position_scale_head(player_feat).squeeze(-1)
            ),
            "player_rotation_log_b": self._clamp_log_b(
                self.player_rotation_scale_head(player_feat).squeeze(-1)
            ),
            "ball_position": self.ball_position_head(ball_feat),
            "ball_position_log_b": self._clamp_log_b(
                self.ball_position_scale_head(ball_feat).squeeze(-1)
            ),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _clamp_log_b(self, log_b: Tensor) -> Tensor:
        return log_b.clamp(min=self.log_b_min, max=self.log_b_max)

    def _query_freqs(self, *, batch_size: int, seq_len: int) -> Tensor:
        """RoPE frequencies for cross-attention queries, layout ``(B, T*E, r/2)``."""
        freqs = self.token_freqs_cis[:seq_len, : self.num_entities]  # (T, E, r/2)
        return (
            freqs.reshape(seq_len * self.num_entities, self.rope_dim // 2)
            .unsqueeze(0)
            .expand(batch_size, seq_len * self.num_entities, self.rope_dim // 2)
        )

    def _encode_dino(
        self,
        *,
        dino_tokens: Tensor | None,
        dino_frame_idx: Tensor | None,
        dino_valid: Tensor | None,
        batch_size: int,
    ) -> tuple[Tensor, Tensor, Tensor] | None:
        """Encode patch tokens into ``(kv, key_valid, k_freqs)`` or ``None``.

        ``None`` means "no visual evidence in this batch": the cross-attention
        pathway is skipped entirely (explicitly, not silently — the caller
        decides whether tokens are provided).
        """
        if dino_tokens is None:
            return None
        assert dino_frame_idx is not None and dino_valid is not None
        if not bool(dino_valid.any().item()):
            return None

        num_samples = dino_tokens.shape[1]
        num_patches = self.dino_encoder.num_tokens
        encoded = self.dino_encoder(dino_tokens)  # (B, T_d, S, D)
        kv = encoded.reshape(batch_size, num_samples * num_patches, self.hidden_dim)

        key_valid = (
            (dino_valid > 0)
            .unsqueeze(-1)
            .expand(batch_size, num_samples, num_patches)
            .reshape(batch_size, num_samples * num_patches)
        )

        # Time-axis RoPE with the *actual* sampled frame indices; the entity
        # axis uses the reserved visual-stream slot. Buffer row f holds the
        # RoPE phase of window frame f (time position f+1, mixin convention),
        # so keys index rows by the raw window-relative frame index.
        time_row = dino_frame_idx.clamp(min=0)
        freqs = self.token_freqs_cis[time_row, self.num_entities]  # (B, T_d, r/2)
        k_freqs = (
            freqs.unsqueeze(2)
            .expand(batch_size, num_samples, num_patches, self.rope_dim // 2)
            .reshape(batch_size, num_samples * num_patches, self.rope_dim // 2)
        )
        return kv, key_valid, k_freqs

    def _validate_forward_inputs(
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
        dino_tokens: Tensor | None,
        dino_frame_idx: Tensor | None,
        dino_valid: Tensor | None,
    ) -> tuple[int, int]:
        if player_kp.dim() != 5 or player_kp.shape[-2:] != (NUM_HUMAN_KP, 2):
            raise ValueError(
                f"player_kp must be (B, P, T, {NUM_HUMAN_KP}, 2), got {tuple(player_kp.shape)}."
            )
        batch_size, num_players, seq_len = player_kp.shape[:3]
        if num_players != self.num_players:
            raise ValueError(
                f"player_kp has P={num_players}, model configured for {self.num_players}."
            )
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length T={seq_len} exceeds max_seq_len={self.max_seq_len}."
            )
        if player_kp_vis.shape != (batch_size, num_players, seq_len, NUM_HUMAN_KP):
            raise ValueError(
                f"player_kp_vis must be (B, P, T, {NUM_HUMAN_KP}), "
                f"got {tuple(player_kp_vis.shape)}."
            )
        if player_valid.shape != (batch_size, num_players, seq_len):
            raise ValueError(
                f"player_valid must be (B, P, T), got {tuple(player_valid.shape)}."
            )
        if ball_uv.shape != (batch_size, seq_len, 2):
            raise ValueError(f"ball_uv must be (B, T, 2), got {tuple(ball_uv.shape)}.")
        if ball_vis.shape != (batch_size, seq_len):
            raise ValueError(f"ball_vis must be (B, T), got {tuple(ball_vis.shape)}.")
        if court_kp.shape != (batch_size, seq_len, self.num_court_kp, 2):
            raise ValueError(
                f"court_kp must be (B, T, {self.num_court_kp}, 2), "
                f"got {tuple(court_kp.shape)}."
            )
        if court_vis.shape != (batch_size, seq_len, self.num_court_kp):
            raise ValueError(
                f"court_vis must be (B, T, {self.num_court_kp}), got {tuple(court_vis.shape)}."
            )
        if frame_mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"frame_mask must be (B, T), got {tuple(frame_mask.shape)}."
            )

        dino_args = (dino_tokens, dino_frame_idx, dino_valid)
        provided = [arg is not None for arg in dino_args]
        if any(provided) and not all(provided):
            raise ValueError(
                "dino_tokens, dino_frame_idx and dino_valid must be provided together."
            )
        if dino_tokens is not None:
            assert dino_frame_idx is not None and dino_valid is not None
            if dino_tokens.dim() != 4 or dino_tokens.shape[0] != batch_size:
                raise ValueError(
                    f"dino_tokens must be (B, T_d, S, C), got {tuple(dino_tokens.shape)}."
                )
            num_samples = dino_tokens.shape[1]
            if dino_frame_idx.shape != (batch_size, num_samples):
                raise ValueError(
                    f"dino_frame_idx must be (B, T_d), got {tuple(dino_frame_idx.shape)}."
                )
            if dino_valid.shape != (batch_size, num_samples):
                raise ValueError(
                    f"dino_valid must be (B, T_d), got {tuple(dino_valid.shape)}."
                )
            valid_frames = dino_frame_idx[dino_valid > 0]
            if valid_frames.numel() > 0 and (
                int(valid_frames.min().item()) < 0
                or int(valid_frames.max().item()) >= seq_len
            ):
                raise ValueError(
                    "dino_frame_idx of valid samples must lie inside the window "
                    f"[0, {seq_len}), got range "
                    f"[{int(valid_frames.min().item())}, {int(valid_frames.max().item())}]."
                )
        return batch_size, seq_len


__all__ = ["SLCSFusionModel"]
