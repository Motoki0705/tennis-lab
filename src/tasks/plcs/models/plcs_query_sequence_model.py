"""Query-based sequential PLCS model (2-stage, joint-wise temporal)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import torch
import torch.nn as nn
from torch import Tensor

from src.tasks.plcs.models.components.heads import PositionHead, RotationHead
from src.utils.models import (
    CrossAttnBlock,
    CrossAttnBlockConfig,
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    precompute_freqs_cis_nd,
)
from src.utils.models.embeddings import (
    CourtKPUVEmbedding,
    InvisibleTokenEmbedding,
    PlayerKPUVEmbedding,
)
from src.utils.schema.court import NUM_COURT_KP
from src.utils.schema.player import NUM_HUMAN_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig


class PLCSQuerySequenceModel(nn.Module):
    """Query-based sequential PLCS model (2-stage, joint-wise temporal).

    Stage 1:
    - Interleaved Player->Court cross-attention and joint-wise temporal self-attention.

    Stage 2:
    - Shared readout query stream attends to per-frame joint states, then temporal self-attend.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        rope_dim: int | None = None,
        rope_theta: float = 10000.0,
        rope_theta_time: float | None = None,
        rope_theta_entity: float | None = None,
        rope_theta_type: float = 100.0,
        num_player_layers: int = 4,
        num_query_layers: int = 2,
        ffn_type: Literal["swiglu", "mlp"] = "swiglu",
        max_seq_len: int = 120,
        invisible_init_std: float = 0.02,
        query_init_std: float = 0.02,
        num_court_tokens: int = NUM_COURT_KP,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if num_player_layers < 0:
            raise ValueError(
                f"num_player_layers must be non-negative, got {num_player_layers}"
            )
        if num_query_layers < 0:
            raise ValueError(f"num_query_layers must be non-negative, got {num_query_layers}")
        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)
        self.num_court_tokens = int(num_court_tokens)
        self.num_joints = int(NUM_HUMAN_KP)
        self.num_query_tokens = 2

        head_dim = hidden_dim // num_heads
        rope_dim = head_dim if rope_dim is None else int(rope_dim)
        if rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be even, got {rope_dim}")
        if rope_dim > head_dim:
            raise ValueError(f"rope_dim={rope_dim} cannot exceed head_dim={head_dim}")
        self.rope_dim = int(rope_dim)
        self.rope_theta = float(rope_theta)
        self.rope_bases = (
            float(self.rope_theta if rope_theta_time is None else rope_theta_time),
            float(self.rope_theta if rope_theta_entity is None else rope_theta_entity),
            float(rope_theta_type),
        )

        if ffn_dim is None:
            ffn_dim = int((8 * hidden_dim) / 3)
            ffn_dim = (ffn_dim + 63) // 64 * 64

        self.invisible_token = InvisibleTokenEmbedding(dim=hidden_dim, init_std=invisible_init_std)
        self.court_embed = CourtKPUVEmbedding(
            dim=hidden_dim,
            dropout=dropout,
            invisible_token=self.invisible_token,
        )
        self.player_embed = PlayerKPUVEmbedding(
            dim=hidden_dim,
            dropout=dropout,
            invisible_token=self.invisible_token,
        )

        self.court_id_embed = nn.Embedding(self.num_court_tokens, hidden_dim)
        self.joint_id_embed = nn.Embedding(self.num_joints, hidden_dim)

        self.query_base = nn.Parameter(torch.randn(1, 1, 2, hidden_dim) * query_init_std)

        self.player_cross_layers = nn.ModuleList(
            [
                CrossAttnBlock(
                    CrossAttnBlockConfig(
                        dim=hidden_dim,
                        n_heads=num_heads,
                        ffn_dim=ffn_dim,
                        head_dim=head_dim,
                        rope_dim=self.rope_dim,
                        attn_dropout=dropout,
                        ffn_type=ffn_type,
                    )
                )
                for _ in range(num_player_layers)
            ]
        )
        self.player_self_layers = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=hidden_dim,
                        n_heads=num_heads,
                        ffn_dim=ffn_dim,
                        head_dim=head_dim,
                        rope_dim=self.rope_dim,
                        attn_dropout=dropout,
                        rope_base=self.rope_theta,
                        ffn_type=ffn_type,
                    )
                )
                for _ in range(num_player_layers)
            ]
        )

        self.query_cross_layers = nn.ModuleList(
            [
                CrossAttnBlock(
                    CrossAttnBlockConfig(
                        dim=hidden_dim,
                        n_heads=num_heads,
                        ffn_dim=ffn_dim,
                        head_dim=head_dim,
                        rope_dim=self.rope_dim,
                        attn_dropout=dropout,
                        ffn_type=ffn_type,
                    )
                )
                for _ in range(num_query_layers)
            ]
        )
        self.query_self_layers = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=hidden_dim,
                        n_heads=num_heads,
                        ffn_dim=ffn_dim,
                        head_dim=head_dim,
                        rope_dim=self.rope_dim,
                        attn_dropout=dropout,
                        rope_base=self.rope_theta,
                        ffn_type=ffn_type,
                    )
                )
                for _ in range(num_query_layers)
            ]
        )

        self.final_norm = RMSNorm(hidden_dim)

        self.position_head = PositionHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            output_dim=3,
            num_layers=2,
            dropout=dropout,
        )
        self.rotation_head = RotationHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            num_layers=2,
            dropout=dropout,
        )

        court_positions = self._build_court_positions()
        court_freqs_cis = precompute_freqs_cis_nd(
            dim=self.rope_dim,
            pos=court_positions,
            base=self.rope_bases,
        )
        self.register_buffer("court_freqs_cis", court_freqs_cis, persistent=False)

    @classmethod
    def from_config(cls, config: DictConfig) -> PLCSQuerySequenceModel:
        """Create query-based sequence model from hydra config."""
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})

        return cls(
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            ffn_dim=model_cfg.get("ffn_dim", None),
            dropout=float(model_cfg.get("dropout", 0.1)),
            rope_dim=model_cfg.get("rope_dim", None),
            rope_theta=float(model_cfg.get("rope_theta", 10000.0)),
            rope_theta_time=model_cfg.get("rope_theta_time", None),
            rope_theta_entity=model_cfg.get("rope_theta_entity", None),
            rope_theta_type=model_cfg.get("rope_theta_type", 100.0),
            num_player_layers=int(model_cfg.get("num_player_layers", 4)),
            num_query_layers=int(model_cfg.get("num_query_layers", 2)),
            ffn_type=cast(Literal["swiglu", "mlp"], str(model_cfg.get("ffn_type", "swiglu"))),
            max_seq_len=int(model_cfg.get("max_seq_len", 120)),
            invisible_init_std=float(model_cfg.get("invisible_init_std", 0.02)),
            query_init_std=float(model_cfg.get("query_init_std", 0.02)),
            num_court_tokens=int(data_cfg.get("num_court_kp", NUM_COURT_KP)),
        )

    def _build_court_positions(self) -> Tensor:
        court_idx = torch.arange(self.num_court_tokens, dtype=torch.long)
        return torch.stack(
            [
                torch.zeros_like(court_idx),
                court_idx,
                torch.zeros_like(court_idx),
            ],
            dim=-1,
        )

    def _build_player_cross_positions(self, seq_len: int, device: torch.device) -> Tensor:
        time_idx = torch.arange(seq_len, device=device, dtype=torch.long).repeat_interleave(
            self.num_joints
        ) + 1
        joint_idx = torch.arange(self.num_joints, device=device, dtype=torch.long).repeat(seq_len)
        token_type = torch.ones(seq_len * self.num_joints, device=device, dtype=torch.long)
        return torch.stack([time_idx, joint_idx, token_type], dim=-1)

    def _build_player_temporal_positions(self, seq_len: int, device: torch.device) -> Tensor:
        time_idx = torch.arange(seq_len, device=device, dtype=torch.long).view(1, seq_len)
        joint_idx = torch.arange(self.num_joints, device=device, dtype=torch.long).view(
            self.num_joints, 1
        )
        token_type = torch.ones((self.num_joints, seq_len), device=device, dtype=torch.long)
        return torch.stack(
            [
                time_idx.expand(self.num_joints, seq_len) + 1,
                joint_idx.expand(self.num_joints, seq_len),
                token_type,
            ],
            dim=-1,
        )

    def _build_query_frame_positions(self, seq_len: int, device: torch.device) -> Tensor:
        time_idx = torch.arange(seq_len, device=device, dtype=torch.long).view(seq_len, 1) + 1
        entity_idx = torch.zeros((seq_len, self.num_query_tokens), device=device, dtype=torch.long)
        query_type = torch.arange(
            2,
            2 + self.num_query_tokens,
            device=device,
            dtype=torch.long,
        ).view(1, self.num_query_tokens)
        return torch.stack(
            [
                time_idx.expand(seq_len, self.num_query_tokens),
                entity_idx,
                query_type.expand(seq_len, self.num_query_tokens),
            ],
            dim=-1,
        )

    def _build_player_key_positions(self, seq_len: int, device: torch.device) -> Tensor:
        time_idx = torch.arange(seq_len, device=device, dtype=torch.long).view(seq_len, 1) + 1
        joint_idx = torch.arange(self.num_joints, device=device, dtype=torch.long).view(1, self.num_joints)
        token_type = torch.ones((seq_len, self.num_joints), device=device, dtype=torch.long)
        return torch.stack(
            [
                time_idx.expand(seq_len, self.num_joints),
                joint_idx.expand(seq_len, self.num_joints),
                token_type,
            ],
            dim=-1,
        )

    def _build_query_temporal_positions(self, seq_len: int, device: torch.device) -> Tensor:
        time_idx = torch.arange(seq_len, device=device, dtype=torch.long).view(1, seq_len)
        query_type = torch.arange(
            2,
            2 + self.num_query_tokens,
            device=device,
            dtype=torch.long,
        ).view(self.num_query_tokens, 1)
        entity_idx = torch.zeros((self.num_query_tokens, seq_len), device=device, dtype=torch.long)
        return torch.stack(
            [
                time_idx.expand(self.num_query_tokens, seq_len) + 1,
                entity_idx,
                query_type.expand(self.num_query_tokens, seq_len),
            ],
            dim=-1,
        )

    def _normalize_court_inputs(
        self,
        court_kp: Tensor,
        court_vis: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        """Normalize court tensors to scene-level shapes: (B,K,2)/(B,K)."""
        if court_kp.dim() == 4:
            court_kp = court_kp[:, 0, :, :]
        elif court_kp.dim() == 3:
            if court_kp.size(1) == self.num_court_tokens and court_kp.size(2) == 2:
                pass
            elif court_kp.size(2) == self.num_court_tokens * 2:
                court_kp = court_kp[:, 0, :]
            else:
                raise ValueError(
                    f"Unsupported court_kp shape {tuple(court_kp.shape)}. "
                    f"Expected (B,{self.num_court_tokens * 2}), (B,{self.num_court_tokens},2), "
                    f"(B,T,{self.num_court_tokens * 2}), or (B,T,{self.num_court_tokens},2)."
                )

        if court_vis is not None and court_vis.dim() == 3:
            court_vis = court_vis[:, 0, :]

        return court_kp, court_vis

    def _build_player_temporal_valid(
        self,
        *,
        player_mask: Tensor | None,
        batch_size: int,
        seq_len: int,
        num_joints: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Build temporal validity masks for stage-1/2 attention.

        Returns:
            tuple:
              - player_valid_tj: (B, T, J), joint-wise temporal valid mask.
              - frame_valid_t: (B, T), frame-level valid mask.
        """
        if player_mask is None:
            player_valid_tj = torch.ones(
                batch_size,
                seq_len,
                num_joints,
                device=device,
                dtype=torch.bool,
            )
            frame_valid_t = torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)
            return player_valid_tj, frame_valid_t

        if player_mask.dim() == 2:
            if player_mask.shape != (batch_size, seq_len):
                raise ValueError(
                    f"player_mask must have shape {(batch_size, seq_len)} when 2D, "
                    f"got {tuple(player_mask.shape)}"
                )
            frame_valid_t = player_mask > 0
            player_valid_tj = frame_valid_t.unsqueeze(-1).expand(batch_size, seq_len, num_joints)
            return player_valid_tj, frame_valid_t

        if player_mask.dim() == 3:
            if player_mask.shape != (batch_size, seq_len, num_joints):
                raise ValueError(
                    f"player_mask must have shape {(batch_size, seq_len, num_joints)} when 3D, "
                    f"got {tuple(player_mask.shape)}"
                )
            player_valid_tj = player_mask > 0
            frame_valid_t = player_valid_tj.any(dim=2)
            return player_valid_tj, frame_valid_t

        raise ValueError(
            f"player_mask must be 2D or 3D, got dim={player_mask.dim()} "
            f"with shape {tuple(player_mask.shape)}"
        )

    def forward(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None = None,
        human_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            human_kp: Player keypoints, shape (B,T,34) or (B,T,J,2).
            court_kp: Court keypoints, shape (B,40)/(B,20,2)/(B,T,40)/(B,T,20,2).
            human_vis: Player visibility, shape (B,T,J). Used for invisible tokens.
            court_vis: Court visibility, shape (B,20) or (B,T,20). Used for invisible tokens.
            human_mask:
                Padding mask. Supported shapes:
                - (B, T): frame validity
                - (B, T, J): joint-wise frame validity

        Returns:
            dict:
              - position: (B,T,3)
              - rotation: (B,T,2)
        """
        if human_kp.dim() not in {3, 4}:
            raise ValueError(
                "PLCSQuerySequenceModel expects human_kp as (B,T,17,2) or (B,T,34), "
                f"got shape {tuple(human_kp.shape)}"
            )
        if human_vis is not None and human_vis.dim() != 3:
            raise ValueError(
                "PLCSQuerySequenceModel expects human_vis as (B,T,17), "
                f"got shape {tuple(human_vis.shape)}"
            )
        if court_kp.dim() not in {2, 3, 4}:
            raise ValueError(
                "PLCSQuerySequenceModel expects court_kp as (B,40)/(B,20,2)/(B,T,40)/(B,T,20,2), "
                f"got shape {tuple(court_kp.shape)}"
            )
        if court_vis is not None and court_vis.dim() not in {2, 3}:
            raise ValueError(
                "PLCSQuerySequenceModel expects court_vis as (B,20) or (B,T,20), "
                f"got shape {tuple(court_vis.shape)}"
            )

        seq_mask: Tensor | None = None
        if human_mask is not None:
            if human_mask.dim() == 2:  # (B, T)
                seq_mask = human_mask > 0
            elif human_mask.dim() == 3:  # (B, T, J)
                seq_mask = (human_mask > 0).any(dim=-1)
            else:
                raise ValueError(
                    "human_mask for query sequence models must be (B,T) or (B,T,J), "
                    f"got shape {tuple(human_mask.shape)}"
                )

        batch_size, seq_len = human_kp.shape[:2]
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}. "
                "Increase model.max_seq_len."
            )

        court_kp, court_vis = self._normalize_court_inputs(court_kp, court_vis)
        court_tok = self.court_embed(court_kp, court_vis)
        if court_tok.shape[1] != self.num_court_tokens:
            raise ValueError(
                f"Expected {self.num_court_tokens} court tokens, got {court_tok.shape[1]}"
            )

        if human_kp.dim() == 3:
            human_kp = human_kp.view(batch_size, seq_len, self.num_joints, 2)
        if human_kp.shape[2] != self.num_joints:
            raise ValueError(
                f"Expected {self.num_joints} joints, got {human_kp.shape[2]}"
            )

        human_kp_flat = human_kp.reshape(batch_size * seq_len, self.num_joints, 2)
        human_vis_flat: Tensor | None = None
        if human_vis is not None:
            if human_vis.shape != (batch_size, seq_len, self.num_joints):
                raise ValueError(
                    f"human_vis must have shape {(batch_size, seq_len, self.num_joints)}, "
                    f"got {tuple(human_vis.shape)}"
                )
            human_vis_flat = human_vis.reshape(batch_size * seq_len, self.num_joints)

        player_tok = self.player_embed(human_kp_flat, human_vis_flat)
        player_tok = player_tok.view(batch_size, seq_len, self.num_joints, self.hidden_dim)

        court_ids = torch.arange(
            self.num_court_tokens,
            device=human_kp.device,
            dtype=torch.long,
        )
        court_tok = court_tok + self.court_id_embed(court_ids).unsqueeze(0)

        joint_ids = torch.arange(self.num_joints, device=human_kp.device, dtype=torch.long)
        player_tok = player_tok + self.joint_id_embed(joint_ids).view(1, 1, self.num_joints, -1)

        player_cross_freqs = precompute_freqs_cis_nd(
            dim=self.rope_dim,
            pos=self._build_player_cross_positions(seq_len, human_kp.device),
            base=self.rope_bases,
        )
        player_temporal_freqs = precompute_freqs_cis_nd(
            dim=self.rope_dim,
            pos=self._build_player_temporal_positions(seq_len, human_kp.device),
            base=self.rope_bases,
        )
        query_frame_freqs = precompute_freqs_cis_nd(
            dim=self.rope_dim,
            pos=self._build_query_frame_positions(seq_len, human_kp.device),
            base=self.rope_bases,
        )
        player_key_freqs = precompute_freqs_cis_nd(
            dim=self.rope_dim,
            pos=self._build_player_key_positions(seq_len, human_kp.device),
            base=self.rope_bases,
        )
        query_temporal_freqs = precompute_freqs_cis_nd(
            dim=self.rope_dim,
            pos=self._build_query_temporal_positions(seq_len, human_kp.device),
            base=self.rope_bases,
        )
        court_freqs_cis = cast(Tensor, self.court_freqs_cis)
        if court_freqs_cis.device != human_kp.device:
            court_freqs_cis = court_freqs_cis.to(human_kp.device)

        court_valid = torch.ones(
            batch_size,
            self.num_court_tokens,
            device=human_kp.device,
            dtype=torch.bool,
        )
        player_valid_tj, frame_valid_t = self._build_player_temporal_valid(
            player_mask=seq_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            num_joints=self.num_joints,
            device=human_kp.device,
        )

        player_x = player_tok
        for cross_layer, self_layer in zip(
            self.player_cross_layers,
            self.player_self_layers,
            strict=True,
        ):
            player_flat = player_x.reshape(batch_size, seq_len * self.num_joints, self.hidden_dim)
            player_flat = cross_layer(
                player_flat,
                court_tok,
                key_valid=court_valid,
                freqs_q_cis=player_cross_freqs,
                freqs_k_cis=court_freqs_cis,
            )
            player_x = player_flat.view(batch_size, seq_len, self.num_joints, self.hidden_dim)

            # Joint-wise temporal self-attention: (B,T,J,D) -> (B*J,T,D)
            player_btjd = player_x.permute(0, 2, 1, 3).reshape(
                batch_size * self.num_joints, seq_len, self.hidden_dim
            )
            joint_valid_bt = player_valid_tj.permute(0, 2, 1).reshape(
                batch_size * self.num_joints, seq_len
            )
            joint_attn_mask = joint_valid_bt[:, None, :].expand(
                batch_size * self.num_joints, seq_len, seq_len
            )
            empty_joint = ~joint_valid_bt.any(dim=1)
            if empty_joint.any():
                joint_attn_mask = joint_attn_mask.clone()
                joint_attn_mask[empty_joint, :, 0] = True

            player_joint_freqs = player_temporal_freqs.unsqueeze(0).expand(
                batch_size,
                self.num_joints,
                seq_len,
                self.rope_dim // 2,
            ).reshape(batch_size * self.num_joints, seq_len, self.rope_dim // 2)

            player_btjd = self_layer(
                player_btjd,
                freqs_cis=player_joint_freqs,
                attn_mask=joint_attn_mask,
            )
            player_x = player_btjd.reshape(batch_size, self.num_joints, seq_len, self.hidden_dim)
            player_x = player_x.permute(0, 2, 1, 3).contiguous()

        query = self.query_base.expand(batch_size, seq_len, -1, -1)

        frame_attn_mask = frame_valid_t[:, None, :].expand(batch_size, seq_len, seq_len)
        empty_frame = ~frame_valid_t.any(dim=1)
        if empty_frame.any():
            frame_attn_mask = frame_attn_mask.clone()
            frame_attn_mask[empty_frame, :, 0] = True

        player_keys = player_x.reshape(batch_size * seq_len, self.num_joints, self.hidden_dim)
        key_valid_btj = player_valid_tj.reshape(batch_size * seq_len, self.num_joints)
        empty_bt = ~key_valid_btj.any(dim=1)
        if empty_bt.any():
            key_valid_btj = key_valid_btj.clone()
            key_valid_btj[empty_bt, 0] = True

        query_frame_freqs_bt = query_frame_freqs.unsqueeze(0).expand(
            batch_size,
            seq_len,
            self.num_query_tokens,
            self.rope_dim // 2,
        ).reshape(batch_size * seq_len, self.num_query_tokens, self.rope_dim // 2)
        player_key_freqs_bt = player_key_freqs.unsqueeze(0).expand(
            batch_size,
            seq_len,
            self.num_joints,
            self.rope_dim // 2,
        ).reshape(batch_size * seq_len, self.num_joints, self.rope_dim // 2)

        frame_attn_mask_b2 = frame_attn_mask.repeat_interleave(2, dim=0)

        for cross_layer, self_layer in zip(
            self.query_cross_layers,
            self.query_self_layers,
            strict=True,
        ):
            # Per-timestep readout: (B,T,2,D) -> (B*T,2,D), keys (B*T,J,D)
            query_bt2 = query.reshape(batch_size * seq_len, 2, self.hidden_dim)
            query_bt2 = cross_layer(
                query_bt2,
                player_keys,
                key_valid=key_valid_btj,
                freqs_q_cis=query_frame_freqs_bt,
                freqs_k_cis=player_key_freqs_bt,
            )
            query = query_bt2.reshape(batch_size, seq_len, 2, self.hidden_dim)

            # Temporal self-attn with shared layers over query types.
            query_b2t = query.permute(0, 2, 1, 3).reshape(batch_size * 2, seq_len, self.hidden_dim)
            query_temporal_freqs_batched = query_temporal_freqs.unsqueeze(0).expand(
                batch_size,
                self.num_query_tokens,
                seq_len,
                self.rope_dim // 2,
            ).reshape(batch_size * self.num_query_tokens, seq_len, self.rope_dim // 2)
            query_b2t = self_layer(
                query_b2t,
                freqs_cis=query_temporal_freqs_batched,
                attn_mask=frame_attn_mask_b2,
            )
            query = query_b2t.reshape(batch_size, 2, seq_len, self.hidden_dim).permute(0, 2, 1, 3)

        query = self.final_norm(query.reshape(batch_size * seq_len * 2, self.hidden_dim))
        query = query.view(batch_size, seq_len, 2, self.hidden_dim)
        po_query = query[:, :, 0, :]
        ro_query = query[:, :, 1, :]

        position = self.position_head(po_query.reshape(batch_size * seq_len, self.hidden_dim))
        rotation = self.rotation_head(ro_query.reshape(batch_size * seq_len, self.hidden_dim))

        return {
            "position": position.view(batch_size, seq_len, 3),
            "rotation": rotation.view(batch_size, seq_len, 2),
        }

    def get_num_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
