"""Multi-view PLCS model with camera-time 2D RoPE.

Per-camera token layout:
    [court_m(20), frame_{m,0}(2+17), ..., frame_{m,T-1}(2+17)]
where each frame block is:
    [CLS_po_{m,t}, CLS_ro_{m,t}, player_{m,t,0..16}]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import torch
import torch.nn as nn
from torch import Tensor

from src.tasks.plcs.models.components.heads import (
    CanonicalPoseHead,
    PositionHead,
    RotationHead,
)
from src.utils.models import (
    RMSNorm,
    RotaryFrequencyComputer,
    TransformerBlock,
    TransformerBlockConfig,
    resolve_rope_bases,
)
from src.utils.models.embeddings import (
    CourtKPUVEmbedding,
    InvisibleTokenEmbedding,
    PlayerKPUVEmbedding,
)
from src.utils.schema.player import NUM_HUMAN_KP

if TYPE_CHECKING:
    from src.tasks.plcs.configuration import PLCSModelConfig


class PLCSMultiViewModel(nn.Module):
    """Multi-view PLCS model using 3-axis interleaved MROPE."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        rope_dim: int,
        rope_theta: float,
        rope_theta_time: float,
        rope_theta_camera: float,
        rope_theta_type: float,
        ffn_type: Literal["swiglu", "mlp"],
        predict_canonical_pose: bool,
        max_views: int,
        max_seq_len: int,
        invisible_init_std: float,
        num_court_tokens: int,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.predict_canonical_pose = bool(predict_canonical_pose)
        self.max_views = int(max_views)
        self.max_seq_len = int(max_seq_len)
        self.num_court_tokens = int(num_court_tokens)

        self.frame_block_tokens = 2 + NUM_HUMAN_KP

        head_dim = hidden_dim // num_heads
        self.rope_dim = rope_dim
        self.rope_theta = float(rope_theta)
        self.rope_bases = resolve_rope_bases(
            rope_theta_time=rope_theta_time,
            rope_theta_camera=rope_theta_camera,
            rope_theta_type=rope_theta_type,
        )

        self._validate_init_args(rope_dim=self.rope_dim)

        self.invisible_token = InvisibleTokenEmbedding(
            dim=hidden_dim, init_std=invisible_init_std
        )
        self.court_embed = CourtKPUVEmbedding(
            dim=hidden_dim,
            invisible_token=self.invisible_token,
        )
        self.player_embed = PlayerKPUVEmbedding(
            dim=hidden_dim,
            invisible_token=self.invisible_token,
        )

        self.cls_po_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.cls_ro_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_po_token, std=0.02)
        nn.init.trunc_normal_(self.cls_ro_token, std=0.02)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=hidden_dim,
                        n_heads=num_heads,
                        ffn_dim=ffn_dim,
                        head_dim=head_dim,
                        rope_dim=self.rope_dim,
                        attn_dropout=dropout,
                        attention_type="mha",
                        n_kv_heads=None,
                        rope_base=self.rope_theta,
                        ffn_type=ffn_type,
                    )
                )
                for _ in range(num_layers)
            ]
        )
        self.rope_frequency_computer = RotaryFrequencyComputer(
            dim=self.rope_dim,
            base=self.rope_bases,
            n_axes=3,
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
        self.canonical_pose_head = None
        if self.predict_canonical_pose:
            self.canonical_pose_head = CanonicalPoseHead(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim // 2,
                num_layers=2,
                dropout=dropout,
                num_keypoints=NUM_HUMAN_KP,
            )
            self._decode_features = self._decode_features_with_canonical_pose
        else:
            self._decode_features = self._decode_features_without_canonical_pose

    @staticmethod
    def _validate_init_args(*, rope_dim: int) -> None:
        if rope_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even rope_dim, got {rope_dim}")

    @classmethod
    def from_config(
        cls, config: PLCSModelConfig, *, num_court_tokens: int
    ) -> PLCSMultiViewModel:
        """Create model from hydra config."""
        return cls(
            hidden_dim=config.integer("hidden_dim"),
            num_layers=config.integer("num_layers"),
            num_heads=config.integer("num_heads"),
            ffn_dim=config.integer("ffn_dim"),
            dropout=config.number("dropout"),
            rope_dim=config.integer("rope_dim"),
            rope_theta=config.number("rope_theta"),
            rope_theta_time=config.number("rope_theta_time"),
            rope_theta_camera=config.number("rope_theta_camera"),
            rope_theta_type=config.number("rope_theta_type"),
            ffn_type=cast(Literal["swiglu", "mlp"], config.string("ffn_type")),
            predict_canonical_pose=config.boolean("predict_canonical_pose"),
            max_views=config.integer("max_views"),
            max_seq_len=config.integer("max_seq_len"),
            invisible_init_std=config.number("invisible_init_std"),
            num_court_tokens=num_court_tokens,
        )

    def _build_positions_3d(
        self,
        *,
        n_cameras: int,
        seq_len: int,
        device: torch.device,
    ) -> Tensor:
        """Build `(S, 3)` coordinates with axes: (time, camera, type)."""
        per_cam: list[Tensor] = []
        for cam_idx in range(n_cameras):
            court_time = torch.zeros(
                self.num_court_tokens, device=device, dtype=torch.long
            )
            court_cam = torch.full(
                (self.num_court_tokens,), cam_idx, device=device, dtype=torch.long
            )
            court_type = torch.zeros(
                self.num_court_tokens, device=device, dtype=torch.long
            )
            court_pos = torch.stack([court_time, court_cam, court_type], dim=-1)

            frame_time = (
                torch.arange(
                    seq_len, device=device, dtype=torch.long
                ).repeat_interleave(self.frame_block_tokens)
                + 1
            )
            frame_cam = torch.full(
                (seq_len * self.frame_block_tokens,),
                cam_idx,
                device=device,
                dtype=torch.long,
            )
            frame_type = torch.ones(
                seq_len * self.frame_block_tokens, device=device, dtype=torch.long
            )
            frame_pos = torch.stack([frame_time, frame_cam, frame_type], dim=-1)
            per_cam.append(torch.cat([court_pos, frame_pos], dim=0))

        return torch.cat(per_cam, dim=0)

    def forward(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor,
        padding_mask: Tensor,
        court_vis: Tensor,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            human_kp:
                Human 2D keypoints in normalized image UV.
                - Temporal: (B, N, T, 17, 2)
                - Single frame: (B, N, 17, 2)
            court_kp:
                Court 2D keypoints in normalized image UV.
                - Temporal: (B, N, T, K, 2)
                - Single frame: (B, N, K, 2)
            human_vis:
                Human keypoint visibility flags aligned with `human_kp`.
                - Temporal: (B, N, T, 17)
                - Single frame: (B, N, 17)
                Each element is interpreted as visible if > 0 (bool/0-1).
                Optional; if None, all human keypoints are treated as visible.
            court_vis:
                Court keypoint visibility flags aligned with `court_kp`.
                - Temporal: (B, N, T, K)
                - Single frame: (B, N, K)
                Each element is interpreted as visible if > 0.
                Optional; if None, all court keypoints are treated as visible.
            padding_mask:
                Padding mask with shape (B, N, T), True for padding.

        Returns:
            dict:
                - position: (B, T, 3)
                - rotation: (B, T, 2)
        """
        B, N, T = human_kp.shape[:3]
        device = human_kp.device

        token_valid = ~padding_mask
        view_valid = token_valid.any(dim=2)

        # court is camera-level (use first frame per camera)
        court_scene = court_kp[:, :, 0, :, :]  # (B,N,K,2)
        K = court_scene.shape[2]
        court_scene_flat = court_scene.reshape(B * N, K, 2)

        court_vis_scene = court_vis[:, :, 0, :].reshape(B * N, K)

        court_tok = self.court_embed(court_scene_flat, court_vis_scene)
        court_tok = court_tok.view(B, N, K, self.hidden_dim)

        # player tokens per (camera,time,kp)
        human_flat = human_kp.reshape(B * N * T, NUM_HUMAN_KP, 2)

        human_vis_flat = human_vis.reshape(B * N * T, NUM_HUMAN_KP)

        player_tok = self.player_embed(human_flat, human_vis_flat)
        player_tok = player_tok.view(B, N, T, NUM_HUMAN_KP, self.hidden_dim)

        cls_po = self.cls_po_token.expand(B, N, T, -1, -1)
        cls_ro = self.cls_ro_token.expand(B, N, T, -1, -1)
        frame_tok = torch.cat([cls_po, cls_ro, player_tok], dim=3)  # (B,N,T,19,D)
        frame_tok = frame_tok.reshape(
            B, N, T * self.frame_block_tokens, self.hidden_dim
        )

        camera_block_tokens = K + T * self.frame_block_tokens
        tokens_per_camera = torch.cat([court_tok, frame_tok], dim=2)  # (B,N,L_cam,D)
        x = tokens_per_camera.reshape(B, N * camera_block_tokens, self.hidden_dim)

        # Build token validity mask from padding masks only.
        frame_token_valid = token_valid
        court_token_valid = view_valid

        court_rep = court_token_valid.unsqueeze(-1).expand(B, N, K)
        frame_rep = frame_token_valid.unsqueeze(-1).expand(
            B, N, T, self.frame_block_tokens
        )
        frame_rep = frame_rep.reshape(B, N, T * self.frame_block_tokens)

        token_valid = torch.cat([court_rep, frame_rep], dim=-1).reshape(
            B, N * camera_block_tokens
        )

        x = x * token_valid.unsqueeze(-1).to(dtype=x.dtype)

        S = x.size(1)
        attn_keep = token_valid[:, :, None] & token_valid[:, None, :]
        eye = torch.eye(S, device=device, dtype=torch.bool).unsqueeze(0)
        attn_mask = attn_keep | eye

        positions_3d = self._build_positions_3d(
            n_cameras=N,
            seq_len=T,
            device=device,
        )
        freqs_cis = self.rope_frequency_computer(positions_3d)

        for blk in self.blocks:
            x = blk(
                x,
                freqs_cis=freqs_cis,
                attn_mask=attn_mask,
            )
            x = x * token_valid.unsqueeze(-1).to(dtype=x.dtype)

        x = self.final_norm(x)
        x = x * token_valid.unsqueeze(-1).to(dtype=x.dtype)

        time_offsets = (
            K
            + torch.arange(T, device=device, dtype=torch.long) * self.frame_block_tokens
        )
        cam_offsets = (
            torch.arange(N, device=device, dtype=torch.long).view(N, 1)
            * camera_block_tokens
        )
        po_idx = (cam_offsets + time_offsets.view(1, T)).reshape(-1)
        ro_idx = po_idx + 1

        po_feat = x.gather(
            1, po_idx.view(1, N * T, 1).expand(B, N * T, self.hidden_dim)
        )
        ro_feat = x.gather(
            1, ro_idx.view(1, N * T, 1).expand(B, N * T, self.hidden_dim)
        )
        po_feat = po_feat.view(B, N, T, self.hidden_dim)
        ro_feat = ro_feat.view(B, N, T, self.hidden_dim)

        cls_valid = frame_token_valid.to(dtype=x.dtype).unsqueeze(-1)  # (B,N,T,1)
        denom = cls_valid.sum(dim=1).clamp_min(1.0)
        po_agg = (po_feat * cls_valid).sum(dim=1) / denom
        ro_agg = (ro_feat * cls_valid).sum(dim=1) / denom

        po_flat = po_agg.reshape(B * T, self.hidden_dim)
        ro_flat = ro_agg.reshape(B * T, self.hidden_dim)

        pose_latent = 0.5 * (po_agg + ro_agg)

        return self._decode_features(po_flat, ro_flat, pose_latent, B, T)

    def _decode_features_without_canonical_pose(
        self,
        po_flat: Tensor,
        ro_flat: Tensor,
        pose_latent: Tensor,
        batch_size: int,
        seq_len: int,
    ) -> dict[str, Tensor]:
        del pose_latent
        return {
            "position": self.position_head(po_flat).view(batch_size, seq_len, 3),
            "rotation": self.rotation_head(ro_flat).view(batch_size, seq_len, 2),
        }

    def _decode_features_with_canonical_pose(
        self,
        po_flat: Tensor,
        ro_flat: Tensor,
        pose_latent: Tensor,
        batch_size: int,
        seq_len: int,
    ) -> dict[str, Tensor]:
        head = cast(CanonicalPoseHead, self.canonical_pose_head)
        return {
            "position": self.position_head(po_flat).view(batch_size, seq_len, 3),
            "rotation": self.rotation_head(ro_flat).view(batch_size, seq_len, 2),
            "canonical_pose": head(pose_latent),
        }
