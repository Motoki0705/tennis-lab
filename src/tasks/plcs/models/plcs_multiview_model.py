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
    MultiHeadSelfAttention,
    RMSNorm,
    precompute_freqs_cis_nd,
)
from src.utils.models.components.ffn_layers import MLP, SwiGLU, default_ffn_dim
from src.utils.models.embeddings import (
    CourtKPUVEmbedding,
    InvisibleTokenEmbedding,
    PlayerKPUVEmbedding,
)
from src.utils.schema.court import NUM_COURT_KP
from src.utils.schema.player import NUM_HUMAN_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig


class RoPETransformerBlock(nn.Module):
    """Transformer block with SDPA and generic RoPE support."""

    def __init__(
        self,
        *,
        dim: int,
        n_heads: int,
        ffn_dim: int,
        head_dim: int,
        rope_dim: int,
        attn_dropout: float,
        ffn_type: Literal["swiglu", "mlp"],
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim=dim,
            n_heads=n_heads,
            head_dim=head_dim,
            rope_dim=rope_dim,
            attn_dropout=attn_dropout,
        )

        self.ffn_norm = RMSNorm(dim)
        self.ffn = self._build_ffn(dim=dim, ffn_dim=ffn_dim, ffn_type=ffn_type)

    @staticmethod
    def _build_ffn(
        *,
        dim: int,
        ffn_dim: int,
        ffn_type: Literal["swiglu", "mlp"],
    ) -> nn.Module:
        if ffn_type == "swiglu":
            return SwiGLU(dim, ffn_dim)
        if ffn_type == "mlp":
            return MLP(dim, ffn_dim)
        raise ValueError(f"Unsupported ffn_type={ffn_type}")

    def forward(
        self,
        x: Tensor,
        *,
        freqs_cis: Tensor,
        attn_mask: Tensor | None,
    ) -> Tensor:
        x_attn = x + self.attn(
            self.attn_norm(x),
            freqs_cis=freqs_cis,
            attn_mask=attn_mask,
        )
        x_fnn = x_attn + cast(Tensor, self.ffn(self.ffn_norm(x_attn)))
        return x_fnn


class PLCSMultiViewModel(nn.Module):
    """Multi-view PLCS model using 3-axis interleaved MROPE."""

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
        rope_theta_camera: float | None = None,
        rope_theta_type: float = 100.0,
        ffn_type: Literal["swiglu", "mlp"] = "swiglu",
        predict_canonical_pose: bool = False,
        max_views: int = 8,
        max_seq_len: int = 120,
        invisible_init_std: float = 0.02,
        num_court_tokens: int = NUM_COURT_KP,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.predict_canonical_pose = bool(predict_canonical_pose)
        self.max_views = int(max_views)
        self.max_seq_len = int(max_seq_len)
        self.num_court_tokens = int(num_court_tokens)

        self.frame_block_tokens = 2 + NUM_HUMAN_KP

        head_dim = hidden_dim // num_heads
        rope_dim = head_dim if rope_dim is None else rope_dim
        self.rope_dim = int(rope_dim)
        self.rope_theta = float(rope_theta)
        self.rope_bases = (
            float(self.rope_theta if rope_theta_time is None else rope_theta_time),
            float(self.rope_theta if rope_theta_camera is None else rope_theta_camera),
            float(rope_theta_type),
        )

        self._validate_init_args(rope_dim=self.rope_dim)

        if ffn_dim is None:
            ffn_dim = default_ffn_dim(hidden_dim)

        self.invisible_token = InvisibleTokenEmbedding(
            dim=hidden_dim, init_std=invisible_init_std
        )
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

        self.cls_po_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.cls_ro_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_po_token, std=0.02)
        nn.init.trunc_normal_(self.cls_ro_token, std=0.02)

        self.blocks = nn.ModuleList(
            [
                RoPETransformerBlock(
                    dim=hidden_dim,
                    n_heads=num_heads,
                    ffn_dim=ffn_dim,
                    head_dim=head_dim,
                    rope_dim=self.rope_dim,
                    attn_dropout=dropout,
                    ffn_type=ffn_type,
                )
                for _ in range(num_layers)
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
        self.canonical_pose_head = None
        if self.predict_canonical_pose:
            self.canonical_pose_head = CanonicalPoseHead(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim // 2,
                num_layers=2,
                dropout=dropout,
            )

    @staticmethod
    def _validate_init_args(*, rope_dim: int) -> None:
        if rope_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even rope_dim, got {rope_dim}")

    @classmethod
    def from_config(cls, config: DictConfig) -> PLCSMultiViewModel:
        """Create model from hydra config."""
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})

        return cls(
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_layers=model_cfg.get("num_layers", 6),
            num_heads=model_cfg.get("num_heads", 8),
            ffn_dim=model_cfg.get("ffn_dim", None),
            dropout=model_cfg.get("dropout", 0.1),
            rope_dim=model_cfg.get("rope_dim", None),
            rope_theta=model_cfg.get("rope_theta", 10000.0),
            rope_theta_time=model_cfg.get("rope_theta_time", None),
            rope_theta_camera=model_cfg.get("rope_theta_camera", None),
            rope_theta_type=model_cfg.get("rope_theta_type", 100.0),
            ffn_type=cast(
                Literal["swiglu", "mlp"], str(model_cfg.get("ffn_type", "swiglu"))
            ),
            predict_canonical_pose=bool(model_cfg.get("predict_canonical_pose", False)),
            max_views=model_cfg.get("max_views", 8),
            max_seq_len=model_cfg.get("max_seq_len", 120),
            invisible_init_std=float(model_cfg.get("invisible_init_std", 0.02)),
            num_court_tokens=int(data_cfg.get("num_court_kp", NUM_COURT_KP)),
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
        human_vis: Tensor | None = None,
        human_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
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
            human_mask:
                Padding mask with shape (B, N, T), True/1 for valid tokens.
                Optional; if None, all views/frames are treated as valid.

        Returns:
            dict:
                - position: (B, T, 3)
                - rotation: (B, T, 2)
        """
        B, N, T = self._validate_forward_inputs(
            human_kp=human_kp,
            court_kp=court_kp,
            human_vis=human_vis,
            human_mask=human_mask,
            court_vis=court_vis,
        )
        device = human_kp.device

        if human_mask is not None:
            token_mask = human_mask > 0
            view_valid = token_mask.any(dim=2)
            seq_valid = token_mask.any(dim=1)
        else:
            view_valid = torch.ones(B, N, dtype=torch.bool, device=device)
            seq_valid = torch.ones(B, T, dtype=torch.bool, device=device)

        # court is camera-level (use first frame per camera)
        court_scene = court_kp[:, :, 0, :, :]  # (B,N,K,2)
        K = court_scene.shape[2]
        court_scene_flat = court_scene.reshape(B * N, K, 2)

        court_vis_scene: Tensor | None = None
        if court_vis is not None:
            court_vis_scene = court_vis[:, :, 0, :].reshape(B * N, K)

        court_tok = self.court_embed(court_scene_flat, court_vis_scene)
        court_tok = court_tok.view(B, N, K, self.hidden_dim)

        # player tokens per (camera,time,kp)
        human_flat = human_kp.reshape(B * N * T, NUM_HUMAN_KP, 2)

        human_vis_flat: Tensor | None = None
        if human_vis is not None:
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
        frame_token_valid = view_valid.unsqueeze(-1) & seq_valid.unsqueeze(1)
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
        freqs_cis = precompute_freqs_cis_nd(
            dim=self.rope_dim,
            pos=positions_3d,
            base=self.rope_bases,
        )

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

        position = self.position_head(po_flat).view(B, T, 3)
        rotation = self.rotation_head(ro_flat).view(B, T, 2)
        pose_latent = 0.5 * (po_agg + ro_agg)

        out = {
            "position": position,
            "rotation": rotation,
        }
        if self.predict_canonical_pose and self.canonical_pose_head is not None:
            out["canonical_pose"] = self.canonical_pose_head(pose_latent)
        return out

    def _validate_forward_inputs(
        self,
        *,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None,
        human_mask: Tensor | None,
        court_vis: Tensor | None,
    ) -> tuple[int, int, int]:
        if human_kp.dim() != 5:
            raise ValueError(
                "PLCSMultiViewModel expects human_kp as (B,N,T,17,2), "
                f"got shape {tuple(human_kp.shape)}"
            )
        if court_kp.dim() != 5:
            raise ValueError(
                "PLCSMultiViewModel expects court_kp as "
                f"(B,N,T,{self.num_court_tokens},2), "
                f"got shape {tuple(court_kp.shape)}"
            )
        if court_kp.shape[-2] != self.num_court_tokens:
            raise ValueError(
                f"Expected court_kp with K={self.num_court_tokens}, got K={court_kp.shape[-2]}."
            )
        if human_vis is not None and human_vis.dim() != 4:
            raise ValueError(
                "PLCSMultiViewModel expects human_vis as (B,N,T,17), "
                f"got shape {tuple(human_vis.shape)}"
            )
        if court_vis is not None and court_vis.dim() != 4:
            raise ValueError(
                "PLCSMultiViewModel expects court_vis as "
                f"(B,N,T,{self.num_court_tokens}), "
                f"got shape {tuple(court_vis.shape)}"
            )
        if court_vis is not None and court_vis.shape[-1] != self.num_court_tokens:
            raise ValueError(
                f"Expected court_vis with K={self.num_court_tokens}, got K={court_vis.shape[-1]}."
            )

        batch_size, n_cams, seq_len = human_kp.shape[:3]
        if self.max_views < n_cams:
            raise ValueError(
                f"Number of views N={n_cams} exceeds max_views={self.max_views}."
            )
        if self.max_seq_len < seq_len:
            raise ValueError(
                f"Sequence length T={seq_len} exceeds max_seq_len={self.max_seq_len}."
            )

        if human_mask is not None and (
            human_mask.dim() != 3 or human_mask.shape != (batch_size, n_cams, seq_len)
        ):
            raise ValueError(
                "human_mask for multiview models must be (B,N,T), "
                f"got {tuple(human_mask.shape)}"
            )
        return batch_size, n_cams, seq_len
