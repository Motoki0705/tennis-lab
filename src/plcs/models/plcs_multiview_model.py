"""Multi-view PLCS model with camera-time 2D RoPE.

Per-camera token layout:
    [court_m(20), frame_{m,0}(2+17), ..., frame_{m,T-1}(2+17)]
where each frame block is:
    [CLS_po_{m,t}, CLS_ro_{m,t}, player_{m,t,0..16}]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from src.common.models import (
    MoE,
    MoEConfig,
    MultiHeadSelfAttention,
    RMSNorm,
    SwiGLU,
    precompute_freqs_cis_2d,
)
from src.common.models.embeddings import CourtKPUVEmbedding, InvisibleTokenEmbedding, PlayerKPUVEmbedding
from src.plcs.models.components.heads import PositionHead, RotationHead
from src.utils.geometry import NUM_COURT_KP, NUM_HUMAN_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig


class Rope2DTransformerBlock(nn.Module):
    """Transformer block with SDPA and 2D RoPE support."""

    def __init__(
        self,
        *,
        dim: int,
        n_heads: int,
        mlp_inter_dim: int,
        head_dim: int,
        rope_dim: int,
        attn_dropout: float,
        use_moe: bool,
        moe_config: MoEConfig | None,
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
        if use_moe:
            if moe_config is None:
                raise ValueError("use_moe=True requires moe_config.")
            self.ffn: nn.Module = MoE(moe_config)
        else:
            self.ffn = SwiGLU(dim, mlp_inter_dim)

    def forward(
        self,
        x: Tensor,
        residual: Tensor | None,
        *,
        rope2d: tuple[Tensor, Tensor],
        positions_2d: Tensor,
        attn_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if residual is None:
            x_norm = self.attn_norm(x)
            residual = x
        else:
            x_norm, residual = self.attn_norm(x, residual)

        x = self.attn(
            x_norm,
            start_pos=0,
            rope2d=rope2d,
            positions_2d=positions_2d,
            attn_mask=attn_mask,
            is_causal=False,
        )

        x_norm, residual = self.ffn_norm(x, residual)
        x = self.ffn(x_norm)
        return x, residual


class PLCSMultiViewModel(nn.Module):
    """Multi-view PLCS model using camera-time 2D RoPE."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        rope_dim: int | None = None,
        rope_theta: float = 10000.0,
        use_moe: bool = False,
        moe_config: MoEConfig | None = None,
        max_views: int = 8,
        max_seq_len: int = 120,
        invisible_init_std: float = 0.02,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.max_views = int(max_views)
        self.max_seq_len = int(max_seq_len)

        self.frame_block_tokens = 2 + NUM_HUMAN_KP

        head_dim = hidden_dim // num_heads
        rope_dim = head_dim if rope_dim is None else rope_dim
        self.rope_dim = int(rope_dim)
        self.rope_theta = float(rope_theta)

        if self.rope_dim % 4 != 0:
            raise ValueError(f"2D RoPE requires rope_dim % 4 == 0, got {self.rope_dim}")

        if ffn_dim is None:
            ffn_dim = int((8 * hidden_dim) / 3)
            ffn_dim = (ffn_dim + 63) // 64 * 64

        if use_moe and moe_config is None:
            raise ValueError("use_moe=True requires moe_config.")
        if moe_config is not None and moe_config.dim != hidden_dim:
            raise ValueError(f"moe_config.dim={moe_config.dim} must match hidden_dim={hidden_dim}")

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

        self.cls_po_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.cls_ro_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_po_token, std=0.02)
        nn.init.trunc_normal_(self.cls_ro_token, std=0.02)

        self.blocks = nn.ModuleList(
            [
                Rope2DTransformerBlock(
                    dim=hidden_dim,
                    n_heads=num_heads,
                    mlp_inter_dim=ffn_dim,
                    head_dim=head_dim,
                    rope_dim=self.rope_dim,
                    attn_dropout=dropout,
                    use_moe=use_moe,
                    moe_config=moe_config,
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

        freqs_y, freqs_x = precompute_freqs_cis_2d(
            dim=self.rope_dim,
            height=self.max_seq_len + 1,
            width=self.max_views,
            base=self.rope_theta,
            device=None,
        )
        self.register_buffer("freqs_cis_y", freqs_y, persistent=False)
        self.register_buffer("freqs_cis_x", freqs_x, persistent=False)

    @classmethod
    def from_config(cls, config: DictConfig) -> PLCSMultiViewModel:
        """Create model from hydra config."""
        model_cfg = config.get("model", {})

        use_moe = bool(model_cfg.get("use_moe", False))
        moe_cfg = model_cfg.get("moe_config", None)
        moe_config: MoEConfig | None = None
        if use_moe and moe_cfg is not None:
            moe_config = MoEConfig(dim=int(model_cfg.get("hidden_dim", 256)), **dict(moe_cfg))

        return cls(
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_layers=model_cfg.get("num_layers", 6),
            num_heads=model_cfg.get("num_heads", 8),
            ffn_dim=model_cfg.get("ffn_dim", None),
            dropout=model_cfg.get("dropout", 0.1),
            rope_dim=model_cfg.get("rope_dim", None),
            rope_theta=model_cfg.get("rope_theta", 10000.0),
            use_moe=use_moe,
            moe_config=moe_config,
            max_views=model_cfg.get("max_views", 8),
            max_seq_len=model_cfg.get("max_seq_len", 120),
            invisible_init_std=float(model_cfg.get("invisible_init_std", 0.02)),
        )

    def _build_positions_2d(
        self,
        *,
        batch_size: int,
        n_cameras: int,
        seq_len: int,
        device: torch.device,
    ) -> Tensor:
        """Build (B, S, 2) coordinates with axes: (time, camera)."""
        per_cam: list[Tensor] = []
        for cam_idx in range(n_cameras):
            court_time = torch.zeros(NUM_COURT_KP, device=device, dtype=torch.long)
            court_cam = torch.full((NUM_COURT_KP,), cam_idx, device=device, dtype=torch.long)
            court_pos = torch.stack([court_time, court_cam], dim=-1)

            frame_time = torch.arange(seq_len, device=device, dtype=torch.long).repeat_interleave(
                self.frame_block_tokens
            ) + 1
            frame_cam = torch.full(
                (seq_len * self.frame_block_tokens,),
                cam_idx,
                device=device,
                dtype=torch.long,
            )
            frame_pos = torch.stack([frame_time, frame_cam], dim=-1)
            per_cam.append(torch.cat([court_pos, frame_pos], dim=0))

        positions = torch.cat(per_cam, dim=0)
        return positions.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None = None,
        court_vis: Tensor | None = None,
        view_mask: Tensor | None = None,
        seq_mask: Tensor | None = None,
        camera_params: list[list[dict]] | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            human_kp:
                Human 2D keypoints in normalized image UV.
                - Temporal: (B, N, T, 17, 2)
                - Single frame: (B, N, 17, 2)
            court_kp:
                Court 2D keypoints in normalized image UV.
                - Temporal: (B, N, T, 20, 2)
                - Single frame: (B, N, 20, 2)
            human_vis:
                Human keypoint visibility flags aligned with `human_kp`.
                - Temporal: (B, N, T, 17)
                - Single frame: (B, N, 17)
                Each element is interpreted as visible if > 0 (bool/0-1).
                Optional; if None, all human keypoints are treated as visible.
            court_vis:
                Court keypoint visibility flags aligned with `court_kp`.
                - Temporal: (B, N, T, 20)
                - Single frame: (B, N, 20)
                Each element is interpreted as visible if > 0.
                Optional; if None, all court keypoints are treated as visible.
            view_mask:
                Padding mask for camera views. Shape: (B, N), True = valid view.
                Optional; if None, all N views are treated as valid.
            seq_mask:
                Padding mask for sequence frames. Shape: (B, T), True = valid frame.
                Optional; if None, all T frames are treated as valid.
            camera_params:
                Camera metadata per sample/view (currently unused by this model).

        Returns:
            dict:
                - position:
                    - Temporal input: (B, T, 3)
                    - Single-frame input: (B, 3)
                - rotation:
                    - Temporal input: (B, T, 2)
                    - Single-frame input: (B, 2)

        Notes:
            Input ordering is camera-time: (B, N, T, ...). Single-frame input
            is internally converted to T=1 and squeezed back before return.
        """
        del camera_params  # currently unused

        is_temporal = human_kp.dim() == 5
        if not is_temporal:
            human_kp = human_kp.unsqueeze(2)
            court_kp = court_kp.unsqueeze(2)
            if human_vis is not None:
                human_vis = human_vis.unsqueeze(2)
            if court_vis is not None:
                court_vis = court_vis.unsqueeze(2)

        # (B, N, T, K, 2)
        B, N, T = human_kp.shape[:3]
        device = human_kp.device

        if N > self.max_views:
            raise ValueError(f"Number of views N={N} exceeds max_views={self.max_views}.")
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length T={T} exceeds max_seq_len={self.max_seq_len}.")

        if view_mask is not None:
            if view_mask.dim() == 1:
                view_mask = view_mask.unsqueeze(0)
            if view_mask.shape != (B, N):
                raise ValueError(
                    f"view_mask must have shape {(B, N)}, got {tuple(view_mask.shape)}"
                )
            view_valid = view_mask > 0
        else:
            view_valid = torch.ones(B, N, dtype=torch.bool, device=device)

        if seq_mask is not None:
            if seq_mask.dim() == 1:
                seq_mask = seq_mask.unsqueeze(0)
            if seq_mask.shape != (B, T):
                raise ValueError(
                    f"seq_mask must have shape {(B, T)}, got {tuple(seq_mask.shape)}"
                )
            seq_valid = seq_mask > 0
        else:
            seq_valid = torch.ones(B, T, dtype=torch.bool, device=device)

        # court is camera-level (use first frame per camera)
        court_scene = court_kp[:, :, 0, :, :]  # (B,N,20,2)
        court_scene_flat = court_scene.reshape(B * N, NUM_COURT_KP, 2)

        court_vis_scene: Tensor | None = None
        if court_vis is not None:
            court_vis_scene = court_vis[:, :, 0, :].reshape(B * N, NUM_COURT_KP)

        court_tok = self.court_embed(court_scene_flat, court_vis_scene)
        court_tok = court_tok.view(B, N, NUM_COURT_KP, self.hidden_dim)

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
        frame_tok = frame_tok.reshape(B, N, T * self.frame_block_tokens, self.hidden_dim)

        camera_block_tokens = NUM_COURT_KP + T * self.frame_block_tokens
        tokens_per_camera = torch.cat([court_tok, frame_tok], dim=2)  # (B,N,L_cam,D)
        x = tokens_per_camera.reshape(B, N * camera_block_tokens, self.hidden_dim)

        # Build token validity mask from padding masks only.
        frame_token_valid = view_valid.unsqueeze(-1) & seq_valid.unsqueeze(1)
        court_token_valid = view_valid

        court_rep = court_token_valid.unsqueeze(-1).expand(B, N, NUM_COURT_KP)
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

        positions_2d = self._build_positions_2d(
            batch_size=B,
            n_cameras=N,
            seq_len=T,
            device=device,
        )

        freqs_y = self.freqs_cis_y[: T + 1]
        freqs_x = self.freqs_cis_x[:N]
        if freqs_y.device != device:
            freqs_y = freqs_y.to(device)
        if freqs_x.device != device:
            freqs_x = freqs_x.to(device)
        rope2d = (freqs_y, freqs_x)

        residual = None
        for blk in self.blocks:
            x, residual = blk(
                x,
                residual,
                rope2d=rope2d,
                positions_2d=positions_2d,
                attn_mask=attn_mask,
            )
            x = x * token_valid.unsqueeze(-1).to(dtype=x.dtype)

        if residual is None:
            x = self.final_norm(x)
        else:
            x, _ = self.final_norm(x, residual)
        x = x * token_valid.unsqueeze(-1).to(dtype=x.dtype)

        time_offsets = NUM_COURT_KP + torch.arange(T, device=device, dtype=torch.long) * self.frame_block_tokens
        cam_offsets = torch.arange(N, device=device, dtype=torch.long).view(N, 1) * camera_block_tokens
        po_idx = (cam_offsets + time_offsets.view(1, T)).reshape(-1)
        ro_idx = po_idx + 1

        po_feat = x.gather(1, po_idx.view(1, N * T, 1).expand(B, N * T, self.hidden_dim))
        ro_feat = x.gather(1, ro_idx.view(1, N * T, 1).expand(B, N * T, self.hidden_dim))
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

        if not is_temporal:
            position = position.squeeze(1)
            rotation = rotation.squeeze(1)

        return {
            "position": position,
            "rotation": rotation,
        }
