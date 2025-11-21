"""Tennis-DETR v2.5: enhanced hierarchical encoder with camera/time embeddings.

This model is a variant of :class:`TennisDETR_v2` that adds explicit camera and
frame positional embeddings to the encoder input tokens. The overall task and
I/O shapes are identical to v2:

- Inputs
  - player_kpts_2d: [B, T, V, M, J, 2]
  - player_mask:    [B, T, V, M]
  - court_kpts_2d:  [B, V, 20, 2]
- Outputs
  - canonical_pose: [B, Q, T, J, 3]
  - root_trans:     [B, Q, T, 3]
  - root_rot:       [B, Q, T, 2]
  - pose_3d:        [B, Q, T, J, 3]
  - exist_logit:    [B, Q, 1]
  - exist_conf:     [B, Q, 1]
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from src.models.tennis_multi_cam_3d_pose.config_v2 import TennisDetrV2Config


class TennisDETR_v2_5(nn.Module):
    """v2.5 variant of Tennis-DETR with camera/time-aware encoder tokens.

    The architecture closely follows :class:`TennisDETR_v2`, but adds
    camera and frame embeddings directly to the encoder input tokens before
    the hierarchical intra/inter/temporal encoder stack.
    """

    def __init__(self, cfg: TennisDetrV2Config) -> None:
        super().__init__()
        self.cfg = cfg
        D = cfg.D_model

        # ---- Encoders (hierarchical) ----
        self.joint_in_dim = cfg.num_joints * 2
        self.court_mlp = nn.Sequential(
            nn.LayerNorm(cfg.num_court_points * 2),
            nn.Linear(cfg.num_court_points * 2, D),
            nn.GELU(),
        )
        fusion_dim = self.joint_in_dim + D
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, D),
            nn.GELU(),
            nn.Linear(D, D),
        )

        # Camera / time positional embeddings for encoder tokens.
        self.camera_embed = nn.Embedding(cfg.max_cameras, D)

        # Hierarchical encoders.
        self.intra_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                D, cfg.nheads, cfg.dim_feedforward, cfg.dropout, batch_first=True
            ),
            num_layers=cfg.intra_layers,
        )
        self.inter_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                D, cfg.nheads, cfg.dim_feedforward, cfg.dropout, batch_first=True
            ),
            num_layers=cfg.inter_layers,
        )

        self.time_embed = nn.Embedding(cfg.max_frames, D)
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                D, cfg.nheads, cfg.dim_feedforward, cfg.dropout, batch_first=True
            ),
            num_layers=cfg.temporal_layers,
        )

        # ---- Decoder ----
        self.query_embed = nn.Embedding(cfg.num_queries, D)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                D, cfg.nheads, cfg.dim_feedforward, cfg.dropout, batch_first=True
            ),
            num_layers=cfg.decoder_layers,
        )

        # ---- Output Heads (decomposed) ----

        # 1. Canonical Pose Head (root-relative, no rotation).
        self.canonical_head = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, cfg.num_joints * 3),
        )

        # 2. Root Head (Translation + Rotation).
        self.root_head = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 3 + 2),
        )

        # 3. Existence head.
        self.exist_head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1))

    def forward(
        self,
        player_kpts_2d: Tensor,
        player_mask: Tensor,
        court_kpts_2d: Tensor,
    ) -> Mapping[str, Tensor]:
        """Forward pass of the v2.5 model.

        Args:
            player_kpts_2d (Tensor): Player keypoints in 2D with shape [B, T, V, M, J, 2].
            player_mask (Tensor): Player mask with shape [B, T, V, M].
            court_kpts_2d (Tensor): Court keypoints in 2D with shape [B, V, C, 2].

        Returns:
            Mapping[str, Tensor]: Dictionary containing model outputs including
            canonical_pose, root_trans, root_rot, pose_3d, exist_logit, exist_conf.

        Raises:
            ValueError: If sequence length exceeds max_frames or number of cameras exceeds max_cameras.

        """
        B, T, V, M, J, _ = player_kpts_2d.shape
        device = player_kpts_2d.device

        if self.cfg.max_frames < T:
            msg = f"sequence length {T} exceeds max_frames={self.cfg.max_frames}"
            raise ValueError(msg)
        if self.cfg.max_cameras < V:
            msg = f"num cameras {V} exceeds max_cameras={self.cfg.max_cameras}"
            raise ValueError(msg)

        # --- Encoding Steps ---
        court_feat = self.court_mlp(court_kpts_2d.reshape(B, V, -1))
        court_feat_exp = court_feat[:, None, :, None, :].expand(B, T, V, M, -1)
        player_flat = player_kpts_2d.reshape(B, T, V, M, -1)
        tokens = self.fusion_mlp(torch.cat([player_flat, court_feat_exp], dim=-1))

        # Add camera and frame embeddings to encoder tokens.
        cam_ids = torch.arange(V, device=device)
        cam_embed = self.camera_embed(cam_ids)[None, None, :, None, :]  # [1,1,V,1,D]
        cam_embed = cam_embed.expand(B, T, V, M, -1)

        time_ids = torch.arange(T, device=device)
        t_embed_tokens = self.time_embed(time_ids)[None, :, None, None, :]
        t_embed_tokens = t_embed_tokens.expand(B, T, V, M, -1)

        tokens = tokens + cam_embed + t_embed_tokens

        # 1. Intra (within players per (t, v)).
        tokens_intra = tokens.reshape(B * T * V, M, -1)
        mask_intra = player_mask.reshape(B * T * V, M)
        mem_intra = self.intra_encoder(tokens_intra, src_key_padding_mask=~mask_intra)

        # 2. Inter (across players and cameras per time).
        mem_intra_reshaped = mem_intra.reshape(B * T, V * M, -1)
        mask_inter = player_mask.reshape(B * T, V * M)
        mem_inter = self.inter_encoder(
            mem_intra_reshaped, src_key_padding_mask=~mask_inter
        )

        # 3. Temporal (across time with global tokens).
        mem_inter_reshaped = mem_inter.reshape(B, T, V * M, -1)
        t_embed = self.time_embed(time_ids)[None, :, None, :]
        tokens_global = (mem_inter_reshaped + t_embed).reshape(B, T * V * M, -1)
        mask_global = player_mask.reshape(B, T * V * M)
        mem_final = self.temporal_encoder(
            tokens_global, src_key_padding_mask=~mask_global
        )

        # --- Decoding ---
        Q = self.cfg.num_queries
        query_base = self.query_embed.weight[None, :, None, :].expand(B, Q, T, -1)
        t_embed_q = self.time_embed(time_ids)[None, None, :, :]
        queries = (query_base + t_embed_q).reshape(B, Q * T, -1)

        dec_out = self.decoder(queries, mem_final, memory_key_padding_mask=~mask_global)
        dec_out = dec_out.reshape(B, Q, T, -1)  # [B, Q, T, D]

        # ==========================================
        # Output Decomposition & Reconstruction
        # ==========================================

        # 1. Predict Canonical Pose (local, root-relative).
        canonical_pose = self.canonical_head(dec_out).reshape(
            B, Q, T, self.cfg.num_joints, 3
        )

        # 2. Predict Root Transform (global translation + yaw rotation).
        root_params = self.root_head(dec_out)  # [B, Q, T, 5]
        root_trans = root_params[..., :3]
        root_rot_raw = root_params[..., 3:]

        root_rot = torch.nn.functional.normalize(root_rot_raw, dim=-1)
        cos_theta = root_rot[..., 0:1]
        sin_theta = root_rot[..., 1:2]

        # 3. Reconstruct Global Pose.
        x_c = canonical_pose[..., 0]
        y_c = canonical_pose[..., 1]
        z_c = canonical_pose[..., 2]

        x_r = x_c * cos_theta - y_c * sin_theta
        y_r = x_c * sin_theta + y_c * cos_theta
        z_r = z_c

        rotated_pose = torch.stack([x_r, y_r, z_r], dim=-1)

        global_pose = rotated_pose + root_trans.unsqueeze(-2)

        # Existence (logit + probability).
        exist_feat = dec_out.mean(dim=2)  # [B, Q, D]
        exist_logit = self.exist_head(exist_feat)  # [B, Q, 1]
        exist_conf = torch.sigmoid(exist_logit)

        return {
            "canonical_pose": canonical_pose,
            "root_trans": root_trans,
            "root_rot": root_rot,
            "pose_3d": global_pose,
            "exist_logit": exist_logit,
            "exist_conf": exist_conf,
        }


if __name__ == "__main__":  # pragma: no cover - manual shape check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TennisDetrV2Config()
    model = TennisDETR_v2_5(cfg).to(device)

    player_kpts_2d = torch.randn(1, 32, 4, 30, 20, 2, device=device)
    player_mask = torch.randint(0, 2, (1, 32, 4, 30), dtype=torch.bool, device=device)
    court_kpts_2d = torch.randn(1, 4, 20, 2, device=device)

    with torch.no_grad():
        outputs = model(player_kpts_2d, player_mask, court_kpts_2d)

    print("TennisDETR_v2_5 output shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {tuple(value.shape)}")
