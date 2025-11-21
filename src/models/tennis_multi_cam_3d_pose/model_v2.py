"""Tennis-DETR v2: Decomposed Pose (Canonical + Global Root)."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from src.models.tennis_multi_cam_3d_pose.config_v2 import TennisDetrV2Config


class TennisDETR_v2(nn.Module):
    """Canonical 3D Pose と Global Root (Position + Orientation) を分離して推定するモデル.

    出力:
      1. canonical_pose: [B, Q, T, J, 3] (ルート相対、回転なし)
      2. root_trans:     [B, Q, T, 3]    (コート上の絶対座標 x, y, z)
      3. root_rot:       [B, Q, T, 2]    (向き cos, sin)
      4. global_pose:    [B, Q, T, J, 3] (再構成された絶対座標)
    """

    def __init__(self, cfg: TennisDetrV2Config) -> None:
        super().__init__()
        self.cfg = cfg
        D = cfg.D_model

        # ---- Encoders (階層構造) ----
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

        # 階層エンコーダ
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

        # ---- Output Heads (分離) ----

        # 1. Canonical Pose Head
        # 相対座標を出力。ルート(腰)は原点にある前提。
        self.canonical_head = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, cfg.num_joints * 3),
        )

        # 2. Root Head (Translation + Rotation)
        # Output dim = 3 (x, y, z) + 2 (cos, sin) = 5
        self.root_head = nn.Sequential(
            nn.LayerNorm(D), nn.Linear(D, D), nn.GELU(), nn.Linear(D, 3 + 2)
        )

        self.exist_head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1))

    def forward(
        self,
        player_kpts_2d: Tensor,
        player_mask: Tensor,
        court_kpts_2d: Tensor,
    ) -> Mapping[str, Tensor]:
        """Forward pass of the model.

        Args:
            player_kpts_2d (Tensor): Player keypoints in 2D with shape [B, T, V, M, J, 2].
            player_mask (Tensor): Player mask with shape [B, T, V, M].
            court_kpts_2d (Tensor): Court keypoints in 2D with shape [B, T, V, C, 2].

        Returns:
            Mapping[str, Tensor]: Dictionary containing model outputs including
                canonical_pose, root_trans, root_rot, and global_pose.

        """
        B, T, V, M, J, _ = player_kpts_2d.shape
        device = player_kpts_2d.device

        # --- Encoding Steps ---
        court_feat = self.court_mlp(court_kpts_2d.reshape(B, V, -1))
        court_feat_exp = court_feat[:, None, :, None, :].expand(B, T, V, M, -1)
        player_flat = player_kpts_2d.reshape(B, T, V, M, -1)
        tokens = self.fusion_mlp(torch.cat([player_flat, court_feat_exp], dim=-1))

        # 1. Intra (プレーヤー内)
        tokens_intra = tokens.reshape(B * T * V, M, -1)
        mask_intra = player_mask.reshape(B * T * V, M)
        mem_intra = self.intra_encoder(tokens_intra, src_key_padding_mask=~mask_intra)

        # 2. Inter (プレーヤー間)
        mem_intra_reshaped = mem_intra.reshape(B * T, V * M, -1)
        mask_inter = player_mask.reshape(B * T, V * M)
        mem_inter = self.inter_encoder(
            mem_intra_reshaped, src_key_padding_mask=~mask_inter
        )

        # 3. Temporal (時間)
        mem_inter_reshaped = mem_inter.reshape(B, T, V * M, -1)
        time_ids = torch.arange(T, device=device)
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

        # 1. Predict Canonical Pose (Local)
        # [B, Q, T, J, 3]
        canonical_pose = self.canonical_head(dec_out).reshape(
            B, Q, T, self.cfg.num_joints, 3
        )

        # 2. Predict Root Transform (Global)
        root_params = self.root_head(dec_out)  # [B, Q, T, 5]

        root_trans = root_params[..., :3]  # (x, y, z)
        root_rot_raw = root_params[..., 3:]  # (cos, sin) 未正規化

        # Normalize rotation vector to unit length (cos^2 + sin^2 = 1)
        root_rot = torch.nn.functional.normalize(root_rot_raw, dim=-1)
        cos_theta = root_rot[..., 0:1]
        sin_theta = root_rot[..., 1:2]

        # 3. Reconstruct Global Pose
        # Formula: P_global = (R @ P_canonical.T).T + T
        # Rotation Matrix (z-axis rotation):
        # [[cos, -sin, 0],
        #  [sin,  cos, 0],
        #  [  0,    0, 1]]

        # 手動で行列演算を展開して適用 (Batch処理のため)
        x_c = canonical_pose[..., 0]
        y_c = canonical_pose[..., 1]
        z_c = canonical_pose[..., 2]

        # Rotate (Yaw rotation only)
        x_r = x_c * cos_theta - y_c * sin_theta
        y_r = x_c * sin_theta + y_c * cos_theta
        z_r = z_c

        rotated_pose = torch.stack([x_r, y_r, z_r], dim=-1)

        # Translate
        # root_trans: [B, Q, T, 3] -> expand to [B, Q, T, J, 3]
        global_pose = rotated_pose + root_trans.unsqueeze(-2)

        # Existence (logit + probability)
        exist_feat = dec_out.mean(dim=2)  # [B, Q, D]
        exist_logit = self.exist_head(exist_feat)  # [B, Q, 1]
        exist_conf = torch.sigmoid(exist_logit)  # [B, Q, 1]

        return {
            "canonical_pose": canonical_pose,  # Loss計算用: 相対座標の正解と比較
            "root_trans": root_trans,  # Loss計算用: ルート位置の正解と比較
            "root_rot": root_rot,  # Loss計算用: 向きの正解と比較
            "pose_3d": global_pose,  # Loss計算用: 絶対座標の正解と比較
            "exist_logit": exist_logit,
            "exist_conf": exist_conf,
        }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TennisDetrV2Config()
    model = TennisDETR_v2(cfg)
    model.to(device)

    player_kpts_2d = torch.randn(1, 32, 8, 30, 20, 2).to(device)  # [B, T, V, M, J, 2]
    player_mask = torch.randint(0, 10, (1, 32, 8, 30)).bool().to(device)
    court_kpts_2d = torch.randn(1, 8, 20, 2).to(device)

    with torch.no_grad():
        outputs = model(
            player_kpts_2d,
            player_mask,
            court_kpts_2d,
        )

    print("TennisDETR_v2 output shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
