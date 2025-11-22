"""Tennis-DETR v3: Track-aware scene transformer for multi-view 3D pose.

This model keeps the same task and I/O interface as v2/v2.5 but adds a
track-aware temporal encoder over per-player query tracks.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from src.models.tennis_multi_cam_3d_pose.config_v3 import TennisDetrV3Config


class TennisDETR_v3(nn.Module):
    """Track-aware variant of Tennis-DETR for multi-view 3D pose.

    Inputs
    ------
    - player_kpts_2d: [B, T, V, M, J, 2]
    - player_mask:    [B, T, V, M]
    - court_kpts_2d:  [B, V, C, 2]

    Outputs
    -------
    - canonical_pose: [B, Q, T, J, 3]
    - root_trans:     [B, Q, T, 3]
    - root_rot:       [B, Q, T, 2]
    - pose_3d:        [B, Q, T, J, 3]
    - exist_logit:    [B, Q, 1]
    - exist_conf:     [B, Q, 1]
    """

    def __init__(self, cfg: TennisDetrV3Config) -> None:
        super().__init__()
        self.cfg = cfg
        D = cfg.D_model

        # ---- Scene encoder (hierarchical over detections) ----
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

        self.intra_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                D,
                cfg.nheads,
                cfg.dim_feedforward,
                cfg.dropout,
                batch_first=True,
            ),
            num_layers=cfg.intra_layers,
        )
        self.inter_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                D,
                cfg.nheads,
                cfg.dim_feedforward,
                cfg.dropout,
                batch_first=True,
            ),
            num_layers=cfg.inter_layers,
        )

        # Time embeddings for tokens and queries.
        self.time_embed = nn.Embedding(cfg.max_frames, D)

        # ---- Track-aware decoder ----
        self.query_embed = nn.Embedding(cfg.num_queries, D)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                D,
                cfg.nheads,
                cfg.dim_feedforward,
                cfg.dropout,
                batch_first=True,
            ),
            num_layers=cfg.decoder_layers,
        )

        # Temporal encoder over per-query tracks [T, D].
        self.track_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                D,
                cfg.nheads,
                cfg.dim_feedforward,
                cfg.dropout,
                batch_first=True,
            ),
            num_layers=cfg.temporal_layers,
        )

        # ---- Output heads (decomposed) ----
        self.canonical_head = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, cfg.num_joints * 3),
        )
        self.root_head = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 3 + 2),
        )
        self.exist_head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1))

        self.canonical_in_proj = nn.Linear(cfg.num_joints * 3, D)
        self.root_trans_in_proj = nn.Linear(3, D)
        self.root_rot_in_proj = nn.Linear(2, D)

        self.denoise_canonical_head = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, cfg.num_joints * 3),
        )
        self.denoise_root_trans_head = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 3),
        )
        self.denoise_root_rot_head = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 2),
        )

    def forward(
        self,
        player_kpts_2d: Tensor,
        player_mask: Tensor,
        court_kpts_2d: Tensor,
    ) -> Mapping[str, Tensor]:
        """フォワードパスを実行する.

        Args:
            player_kpts_2d (Tensor): プレイヤーの2Dキーポイントテンソル
            player_mask (Tensor): プレイヤーマスクテンソル
            court_kpts_2d (Tensor): コートの2Dキーポイントテンソル

        Returns:
            Mapping[str, Tensor]: 出力テンソルのマッピング

        Raises:
            ValueError: シーケンス長またはカメラ数が最大値を超える場合

        """
        B, T, V, M, J, _ = player_kpts_2d.shape
        device = player_kpts_2d.device

        if self.cfg.max_frames < T:
            msg = f"sequence length {T} exceeds max_frames={self.cfg.max_frames}"
            raise ValueError(msg)
        if self.cfg.max_cameras < V:
            msg = f"num cameras {V} exceeds max_cameras={self.cfg.max_cameras}"
            raise ValueError(msg)

        # ---- Encode scene detections ----
        court_feat = self.court_mlp(court_kpts_2d.reshape(B, V, -1))
        court_feat_exp = court_feat[:, None, :, None, :].expand(B, T, V, M, -1)
        player_flat = player_kpts_2d.reshape(B, T, V, M, -1)
        tokens = self.fusion_mlp(torch.cat([player_flat, court_feat_exp], dim=-1))

        tokens_intra = tokens.reshape(B * T * V, M, -1)
        mask_intra = player_mask.reshape(B * T * V, M)
        mem_intra = self.intra_encoder(tokens_intra, src_key_padding_mask=~mask_intra)

        mem_inter_in = mem_intra.reshape(B * T, V * M, -1)
        mask_inter = player_mask.reshape(B * T, V * M)
        mem_inter = self.inter_encoder(mem_inter_in, src_key_padding_mask=~mask_inter)

        memory = mem_inter.reshape(B, T * V * M, -1)
        mask_global = player_mask.reshape(B, T * V * M)

        # ---- Decode with time-aware queries ----
        Q = self.cfg.num_queries
        time_ids = torch.arange(T, device=device)

        query_base = self.query_embed.weight[None, :, None, :].expand(B, Q, T, -1)
        t_embed_q = self.time_embed(time_ids)[None, None, :, :]
        queries = (query_base + t_embed_q).reshape(B, Q * T, -1)

        dec_out = self.decoder(queries, memory, memory_key_padding_mask=~mask_global)
        dec_out = dec_out.reshape(B, Q, T, -1)

        # ---- Track-aware temporal encoding over query tracks ----
        tracks = dec_out.reshape(B * Q, T, -1)
        tracks_enc = self.track_encoder(tracks)
        tracks_enc = tracks_enc.reshape(B, Q, T, -1)

        # ---- Decomposed outputs ----
        canonical_pose = self.canonical_head(tracks_enc).reshape(
            B, Q, T, self.cfg.num_joints, 3
        )

        root_params = self.root_head(tracks_enc)
        root_trans = root_params[..., :3]
        root_rot_raw = root_params[..., 3:]
        root_rot = torch.nn.functional.normalize(root_rot_raw, dim=-1)

        cos_theta = root_rot[..., 0:1]
        sin_theta = root_rot[..., 1:2]

        x_c = canonical_pose[..., 0]
        y_c = canonical_pose[..., 1]
        z_c = canonical_pose[..., 2]

        x_r = x_c * cos_theta - y_c * sin_theta
        y_r = x_c * sin_theta + y_c * cos_theta
        z_r = z_c

        rotated = torch.stack([x_r, y_r, z_r], dim=-1)
        global_pose = rotated + root_trans.unsqueeze(-2)

        exist_feat = tracks_enc.mean(dim=2)
        exist_logit = self.exist_head(exist_feat)
        exist_conf = torch.sigmoid(exist_logit)

        return {
            "canonical_pose": canonical_pose,
            "root_trans": root_trans,
            "root_rot": root_rot,
            "pose_3d": global_pose,
            "exist_logit": exist_logit,
            "exist_conf": exist_conf,
            "tracks_enc": tracks_enc,
        }

    def _build_track_tokens(
        self,
        canonical: Tensor,
        root_trans: Tensor,
        root_rot: Tensor,
    ) -> Tensor:
        N, T, J, _ = canonical.shape
        canon_flat = canonical.reshape(N, T, J * 3)
        canon_embed = self.canonical_in_proj(canon_flat)
        trans_embed = self.root_trans_in_proj(root_trans)
        rot_embed = self.root_rot_in_proj(root_rot)
        tokens = canon_embed + trans_embed + rot_embed
        time_ids = torch.arange(T, device=canonical.device)
        t_embed = self.time_embed(time_ids)[None, :, :].expand(N, T, -1)
        tokens = tokens + t_embed
        return tokens

    def denoise_canonical(
        self,
        canonical_noisy: Tensor,
        root_trans_gt: Tensor,
        root_rot_gt: Tensor,
    ) -> Tensor:
        """Denoise canonical pose tracks while keeping roots fixed.

        Args:
            canonical_noisy (Tensor): Noisy canonical pose tensor of shape [N, T, J, 3].
            root_trans_gt (Tensor): Ground-truth root translations [N, T, 3].
            root_rot_gt (Tensor): Ground-truth root rotations [N, T, 2].

        Returns:
            Tensor: Denoised canonical pose tensor [N, T, J, 3].

        """
        tokens = self._build_track_tokens(canonical_noisy, root_trans_gt, root_rot_gt)
        tracks_enc = self.track_encoder(tokens)
        N, T, J, _ = canonical_noisy.shape
        canon_flat = self.denoise_canonical_head(tracks_enc)
        canonical_out = canon_flat.reshape(N, T, J, 3)
        return canonical_out

    def denoise_root_trans(
        self,
        canonical_gt: Tensor,
        root_trans_noisy: Tensor,
        root_rot_gt: Tensor,
    ) -> Tensor:
        """Denoise root translations while keeping canonical/rot fixed.

        Args:
            canonical_gt (Tensor): Ground-truth canonical poses [N, T, J, 3].
            root_trans_noisy (Tensor): Noisy root translations [N, T, 3].
            root_rot_gt (Tensor): Ground-truth root rotations [N, T, 2].

        Returns:
            Tensor: Denoised root translations [N, T, 3].

        """
        tokens = self._build_track_tokens(canonical_gt, root_trans_noisy, root_rot_gt)
        tracks_enc = self.track_encoder(tokens)
        root_trans_out = self.denoise_root_trans_head(tracks_enc)
        return root_trans_out

    def denoise_root_rot(
        self,
        canonical_gt: Tensor,
        root_trans_gt: Tensor,
        root_rot_noisy: Tensor,
    ) -> Tensor:
        """Denoise root rotations while keeping canonical/trans fixed.

        Args:
            canonical_gt (Tensor): Ground-truth canonical poses [N, T, J, 3].
            root_trans_gt (Tensor): Ground-truth root translations [N, T, 3].
            root_rot_noisy (Tensor): Noisy root rotations [N, T, 2].

        Returns:
            Tensor: Denoised root rotations [N, T, 2].

        """
        tokens = self._build_track_tokens(canonical_gt, root_trans_gt, root_rot_noisy)
        tracks_enc = self.track_encoder(tokens)
        root_rot_out = self.denoise_root_rot_head(tracks_enc)
        return root_rot_out
