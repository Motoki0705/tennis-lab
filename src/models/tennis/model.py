"""Tennis-DETR: multi-view, multi-player 3D pose model."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from src.models.tennis.config import TennisDetrConfig


class TennisDETR(nn.Module):
    """Detection Transformer for multi-view 3D tennis pose.

    本モデルは ``scene_*.json`` で与えられる 2D キーポイント列から、
    最大 ``num_queries`` 人分の 3D ポーズ時系列を復元することを目的とする。
    すべてのカメラ・すべての時間の検出をフラットなトークン列に変換し、
    Transformer Encoder/Decoder と Object Queries により
    「同一人物の結合 (Association)」と「3D 復元 (Lifting)」を同時に解く。

    Args:
        cfg (TennisDetrConfig): Configuration with Transformer dimensions,
            query count, and embedding table sizes.

    """

    def __init__(self, cfg: TennisDetrConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D = cfg.D_model

        # ---- Input tokenization ----
        joints_in_dim = cfg.num_joints * 2
        court_in_dim = cfg.num_court_points * 2

        self.joint_mlp = nn.Sequential(
            nn.LayerNorm(joints_in_dim),
            nn.Linear(joints_in_dim, D),
            nn.GELU(),
            nn.Linear(D, D),
        )
        self.court_mlp = nn.Sequential(
            nn.LayerNorm(court_in_dim),
            nn.Linear(court_in_dim, D),
            nn.GELU(),
            nn.Linear(D, D),
        )

        # Camera/time positional embeddings.
        self.camera_embed = nn.Embedding(cfg.max_cameras, D)
        self.time_embed = nn.Embedding(cfg.max_frames, D)

        # ---- Transformer encoder / decoder ----
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=cfg.nheads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=D,
            nhead=cfg.nheads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.decoder_layers)

        # Learnable base queries (player slots).
        self.query_embed = nn.Embedding(cfg.num_queries, D)

        # ---- Output heads ----
        self.pose_head = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, cfg.num_joints * 3),
        )
        self.exist_head = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, 1),
        )

    def forward(
        self,
        player_kpts_2d: Tensor,
        player_mask: Tensor,
        court_kpts_2d: Tensor,
    ) -> Mapping[str, Tensor]:
        """Run the Tennis-DETR forward pass.

        Args:
            player_kpts_2d (Tensor): 2Dキーポイント座標。
                Shape: ``[B, T, V, M, J, 2]`` where
                ``B`` はバッチサイズ、
                ``T`` は時間フレーム数、
                ``V`` はカメラ数、
                ``M`` は1画像あたりの最大検出人数、
                ``J`` は関節数。
            player_mask (Tensor): True/False マスク。
                Shape: ``[B, T, V, M]`` で、True が有効な検出、False がパディング。
            court_kpts_2d (Tensor): コート2Dキーポイント。
                Shape: ``[B, V, P, 2]`` where ``P`` はコート点数（通常 20）。

        Returns:
            Mapping[str, Tensor]: 予測結果の辞書で、主に次を含む。

            - ``pose_3d``: 3D ポーズ。
              Shape: ``[B, Q, T, J, 3]``
            - ``exist_conf``: Query ごとの存在確率。
              Shape: ``[B, Q, 1]``

        Raises:
            ValueError: If sequence length exceeds max_frames or number of cameras
                exceeds max_cameras.

        """
        B, T, V, M, J, two = player_kpts_2d.shape
        assert two == 2, "player_kpts_2d last dim must be 2"
        device = player_kpts_2d.device

        if self.cfg.max_frames < T:
            msg = f"sequence length {T} exceeds max_frames={self.cfg.max_frames}"
            raise ValueError(msg)
        if self.cfg.max_cameras < V:
            msg = f"num cameras {V} exceeds max_cameras={self.cfg.max_cameras}"
            raise ValueError(msg)

        # ---- Input tokenization ----
        # Player joints: [B, T, V, M, J*2] -> [B, T, V, M, D]
        joints_flat = player_kpts_2d.reshape(B, T, V, M, J * 2)
        joint_embed = self.joint_mlp(joints_flat)

        # Court: [B, V, P, 2] -> [B, V, D] -> broadcast to [B, T, V, M, D]
        _, _, P, _ = court_kpts_2d.shape
        court_flat = court_kpts_2d.reshape(B, V, P * 2)
        court_embed = self.court_mlp(court_flat)  # [B, V, D]
        court_embed = court_embed[:, None, :, None, :]  # [B, 1, V, 1, D]
        court_embed = court_embed.expand(B, T, V, M, -1)

        # Camera/time embeddings.
        cam_ids = torch.arange(V, device=device)
        cam_embed = self.camera_embed(cam_ids)  # [V, D]
        cam_embed = cam_embed[None, None, :, None, :]  # [1, 1, V, 1, D]
        cam_embed = cam_embed.expand(B, T, V, M, -1)

        time_ids = torch.arange(T, device=device)
        time_embed = self.time_embed(time_ids)  # [T, D]
        time_embed = time_embed[None, :, None, None, :]  # [1, T, 1, 1, D]
        time_embed = time_embed.expand(B, T, V, M, -1)

        tokens = joint_embed + court_embed + cam_embed + time_embed

        # Flatten tokens: [B, T*V*M, D]
        tokens = tokens.reshape(B, T * V * M, self.cfg.D_model)

        # Build key padding mask: True for padding positions.
        if player_mask.dtype != torch.bool:
            player_mask_bool = player_mask.to(dtype=torch.bool)
        else:
            player_mask_bool = player_mask
        key_padding_mask = ~player_mask_bool.reshape(B, T * V * M)

        # ---- Encoder ----
        memory = self.encoder(tokens, src_key_padding_mask=key_padding_mask)

        # ---- Time-aware queries and decoder ----
        Q = self.cfg.num_queries
        query_base = self.query_embed.weight  # [Q, D]
        query_base = query_base[None, :, None, :].expand(B, Q, T, -1)  # [B, Q, T, D]

        # Time embeddings for queries reuse the same table.
        time_embed_q = self.time_embed(time_ids)  # [T, D]
        time_embed_q = time_embed_q[None, None, :, :].expand(B, Q, T, -1)

        queries_time = query_base + time_embed_q  # [B, Q, T, D]
        queries_flat = queries_time.reshape(B, Q * T, self.cfg.D_model)

        decoder_out = self.decoder(
            tgt=queries_flat,
            memory=memory,
            memory_key_padding_mask=key_padding_mask,
        )  # [B, Q*T, D]

        decoder_out_time = decoder_out.reshape(B, Q, T, self.cfg.D_model)

        # ---- 3D pose head ----
        pose_3d = self.pose_head(decoder_out_time)  # [B, Q, T, J*3]
        pose_3d = pose_3d.reshape(B, Q, T, self.cfg.num_joints, 3)

        # ---- Existence head ----
        exist_feat = decoder_out_time.mean(dim=2)  # [B, Q, D]
        exist_logit = self.exist_head(exist_feat)  # [B, Q, 1]
        exist_conf = torch.sigmoid(exist_logit)

        return {
            "pose_3d": pose_3d,
            "exist_conf": exist_conf,
        }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TennisDetrConfig()
    model = TennisDETR(cfg)
    model.to(device)

    player_kpts_2d = torch.randn(1, 32, 8, 30, 17, 2)  # [B, T, V, M, J, 2]
    player_mask = torch.randint(0, 10, (1, 32, 8, 30)).bool()
    court_kpts_2d = torch.randn(1, 8, 20, 2)

    for _ in range(100):
        with torch.no_grad():
            outputs = model(
                player_kpts_2d.to(device),
                player_mask.to(device),
                court_kpts_2d.to(device),
            )
        print(outputs["pose_3d"].shape)
