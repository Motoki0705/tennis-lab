"""Transformer-based monocular root localization and rotation model."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .config import TennisMonoLocRotConfig


class TennisMonoLocRotModel(nn.Module):
    """Simple per-player MLP that predicts root translation and rotation.

    Inputs
    ------
    - player_kpts_2d: [B, V, M, J, 2]
    - player_mask:    [B, V, M]
    - court_kpts_2d:  [B, V, C, 2]

    Outputs
    -------
    - root_trans: [B, M, 3]
    - root_rot:   [B, M, 2]  (cos, sin representation of yaw)
    """

    def __init__(self, cfg: TennisMonoLocRotConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D = cfg.D_model

        self.player_in_dim = cfg.num_joints * 2
        self.court_in_dim = cfg.num_court_points * 2

        # Per-player pose embedding and per-view court embedding.
        self.player_mlp = nn.Sequential(
            nn.LayerNorm(self.player_in_dim),
            nn.Linear(self.player_in_dim, D),
            nn.GELU(),
            nn.Linear(D, D),
        )

        self.court_mlp = nn.Sequential(
            nn.LayerNorm(self.court_in_dim),
            nn.Linear(self.court_in_dim, D),
            nn.GELU(),
            nn.Linear(D, D),
        )

        # Camera embedding for encoder tokens.
        self.camera_embed = nn.Embedding(cfg.max_cameras, D)

        # Intra-view encoder over [court_token_v, pose_tokens_{v,*}].
        intra_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=cfg.nheads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.intra_encoder = nn.TransformerEncoder(
            intra_layer, num_layers=cfg.intra_layers
        )

        # Inter-view encoder over all tokens across cameras.
        inter_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=cfg.nheads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.inter_encoder = nn.TransformerEncoder(
            inter_layer, num_layers=cfg.inter_layers
        )

        def _build_decoder() -> nn.TransformerDecoder:
            layer = nn.TransformerDecoderLayer(
                d_model=D,
                nhead=cfg.nheads,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
            )
            return nn.TransformerDecoder(layer, num_layers=cfg.decoder_layers)

        # Decoder over per-player queries.
        self.decoder = _build_decoder()

        # Learnable per-player query embeddings (slots up to max_players).
        self.query_embed = nn.Embedding(cfg.max_players, D)

        # Output heads for root translation and rotation.
        self.root_trans_head = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 3),
        )
        self.root_rot_head = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 2),
        )

        # Existence head for each player query.
        self.exist_head = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 1),
        )

        # Denoising tokens and heads.
        self.noise_token_mlp = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, D),
            nn.GELU(),
            nn.Linear(D, D),
        )
        self.denoise_decoder = _build_decoder()
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
        *,
        player_kpts_2d: Tensor,
        player_mask: Tensor,
        court_kpts_2d: Tensor,
        denoise_inputs: Mapping[str, Tensor] | None = None,
    ) -> Mapping[str, Tensor]:
        """Forward pass without any temporal dimension.

        Args:
            player_kpts_2d (Tensor): ``[B, V, M, J, 2]`` normalized 2D joints.
            player_mask (Tensor): ``[B, V, M]`` visibility mask per view/player.
            court_kpts_2d (Tensor): ``[B, V, C, 2]`` normalized court keypoints.
            denoise_inputs (Mapping[str, Tensor] | None): Optional payload with
                ``noisy_root`` and ``mask`` tensors used to train the auxiliary
                denoising head.

        Returns:
            Mapping[str, Tensor]: Dictionary containing ``root_trans`` and
            ``root_rot`` along with optional denoising predictions.

        Raises:
            ValueError: If any input tensor has an unexpected shape or if
                ``denoise_inputs`` is missing required keys.

        """
        if player_kpts_2d.ndim != 5:
            msg = "player_kpts_2d must have shape [B, V, M, J, 2]"
            raise ValueError(msg)
        if player_mask.ndim != 3:
            msg = "player_mask must have shape [B, V, M]"
            raise ValueError(msg)
        if court_kpts_2d.ndim != 4:
            msg = "court_kpts_2d must have shape [B, V, C, 2]"
            raise ValueError(msg)

        B, V, M, J, _ = player_kpts_2d.shape
        _, V_court, C, _ = court_kpts_2d.shape
        if V_court != V:
            msg = "Number of cameras in player_kpts_2d and court_kpts_2d must match"
            raise ValueError(msg)
        if self.cfg.num_joints != J:
            msg = "Unexpected num_joints in player_kpts_2d"
            raise ValueError(msg)
        if self.cfg.num_court_points != C:
            msg = "Unexpected num_court_points in court_kpts_2d"
            raise ValueError(msg)
        if self.cfg.max_cameras < V:
            msg = "Number of cameras exceeds cfg.max_cameras"
            raise ValueError(msg)
        if self.cfg.max_players < M:
            msg = "Number of players exceeds cfg.max_players"
            raise ValueError(msg)

        device = player_kpts_2d.device
        D = self.cfg.D_model

        # Ensure boolean player mask.
        if player_mask.dtype != torch.bool:
            player_mask_bool = player_mask.to(dtype=torch.bool)
        else:
            player_mask_bool = player_mask

        # ---- Tokenization: pose tokens (per player) ----
        player_flat = player_kpts_2d.view(B, V, M, J * 2)
        pose_tokens = self.player_mlp(player_flat)  # [B, V, M, D]

        # ---- Tokenization: court tokens (one per view) ----
        court_flat = court_kpts_2d.view(B, V, C * 2)
        court_tokens = self.court_mlp(court_flat).unsqueeze(2)  # [B, V, 1, D]

        # Concatenate per-view tokens: [court, pose_1, ..., pose_M].
        tokens = torch.cat([court_tokens, pose_tokens], dim=2)  # [B, V, 1+M, D]

        # Camera embeddings (shared for court+pose tokens within each view).
        cam_ids = torch.arange(V, device=device)
        cam_embed = self.camera_embed(cam_ids)  # [V, D]
        cam_embed = cam_embed[None, :, None, :].expand(B, V, 1 + M, D)
        tokens = tokens + cam_embed

        # Build padding mask for intra-view encoder: True indicates padding.
        pad_intra = torch.zeros(B, V, 1 + M, dtype=torch.bool, device=device)
        pad_intra[..., 1:] = ~player_mask_bool

        tokens_intra = tokens.view(B * V, 1 + M, D)
        key_padding_intra = pad_intra.view(B * V, 1 + M)

        # ---- Intra-view encoder (within each camera) ----
        mem_intra = self.intra_encoder(
            tokens_intra, src_key_padding_mask=key_padding_intra
        )
        mem_intra = mem_intra.view(B, V, 1 + M, D)

        # ---- Inter-view encoder (across cameras) ----
        mem_inter_in = mem_intra.view(B, V * (1 + M), D)
        key_padding_inter = pad_intra.view(B, V * (1 + M))
        mem_inter = self.inter_encoder(
            mem_inter_in, src_key_padding_mask=key_padding_inter
        )

        # ---- Decoder over per-player queries ----
        player_ids = torch.arange(M, device=device)
        query_tokens = self.query_embed(player_ids)  # [M, D]
        query_tokens = query_tokens.unsqueeze(0).expand(B, M, D)

        dec_out = self.decoder(
            tgt=query_tokens,
            memory=mem_inter,
            memory_key_padding_mask=key_padding_inter,
        )  # [B, M, D]

        root_trans = self.root_trans_head(dec_out)  # [B, M, 3]
        root_rot = self.root_rot_head(dec_out)  # [B, M, 2]
        exist_logit = self.exist_head(dec_out)  # [B, M, 1]

        outputs: dict[str, Tensor] = {
            "root_trans": root_trans,
            "root_rot": root_rot,
            "exist_logit": exist_logit,
        }

        if denoise_inputs is not None:
            noisy_root = denoise_inputs.get("noisy_root")
            noise_mask = denoise_inputs.get("mask")
            if noisy_root is None or noise_mask is None:
                msg = "denoise_inputs must provide 'noisy_root' and 'mask'"
                raise ValueError(msg)
            if noisy_root.shape[-1] != 5:
                msg = "noisy_root must have last dimension of size 5 (3 trans + 2 rot)"
                raise ValueError(msg)
            if noise_mask.dtype != torch.bool:
                noise_mask = noise_mask.to(dtype=torch.bool)
            noise_tokens = self.noise_token_mlp(noisy_root)  # [B, N, D]
            tgt_key_padding_mask = ~noise_mask
            denoise_feat = self.denoise_decoder(
                tgt=noise_tokens,
                memory=mem_inter,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=key_padding_inter,
            )
            denoise_root_trans = self.denoise_root_trans_head(denoise_feat)
            denoise_root_rot = self.denoise_root_rot_head(denoise_feat)
            outputs["denoise_root_trans"] = denoise_root_trans
            outputs["denoise_root_rot"] = denoise_root_rot
            outputs["denoise_mask"] = noise_mask

        return outputs
