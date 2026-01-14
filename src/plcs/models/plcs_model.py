"""Main PLCS model implementation.

Player Localization in Court System: estimates player position and
rotation in tennis court coordinates from 2D pose observations.

Architecture:
    - Llama-style Decoder-Only Transformer with GQA + RoPE + SDPA
    - Court keypoints (20) are tokenized as individual tokens
    - Player keypoints (17) are tokenized as individual tokens
    - Both court and player tokens are processed together
    - PositionHead and RotationHead outputs from pooled player tokens
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from src.common.models import (
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    precompute_freqs_cis,
)
from src.plcs.models.components.encoders import (
    CourtTokenEmbedding,
    PlayerTokenEmbedding,
)
from src.plcs.models.components.heads import PositionHead, RotationHead
from src.utils.geometry import NUM_COURT_KP, NUM_HUMAN_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig


class PLCSModel(nn.Module):
    """PLCS: Player Localization in Court System.

    Llama-style architecture with:
    - Grouped-Query Attention (GQA) with SDPA for efficiency
    - Rotary Position Embedding (RoPE)
    - SwiGLU MLP and RMSNorm

    This model takes 2D keypoints (human pose + court landmarks) from a
    camera view and predicts the player's 3D position and rotation in
    the court coordinate system.

    Tokens = [court_tokens(NUM_COURT_KP), player_tokens(NUM_HUMAN_KP)]
    Predicts 3D position and rotation from pooled player tokens.

    Input:
        - human_kp: Human 2D keypoints (COCO 17), shape (B, 34) or (B, 17, 2)
        - court_kp: Court 2D keypoints (20 landmarks), shape (B, 40) or (B, 20, 2)
        - human_vis: Human visibility mask, shape (B, 17). Optional.
        - court_vis: Court visibility mask, shape (B, 20). Optional.

    Output:
        - position: Normalized (x, y, z) in court coordinates, shape (B, 3)
        - rotation: (sin(yaw), cos(yaw)), shape (B, 2)

    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        rope_dim: int | None = None,
        rope_theta: float = 10000.0,
    ) -> None:
        """Initialize the PLCS model.

        Args:
            hidden_dim: Hidden dimension for all components.
            num_layers: Number of Transformer blocks.
            num_heads: Number of query attention heads.
            num_kv_heads: Number of key/value heads (for GQA).
            ffn_dim: FFN intermediate dimension. Defaults to 8/3 * hidden_dim.
            dropout: Dropout probability.
            rope_dim: RoPE dimension. Defaults to head_dim.
            rope_theta: RoPE theta parameter.

        """
        super().__init__()

        self.hidden_dim = hidden_dim

        head_dim = hidden_dim // num_heads
        rope_dim = head_dim if rope_dim is None else rope_dim
        self.rope_dim = int(rope_dim)
        self.rope_theta = float(rope_theta)

        if ffn_dim is None:
            ffn_dim = int((8 * hidden_dim) / 3)
            ffn_dim = (ffn_dim + 63) // 64 * 64  # Round to multiple of 64

        # Token embeddings
        self.court_embed = CourtTokenEmbedding(dim=hidden_dim, dropout=dropout)
        self.player_embed = PlayerTokenEmbedding(dim=hidden_dim, dropout=dropout)

        # Type embedding: 0 = court, 1 = player
        self.type_embed = nn.Embedding(2, hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=hidden_dim,
                        n_heads=num_heads,
                        mlp_inter_dim=ffn_dim,
                        head_dim=head_dim,
                        rope_dim=self.rope_dim,
                        attn_dropout=dropout,
                        use_moe=False,
                    )
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(hidden_dim)

        # Output heads
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

    @classmethod
    def from_config(cls, config: DictConfig) -> PLCSModel:
        """Create model from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            PLCSModel: Initialized model.

        """
        model_cfg = config.get("model", {})
        return cls(
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_layers=model_cfg.get("num_layers", 4),
            num_heads=model_cfg.get("num_heads", 8),
            num_kv_heads=model_cfg.get("num_kv_heads", 2),
            ffn_dim=model_cfg.get("ffn_dim", None),
            dropout=model_cfg.get("dropout", 0.1),
            rope_dim=model_cfg.get("rope_dim", None),
            rope_theta=model_cfg.get("rope_theta", 10000.0),
        )

    def forward(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None = None,
        court_vis: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            human_kp: Human keypoints, shape (B, 34) or (B, 17, 2).
            court_kp: Court keypoints, shape (B, 40) or (B, 20, 2).
            human_vis: Human visibility mask, shape (B, 17). Optional.
            court_vis: Court visibility mask, shape (B, 20). Optional.

        Returns:
            dict: Dictionary with 'position' (B, 3) and 'rotation' (B, 2).

        """
        B = human_kp.size(0)

        # Tokenize court and player keypoints
        court_tok = self.court_embed(court_kp, court_vis)  # (B, 20, D)
        player_tok = self.player_embed(human_kp, human_vis)  # (B, 17, D)

        # Add type embeddings
        court_type = self.type_embed(
            torch.zeros(NUM_COURT_KP, device=human_kp.device, dtype=torch.long)
        )[None, :, :]  # (1, 20, D)
        player_type = self.type_embed(
            torch.ones(NUM_HUMAN_KP, device=human_kp.device, dtype=torch.long)
        )[None, :, :]  # (1, 17, D)

        x = torch.cat(
            [court_tok + court_type, player_tok + player_type], dim=1
        )  # (B, 37, D)

        # Build key_padding_mask if visibility provided
        key_padding_mask: Tensor | None = None
        if human_vis is not None or court_vis is not None:
            if court_vis is not None:
                court_mask = court_vis.bool()  # (B, 20)
            else:
                court_mask = torch.ones(
                    B, NUM_COURT_KP, device=x.device, dtype=torch.bool
                )
            if human_vis is not None:
                player_mask = human_vis.bool()  # (B, 17)
            else:
                player_mask = torch.ones(
                    B, NUM_HUMAN_KP, device=x.device, dtype=torch.bool
                )
            key_padding_mask = torch.cat([court_mask, player_mask], dim=1)  # (B, 37)

        freqs_cis = precompute_freqs_cis(
            dim=self.rope_dim,
            seqlen=x.shape[1],
            base=self.rope_theta,
            device=x.device,
        )

        attn_mask: Tensor | None = None
        if key_padding_mask is not None:
            # bool mask semantics follow SDPA: True=KEEP, False=MASK
            attn_mask = key_padding_mask[:, None, :].expand(B, x.shape[1], x.shape[1])

        residual = None
        for blk in self.blocks:
            x, residual = blk(
                x,
                residual,
                start_pos=0,
                freqs_cis=freqs_cis,
                attn_mask=attn_mask,
                is_causal=False,
            )

        if residual is None:
            x = self.final_norm(x)
        else:
            x, _ = self.final_norm(x, residual)

        # Extract player tokens and pool
        player_out = x[:, NUM_COURT_KP:, :]  # (B, 17, D)

        # Pool player tokens (mean pooling with visibility mask)
        if human_vis is not None:
            vis_mask = human_vis.to(player_out.dtype).unsqueeze(-1)  # (B, 17, 1)
            pooled = (player_out * vis_mask).sum(dim=1) / (
                vis_mask.sum(dim=1) + 1e-8
            )  # (B, D)
        else:
            pooled = player_out.mean(dim=1)  # (B, D)

        # Apply output heads
        position = self.position_head(pooled)  # (B, 3)
        rotation = self.rotation_head(pooled)  # (B, 2)

        return {
            "position": position,
            "rotation": rotation,
        }

    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
