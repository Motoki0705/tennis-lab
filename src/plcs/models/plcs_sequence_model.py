"""Sequential PLCS model implementation.

Decoder-Only Transformer architecture with GQA + RoPE + SDPA.
Estimates player 3D position and rotation in tennis court coordinates
from 2D player keypoints and court keypoints.

Architecture:
    - Court keypoints (20) are tokenized as prefix tokens (fixed per scene)
    - Player keypoints per frame (T) are tokenized as sequence tokens
    - Decoder-only Transformer with RoPE positional encoding
    - Grouped-Query Attention (GQA) using F.scaled_dot_product_attention (SDPA)
    - SwiGLU MLP and RMSNorm for efficiency
    - PositionHead and RotationHead outputs from player tokens only
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.common.models import RMSNorm, RoPE, RoPEConfig, TransformerBlock
from src.plcs.models.components.heads import PositionHead, RotationHead
from src.utils.geometry import NUM_COURT_KP, NUM_HUMAN_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig


# -------------------------
# Token embeddings
# -------------------------
class CourtTokenEmbedding(nn.Module):
    """Embed court keypoints as tokens.

    Each of the 20 court keypoints becomes a separate token.

    Args:
        court_kp: Court keypoints, shape (B, 40) or (B, 20, 2).
        court_vis: Court visibility, shape (B, 20). Optional.

    Returns:
        Tokens of shape (B, NUM_COURT_KP, D).

    """

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        in_dim = 2 + 1  # (u, v) + visibility
        self.proj = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, court_kp: Tensor, court_vis: Optional[Tensor]) -> Tensor:
        B = court_kp.shape[0]
        if court_kp.dim() == 2:
            court_kp = court_kp.view(B, NUM_COURT_KP, 2)
        vis = (
            torch.ones(B, NUM_COURT_KP, device=court_kp.device, dtype=court_kp.dtype)
            if court_vis is None
            else court_vis.to(court_kp.dtype)
        )
        x = torch.cat([court_kp, vis.unsqueeze(-1)], dim=-1)
        return self.proj(x)


class PlayerTokenEmbedding(nn.Module):
    """Embed player keypoints per frame as tokens.

    Each frame's player keypoints are flattened and projected to a single token.

    Args:
        human_kp: Human keypoints, shape (B, T, 34) or (B, T, 17, 2).
        human_vis: Human visibility, shape (B, T, 17). Optional.

    Returns:
        Tokens of shape (B, T, D).

    """

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        # 17 keypoints * (2 coords + 1 visibility) = 51
        in_dim = NUM_HUMAN_KP * 3
        self.proj = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, human_kp: Tensor, human_vis: Optional[Tensor]) -> Tensor:
        B, T = human_kp.shape[:2]

        # Flatten keypoints if needed: (B, T, K, 2) -> (B, T, K*2)
        if human_kp.dim() == 4:
            human_kp = human_kp.view(B, T, -1)  # (B, T, 34)

        # Reshape to (B, T, 17, 2)
        human_kp = human_kp.view(B, T, NUM_HUMAN_KP, 2)

        # Build visibility
        if human_vis is None:
            vis = torch.ones(B, T, NUM_HUMAN_KP, device=human_kp.device, dtype=human_kp.dtype)
        else:
            vis = human_vis.to(human_kp.dtype)

        # Concatenate coords and visibility: (B, T, 17, 3)
        x = torch.cat([human_kp, vis.unsqueeze(-1)], dim=-1)
        # Flatten to (B, T, 51)
        x = x.view(B, T, -1)
        return self.proj(x)


# -------------------------
# Main model
# -------------------------
class PLCSSequenceModel(nn.Module):
    """PLCS sequence model with Decoder-Only Transformer architecture.

    Llama-style architecture with:
    - Grouped-Query Attention (GQA) with SDPA for efficiency
    - Rotary Position Embedding (RoPE)
    - SwiGLU MLP and RMSNorm

    Tokens = [court_tokens(NUM_COURT_KP), player_tokens(T)]
    Predicts 3D position and rotation from player tokens only.

    Input:
        - human_kp: Human 2D keypoints, shape (B, T, 34) or (B, T, 17, 2)
        - court_kp: Court 2D keypoints, shape (B, 40) or (B, 20, 2)
            Note: Court keypoints are scene-level (not per-frame)
        - human_vis: Human visibility mask, shape (B, T, 17). Optional.
        - court_vis: Court visibility mask, shape (B, 20). Optional.

    Output:
        - position: Normalized (x, y, z) per frame, shape (B, T, 3)
        - rotation: (sin(yaw), cos(yaw)) per frame, shape (B, T, 2)

    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        rope_dim: Optional[int] = None,
        rope_theta: float = 10000.0,
        causal: bool = False,
    ) -> None:
        """Initialize the PLCS sequence model.

        Args:
            hidden_dim: Hidden dimension for all components.
            num_layers: Number of Transformer blocks.
            num_heads: Number of query attention heads.
            num_kv_heads: Number of key/value heads (for GQA).
            ffn_dim: FFN intermediate dimension. Defaults to 8/3 * hidden_dim.
            dropout: Dropout probability.
            rope_dim: RoPE dimension. Defaults to head_dim.
            rope_theta: RoPE theta parameter.
            causal: Use causal attention mask.

        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.causal = causal

        head_dim = hidden_dim // num_heads
        rope_dim = head_dim if rope_dim is None else rope_dim
        self.rope = RoPE(RoPEConfig(rope_dim=rope_dim, rope_theta=rope_theta))

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
                    dim=hidden_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    rope=self.rope,
                    causal=causal,
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
    def from_config(cls, config: DictConfig) -> PLCSSequenceModel:
        """Create model from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            PLCSSequenceModel: Initialized model.

        """
        model_cfg = config.get("model", {})
        return cls(
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_layers=model_cfg.get("num_layers", 8),
            num_heads=model_cfg.get("num_heads", 8),
            num_kv_heads=model_cfg.get("num_kv_heads", 2),
            ffn_dim=model_cfg.get("ffn_dim", None),
            dropout=model_cfg.get("dropout", 0.1),
            rope_dim=model_cfg.get("rope_dim", None),
            rope_theta=model_cfg.get("rope_theta", 10000.0),
            causal=model_cfg.get("causal", False),
        )

    def _build_positions(self, T: int, device: torch.device) -> Tensor:
        """Build position indices for court + player tokens."""
        S = NUM_COURT_KP + T
        return torch.arange(S, device=device, dtype=torch.long)

    def forward(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Optional[Tensor] = None,
        court_vis: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            human_kp: Human keypoints, shape (B, T, 34) or (B, T, 17, 2).
            court_kp: Court keypoints, shape (B, T, 40), (B, T, 20, 2),
                or (B, 40) / (B, 20, 2). If per-frame, only first frame is used.
            human_vis: Human visibility mask, shape (B, T, 17). Optional.
            court_vis: Court visibility mask, shape (B, T, 20) or (B, 20). Optional.

        Returns:
            dict: Dictionary with 'position' (B, T, 3) and 'rotation' (B, T, 2).

        """
        B = human_kp.size(0)
        T = human_kp.size(1)

        # Handle court_kp: extract first frame if per-frame format
        if court_kp.dim() == 4:
            # (B, T, 20, 2) -> (B, 20, 2) using first frame
            court_kp = court_kp[:, 0, :, :]
        elif court_kp.dim() == 3 and court_kp.size(1) > 1:
            # (B, T, 40) -> (B, 40) using first frame
            court_kp = court_kp[:, 0, :]
        # Now court_kp is (B, 40) or (B, 20, 2)

        # Handle court_vis similarly
        if court_vis is not None:
            if court_vis.dim() == 3:
                # (B, T, 20) -> (B, 20)
                court_vis = court_vis[:, 0, :]

        # Tokenize court and player
        court_tok = self.court_embed(court_kp, court_vis)  # (B, 20, D)
        player_tok = self.player_embed(human_kp, human_vis)  # (B, T, D)

        # Add type embeddings
        court_type = self.type_embed(
            torch.zeros(NUM_COURT_KP, device=human_kp.device, dtype=torch.long)
        )[None, :, :]  # (1, 20, D)
        player_type = self.type_embed(
            torch.ones(T, device=human_kp.device, dtype=torch.long)
        )[None, :, :]  # (1, T, D)

        x = torch.cat(
            [court_tok + court_type, player_tok + player_type], dim=1
        )  # (B, S, D)

        pos = self._build_positions(T, device=x.device)  # (S,)

        # Build key_padding_mask if human_vis provided
        key_padding_mask: Optional[Tensor] = None
        if human_vis is not None:
            # Court tokens are always valid
            court_mask = torch.ones(B, NUM_COURT_KP, device=x.device, dtype=torch.bool)
            # Player tokens are valid if any keypoint is visible
            player_mask = human_vis.sum(dim=-1) > 0  # (B, T)
            key_padding_mask = torch.cat([court_mask, player_mask], dim=1)  # (B, S)

        for blk in self.blocks:
            x = blk(x, pos=pos, key_padding_mask=key_padding_mask)

        x = self.final_norm(x)
        player_out = x[:, NUM_COURT_KP:, :]  # (B, T, D)

        # Apply output heads
        B_T = B * T
        player_flat = player_out.reshape(B_T, self.hidden_dim)
        position_flat = self.position_head(player_flat)  # (B*T, 3)
        rotation_flat = self.rotation_head(player_flat)  # (B*T, 2)

        position = position_flat.view(B, T, 3)
        rotation = rotation_flat.view(B, T, 2)

        return {
            "position": position,
            "rotation": rotation,
        }

    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

