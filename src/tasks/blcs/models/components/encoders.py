"""Encoder modules for BLCS.

These modules encode ball trajectories and court keypoints into latent
representations for 3D trajectory estimation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from src.utils.schema.court import NUM_COURT_KP


class TemporalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences.

    Adds positional information to sequence embeddings using
    sinusoidal functions of different frequencies.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 120,
        dropout: float = 0.1,
    ) -> None:
        """Initialize positional encoding.

        Args:
            d_model: Embedding dimension.
            max_len: Maximum sequence length.
            dropout: Dropout probability.

        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encodings once
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor: Input with positional encoding added.

        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CourtContextEncoder(nn.Module):
    """Encode court keypoints into a global context vector.

    Takes 2D court keypoints and produces a fixed-size context
    representation that captures the camera perspective.
    """

    def __init__(
        self,
        num_court_kp: int = NUM_COURT_KP,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the court encoder.

        Args:
            num_court_kp: Number of court keypoints.
            hidden_dim: Hidden dimension.
            num_heads: Number of attention heads.
            num_layers: Number of transformer layers.
            dropout: Dropout probability.

        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_court_kp = num_court_kp

        # Project 2D coordinates to hidden_dim
        self.kp_proj = nn.Linear(2, hidden_dim)

        # Learnable keypoint embeddings (which court point this is)
        self.kp_embed = nn.Embedding(num_court_kp, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        court_kp: Tensor,
        court_vis: Tensor | None = None,
    ) -> Tensor:
        """Encode court keypoints.

        Args:
            court_kp: Court keypoints, shape (B, 40) or (B, 20, 2).
            court_vis: Visibility mask, shape (B, 20). Optional.

        Returns:
            Tensor: Court context, shape (B, hidden_dim).

        """
        batch_size = court_kp.size(0)

        # Reshape if flattened
        if court_kp.dim() == 2 and court_kp.size(1) == self.num_court_kp * 2:
            court_kp = court_kp.view(batch_size, self.num_court_kp, 2)

        # Project coordinates
        feat = self.kp_proj(court_kp)  # (B, 20, D)

        # Add keypoint embeddings
        kp_idx = torch.arange(self.num_court_kp, device=court_kp.device)
        kp_idx = kp_idx.unsqueeze(0).expand(batch_size, -1)
        feat = feat + self.kp_embed(kp_idx)

        # Create attention mask from visibility
        mask = ~court_vis.bool() if court_vis is not None else None

        # Apply transformer
        encoded = self.transformer(feat, src_key_padding_mask=mask)  # (B, 20, D)

        # Global average pooling
        if mask is not None:
            vis_expanded = (~mask).float().unsqueeze(-1)
            pooled = (encoded * vis_expanded).sum(dim=1) / (
                vis_expanded.sum(dim=1) + 1e-8
            )
        else:
            pooled = encoded.mean(dim=1)

        return self.output_proj(pooled)


class BallTrajectoryEncoder(nn.Module):
    """Encode ball 2D trajectory into sequence features.

    Takes a sequence of 2D ball positions and produces per-frame
    feature representations.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        max_seq_len: int = 120,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the trajectory encoder.

        Args:
            hidden_dim: Hidden dimension.
            num_heads: Number of attention heads.
            num_layers: Number of transformer layers.
            max_seq_len: Maximum sequence length.
            dropout: Dropout probability.

        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Project 2D coordinates to hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Positional encoding
        self.pos_encoding = TemporalPositionalEncoding(
            hidden_dim,
            max_len=max_seq_len,
            dropout=dropout,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(
        self,
        ball_uv: Tensor,
        ball_mask: Tensor | None = None,
    ) -> Tensor:
        """Encode ball trajectory.

        Args:
            ball_uv: Ball 2D positions, shape (B, T, 2).
            ball_mask: Visibility mask, shape (B, T). True = visible.

        Returns:
            Tensor: Encoded trajectory, shape (B, T, hidden_dim).

        """
        # Project coordinates
        feat = self.input_proj(ball_uv)  # (B, T, D)

        # Add positional encoding
        feat = self.pos_encoding(feat)

        # Create attention mask (True = masked/ignored)
        attn_mask = ~ball_mask.bool() if ball_mask is not None else None

        # Apply transformer
        encoded = self.transformer(feat, src_key_padding_mask=attn_mask)

        return encoded


class CourtBallCrossAttention(nn.Module):
    """Cross-attention between court context and ball trajectory.

    Enriches ball trajectory features with court context information
    to help resolve depth ambiguity.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """Initialize cross-attention.

        Args:
            hidden_dim: Hidden dimension.
            num_heads: Number of attention heads.
            dropout: Dropout probability.

        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Cross-attention: ball (query) attends to court (key, value)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm and feedforward
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        ball_feat: Tensor,
        court_context: Tensor,
        ball_mask: Tensor | None = None,
    ) -> Tensor:
        """Apply cross-attention.

        Args:
            ball_feat: Ball trajectory features, shape (B, T, D).
            court_context: Court context vector, shape (B, D).
            ball_mask: Ball visibility mask, shape (B, T).

        Returns:
            Tensor: Enhanced ball features, shape (B, T, D).

        """
        # Expand court context for attention (B, D) -> (B, 1, D)
        court_kv = court_context.unsqueeze(1)

        # Cross-attention
        attn_out, _ = self.cross_attn(
            query=ball_feat,
            key=court_kv,
            value=court_kv,
        )

        # Residual connection and layer norm
        x = self.norm1(ball_feat + attn_out)

        # Feedforward with residual
        x = self.norm2(x + self.ffn(x))

        return x


class BallCourtEncoder(nn.Module):
    """Encode ball position and court keypoints into a unified token.

    This encoder processes ball 2D position and court 2D keypoints separately,
    then combines them into a unified representation for each (frame, camera) pair.
    Similar architecture to PLCS KeypointEncoder but adapted for ball tracking.
    """

    def __init__(
        self,
        ball_input_dim: int = 2,
        court_kp_dim: int = 40,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the encoder.

        Args:
            ball_input_dim: Dimension of ball input (2 for UV).
            court_kp_dim: Dimension of court keypoint input (20 * 2 = 40).
            hidden_dim: Hidden dimension for the encoder.
            num_layers: Number of MLP layers.
            dropout: Dropout probability.

        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Ball position encoder
        ball_layers: list[nn.Module] = []
        in_dim = ball_input_dim
        for _ in range(num_layers):
            out_dim = hidden_dim
            ball_layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = out_dim
        self.ball_encoder = nn.Sequential(*ball_layers)

        # Court keypoint encoder
        court_layers: list[nn.Module] = []
        in_dim = court_kp_dim
        for _ in range(num_layers):
            out_dim = hidden_dim
            court_layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = out_dim
        self.court_encoder = nn.Sequential(*court_layers)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
    ) -> Tensor:
        """Encode ball position and court keypoints.

        Args:
            ball_uv: Ball UV position, shape (batch, 2).
            court_kp: Court keypoints, shape (batch, 40).

        Returns:
            Tensor: Fused representation, shape (batch, hidden_dim).

        """
        ball_feat = self.ball_encoder(ball_uv)
        court_feat = self.court_encoder(court_kp)

        # Concatenate and fuse
        combined = torch.cat([ball_feat, court_feat], dim=-1)
        return self.fusion(combined)
