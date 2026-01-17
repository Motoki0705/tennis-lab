"""Encoder modules for PLCS.

These modules encode 2D keypoint inputs into latent representations
suitable for the main PLCS model.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from src.utils.geometry import NUM_COURT_KP, NUM_HUMAN_KP


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer-based models.

    This adds positional information to the input embeddings using
    sinusoidal functions of different frequencies.
    """

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1) -> None:
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


class InputProjection(nn.Module):
    """Project input keypoints to a unified hidden dimension.

    Simple 2-layer MLP with LayerNorm for projecting flattened keypoints.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize input projection.

        Args:
            input_dim: Input dimension (e.g., 34 for human, 40 for court).
            hidden_dim: Output hidden dimension.
            dropout: Dropout probability.

        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Project input to hidden dimension.

        Args:
            x: Input tensor, shape (..., input_dim).

        Returns:
            Tensor: Projected tensor, shape (..., hidden_dim).

        """
        return self.mlp(x)


class KeypointEncoder(nn.Module):
    """Encode 2D keypoints into a latent representation.

    This encoder processes human and court keypoints separately,
    then combines them into a unified representation.
    """

    def __init__(
        self,
        human_kp_dim: int = 34,
        court_kp_dim: int = 40,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the encoder.

        Args:
            human_kp_dim: Dimension of human keypoint input (17 * 2 = 34).
            court_kp_dim: Dimension of court keypoint input (20 * 2 = 40).
            hidden_dim: Hidden dimension for the encoder.
            num_layers: Number of MLP layers.
            dropout: Dropout probability.

        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Human keypoint encoder
        human_layers: list[nn.Module] = []
        in_dim = human_kp_dim
        for i in range(num_layers):
            out_dim = hidden_dim
            human_layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = out_dim
        self.human_encoder = nn.Sequential(*human_layers)

        # Court keypoint encoder
        court_layers: list[nn.Module] = []
        in_dim = court_kp_dim
        for i in range(num_layers):
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
        human_kp: Tensor,
        court_kp: Tensor,
    ) -> Tensor:
        """Encode human and court keypoints.

        Args:
            human_kp: Human keypoints, shape (batch, 34).
            court_kp: Court keypoints, shape (batch, 40).

        Returns:
            Tensor: Fused representation, shape (batch, hidden_dim).

        """
        human_feat = self.human_encoder(human_kp)
        court_feat = self.court_encoder(court_kp)

        # Concatenate and fuse
        combined = torch.cat([human_feat, court_feat], dim=-1)
        return self.fusion(combined)


class TransformerKeypointEncoder(nn.Module):
    """Transformer-based encoder for keypoint sequences.

    This encoder treats keypoints as a sequence and applies
    self-attention to capture relationships between them.
    """

    def __init__(
        self,
        num_human_kp: int = 17,
        num_court_kp: int = 20,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the transformer encoder.

        Args:
            num_human_kp: Number of human keypoints.
            num_court_kp: Number of court keypoints.
            hidden_dim: Hidden dimension.
            num_heads: Number of attention heads.
            num_layers: Number of transformer layers.
            dropout: Dropout probability.

        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_human_kp = num_human_kp
        self.num_court_kp = num_court_kp

        # Project 2D coordinates to hidden_dim
        self.human_proj = nn.Linear(2, hidden_dim)
        self.court_proj = nn.Linear(2, hidden_dim)

        # Learnable type embeddings (human vs court)
        self.type_embed = nn.Embedding(2, hidden_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            hidden_dim,
            max_len=num_human_kp + num_court_kp,
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

        # Output projection (pool to single vector)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None = None,
        court_vis: Tensor | None = None,
    ) -> Tensor:
        """Encode keypoints using transformer.

        Args:
            human_kp: Human keypoints, shape (batch, 34) or (batch, 17, 2).
            court_kp: Court keypoints, shape (batch, 40) or (batch, 20, 2).
            human_vis: Human visibility mask, shape (batch, 17).
            court_vis: Court visibility mask, shape (batch, 20).

        Returns:
            Tensor: Encoded representation, shape (batch, hidden_dim).

        """
        batch_size = human_kp.size(0)

        # Reshape if flattened
        if human_kp.dim() == 2 and human_kp.size(1) == self.num_human_kp * 2:
            human_kp = human_kp.view(batch_size, self.num_human_kp, 2)
        if court_kp.dim() == 2 and court_kp.size(1) == self.num_court_kp * 2:
            court_kp = court_kp.view(batch_size, self.num_court_kp, 2)

        # Project to hidden_dim
        human_feat = self.human_proj(human_kp)  # (B, 17, D)
        court_feat = self.court_proj(court_kp)  # (B, 20, D)

        # Add type embeddings
        human_type = self.type_embed(
            torch.zeros(
                batch_size, self.num_human_kp, dtype=torch.long, device=human_kp.device
            )
        )
        court_type = self.type_embed(
            torch.ones(
                batch_size, self.num_court_kp, dtype=torch.long, device=court_kp.device
            )
        )

        human_feat = human_feat + human_type
        court_feat = court_feat + court_type

        # Concatenate sequences
        seq = torch.cat([human_feat, court_feat], dim=1)  # (B, 37, D)

        # Add positional encoding
        seq = self.pos_encoding(seq)

        # Create attention mask from visibility
        if human_vis is not None and court_vis is not None:
            vis = torch.cat([human_vis, court_vis], dim=1)  # (B, 37)
            # Transformer uses True for masked positions
            mask = ~vis.bool()
        else:
            mask = None

        # Apply transformer
        encoded = self.transformer(seq, src_key_padding_mask=mask)  # (B, 37, D)

        # Global average pooling
        if mask is not None:
            # Masked average
            vis_expanded = (~mask).float().unsqueeze(-1)  # (B, 37, 1)
            pooled = (encoded * vis_expanded).sum(dim=1) / (
                vis_expanded.sum(dim=1) + 1e-8
            )
        else:
            pooled = encoded.mean(dim=1)

        return self.output_proj(pooled)


class CourtTokenEmbedding(nn.Module):
    """Embed court keypoints as individual tokens.

    Each of the 20 court keypoints becomes a separate token.
    This preserves per-keypoint information for the Transformer.

    Input:
        - court_kp: Court 2D keypoints, shape (B, 40) or (B, 20, 2).
        - court_vis: Court visibility mask, shape (B, 20). Optional.

    Output:
        - Tokens of shape (B, NUM_COURT_KP, D).

    """

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        """Initialize court token embedding.

        Args:
            dim: Output embedding dimension.
            dropout: Dropout probability.

        """
        super().__init__()
        # Input: (u, v) + visibility
        in_dim = 2 + 1
        self.proj = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, court_kp: Tensor, court_vis: Tensor | None = None) -> Tensor:
        """Embed court keypoints as tokens.

        Args:
            court_kp: Court keypoints, shape (B, 40) or (B, 20, 2).
            court_vis: Court visibility, shape (B, 20). Optional.

        Returns:
            Tensor: Token embeddings, shape (B, NUM_COURT_KP, D).

        """
        B = court_kp.shape[0]
        if court_kp.dim() == 2:
            court_kp = court_kp.view(B, NUM_COURT_KP, 2)

        if court_vis is None:
            vis = torch.ones(B, NUM_COURT_KP, device=court_kp.device, dtype=court_kp.dtype)
        else:
            vis = court_vis.to(court_kp.dtype)

        # Concatenate coordinates and visibility: (B, 20, 3)
        x = torch.cat([court_kp, vis.unsqueeze(-1)], dim=-1)
        return self.proj(x)


class PlayerTokenEmbedding(nn.Module):
    """Embed player keypoints as individual tokens.

    Each of the 17 player keypoints becomes a separate token.
    This preserves per-keypoint information for the Transformer.

    Input:
        - human_kp: Human 2D keypoints, shape (B, 34) or (B, 17, 2).
        - human_vis: Human visibility mask, shape (B, 17). Optional.

    Output:
        - Tokens of shape (B, NUM_HUMAN_KP, D).

    """

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        """Initialize player token embedding.

        Args:
            dim: Output embedding dimension.
            dropout: Dropout probability.

        """
        super().__init__()
        # Input: (u, v) + visibility
        in_dim = 2 + 1
        self.proj = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, human_kp: Tensor, human_vis: Tensor | None = None) -> Tensor:
        """Embed player keypoints as tokens.

        Args:
            human_kp: Human keypoints, shape (B, 34) or (B, 17, 2).
            human_vis: Human visibility, shape (B, 17). Optional.

        Returns:
            Tensor: Token embeddings, shape (B, NUM_HUMAN_KP, D).

        """
        B = human_kp.shape[0]
        if human_kp.dim() == 2:
            human_kp = human_kp.view(B, NUM_HUMAN_KP, 2)

        if human_vis is None:
            vis = torch.ones(B, NUM_HUMAN_KP, device=human_kp.device, dtype=human_kp.dtype)
        else:
            vis = human_vis.to(human_kp.dtype)

        # Concatenate coordinates and visibility: (B, 17, 3)
        x = torch.cat([human_kp, vis.unsqueeze(-1)], dim=-1)
        return self.proj(x)
