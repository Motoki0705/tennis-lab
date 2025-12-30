"""Sequential PLCS model implementation.

Token-based architecture using KeypointEncoder to generate per-frame tokens,
processed through a simple Transformer, with PositionHead and RotationHead outputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from src.plcs.models.components.encoders import KeypointEncoder
from src.plcs.models.components.heads import PositionHead, RotationHead
from src.utils.geometry import NUM_COURT_KP, NUM_HUMAN_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig


class PLCSSequenceModel(nn.Module):
    """PLCS sequence model with KeypointEncoder + Transformer architecture.

    Architecture:
        1. KeypointEncoder encodes each frame's human+court keypoints into tokens
        2. Add positional embeddings for temporal ordering
        3. Process tokens through a simple Transformer encoder
        4. Apply PositionHead and RotationHead to each output token

    Input:
        - human_kp: Human 2D keypoints, shape (B, T, 34) or (B, T, 17, 2)
        - court_kp: Court 2D keypoints, shape (B, T, 40) or (B, T, 20, 2),
            or legacy (B, 1, 40) / (B, 1, 20, 2) (will be expanded to T)

    Output:
        - position: Normalized (x, y, z) per frame, shape (B, T, 3)
        - rotation: (sin(yaw), cos(yaw)) per frame, shape (B, T, 2)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 120,
        encoder_layers: int = 2,
    ) -> None:
        """Initialize the sequence model.

        Args:
            hidden_dim: Hidden dimension for all components.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            max_seq_len: Maximum sequence length.
            encoder_layers: Number of MLP layers in KeypointEncoder.

        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # KeypointEncoder for generating per-frame tokens
        self.keypoint_encoder = KeypointEncoder(
            human_kp_dim=NUM_HUMAN_KP * 2,
            court_kp_dim=NUM_COURT_KP * 2,
            hidden_dim=hidden_dim,
            num_layers=encoder_layers,
            dropout=dropout,
        )

        # Positional embeddings for temporal ordering
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer encoder for sequence processing
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
            num_layers=model_cfg.get("num_layers", 4),
            num_heads=model_cfg.get("num_heads", 8),
            dropout=model_cfg.get("dropout", 0.1),
            max_seq_len=int(model_cfg.get("max_seq_len", 120)),
            encoder_layers=model_cfg.get("encoder_layers", 2),
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
            human_kp: Human keypoints, shape (B, T, 34) or (B, T, 17, 2).
            court_kp: Court keypoints, shape (B, T, 40), (B, T, 20, 2),
                or legacy (B, 1, 40) / (B, 1, 20, 2) (will be expanded to T).
            human_vis: Human visibility mask, shape (B, T, 17). Optional.
            court_vis: Court visibility mask, shape (B, T, 20). Optional.

        Returns:
            dict: Dictionary with 'position' (B, T, 3) and 'rotation' (B, T, 2).

        """
        batch_size = human_kp.size(0)
        seq_len = human_kp.size(1)
        device = human_kp.device

        # Flatten keypoints if needed: (B, T, K, 2) -> (B, T, K*2)
        if human_kp.dim() == 4:
            human_kp = human_kp.view(batch_size, seq_len, -1)

        # Handle court_kp: can be (B, T, 20, 2), (B, T, 40), or legacy (B, 1, ...)
        if court_kp.dim() == 4:
            court_t = court_kp.size(1)
            court_kp = court_kp.view(batch_size, court_t, -1)  # (B, T or 1, 40)
        else:
            court_t = court_kp.size(1)

        # If court_kp has T=1 (legacy aggregated), expand to T
        if court_t == 1 and seq_len > 1:
            court_kp = court_kp.expand(batch_size, seq_len, -1)  # (B, T, 40)

        # Generate tokens for each frame using KeypointEncoder
        # Reshape to (B*T, kp_dim) for encoder, then reshape back
        human_flat = human_kp.reshape(batch_size * seq_len, -1)  # (B*T, 34)
        court_flat = court_kp.reshape(batch_size * seq_len, -1)  # (B*T, 40)
        tokens_flat = self.keypoint_encoder(human_flat, court_flat)  # (B*T, D)
        tokens = tokens_flat.view(batch_size, seq_len, self.hidden_dim)  # (B, T, D)

        # Add positional embeddings
        positions = torch.arange(seq_len, device=device)
        tokens = tokens + self.pos_embed(positions).unsqueeze(0)  # (B, T, D)

        # Build attention mask if visibility provided
        attn_mask: Tensor | None = None
        if human_vis is not None:
            # Frames are valid if any keypoint is visible
            frame_valid = human_vis.sum(dim=-1) > 0  # (B, T)
            attn_mask = ~frame_valid  # Transformer uses True for masked positions

        # Process through transformer
        encoded = self.transformer(tokens, src_key_padding_mask=attn_mask)  # (B, T, D)

        # Apply output heads to each token
        encoded_flat = encoded.reshape(batch_size * seq_len, self.hidden_dim)
        position_flat = self.position_head(encoded_flat)  # (B*T, 3)
        rotation_flat = self.rotation_head(encoded_flat)  # (B*T, 2)

        position = position_flat.view(batch_size, seq_len, 3)
        rotation = rotation_flat.view(batch_size, seq_len, 2)

        return {
            "position": position,
            "rotation": rotation,
        }
