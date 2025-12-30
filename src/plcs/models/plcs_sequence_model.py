"""Sequential PLCS model implementation.

Token-based architecture with court anchor and player temporal tokens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from src.plcs.models.components.encoders import InputProjection
from src.plcs.models.components.heads import PositionHead, RotationHead
from src.utils.geometry import NUM_COURT_KP, NUM_HUMAN_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig


class TemporalTransformerEncoder(nn.Module):
    """Transformer encoder for temporal token sequences.

    Processes tokens with self-attention over the temporal dimension.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
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

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Encode token sequence.

        Args:
            x: Input tokens, shape (B, S, D) where S = 1 + T.
            mask: Optional boolean mask, shape (B, S), True = valid.

        Returns:
            Tensor: Encoded tokens, shape (B, S, D).

        """
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = ~mask.bool()
        return self.transformer(x, src_key_padding_mask=src_key_padding_mask)


class PLCSSequenceModel(nn.Module):
    """PLCS sequence model with token-based architecture.

    Architecture:
        1. Project player keypoints to [B, T, D] and court keypoints to [B, 1, D]
        2. Concatenate: tokens[:, 0, :] = court, tokens[:, 1:, :] = player
        3. Add type embeddings (court=0, player=1) and time embeddings
        4. Process through Temporal Transformer Encoder
        5. Output heads on player tokens (tokens[:, 1:, :])

    Input:
        - human_kp: Human 2D keypoints, shape (B, T, 34) or (B, T, 17, 2)
        - court_kp: Court 2D keypoints, shape (B, 1, 40) or (B, 1, 20, 2) (pre-aggregated)

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
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Input projections
        self.player_proj = InputProjection(
            input_dim=NUM_HUMAN_KP * 2,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.court_proj = InputProjection(
            input_dim=NUM_COURT_KP * 2,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Type embeddings: 0 = court, 1 = player
        self.type_embed = nn.Embedding(2, hidden_dim)

        # Time embeddings: 0 = court (anchor), 1..T = player frames
        self.time_embed = nn.Embedding(max_seq_len + 1, hidden_dim)

        # Temporal transformer encoder
        self.temporal_encoder = TemporalTransformerEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
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
                or legacy (B, 1, 40) / (B, 1, 20, 2) (pre-aggregated).
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

        # Project player keypoints: (B, T, 34) -> (B, T, D)
        player_tokens = self.player_proj(human_kp)

        # Project court keypoints and aggregate to single token
        # Use mean over time to create anchor token
        court_proj = self.court_proj(court_kp)  # (B, T, D)
        court_token = court_proj.mean(dim=1, keepdim=True)  # (B, 1, D)

        # Assemble tokens: [court, player_1, player_2, ..., player_T]
        tokens = torch.cat([court_token, player_tokens], dim=1)  # (B, 1+T, D)

        # Add type embeddings
        type_ids = torch.zeros(batch_size, 1 + seq_len, dtype=torch.long, device=device)
        type_ids[:, 1:] = 1  # court=0, player=1
        tokens = tokens + self.type_embed(type_ids)

        # Add time embeddings
        time_ids = torch.arange(1 + seq_len, dtype=torch.long, device=device)
        time_ids = time_ids.unsqueeze(0).expand(batch_size, -1)  # (B, 1+T)
        tokens = tokens + self.time_embed(time_ids)

        # Build attention mask if visibility provided
        attn_mask: Tensor | None = None
        if human_vis is not None:
            # Player frames are valid if any keypoint is visible
            player_valid = human_vis.sum(dim=-1) > 0  # (B, T)
            # Court token is always valid
            court_valid = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
            attn_mask = torch.cat([court_valid, player_valid], dim=1)  # (B, 1+T)

        # Temporal transformer encoding
        encoded = self.temporal_encoder(tokens, attn_mask)  # (B, 1+T, D)

        # Extract player tokens for output heads
        player_encoded = encoded[:, 1:, :]  # (B, T, D)

        # Decode outputs
        player_flat = player_encoded.reshape(batch_size * seq_len, self.hidden_dim)
        position_flat = self.position_head(player_flat)  # (B*T, 3)
        rotation_flat = self.rotation_head(player_flat)  # (B*T, 2)

        position = position_flat.view(batch_size, seq_len, 3)
        rotation = rotation_flat.view(batch_size, seq_len, 2)

        return {
            "position": position,
            "rotation": rotation,
        }
