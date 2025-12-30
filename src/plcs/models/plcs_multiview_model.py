"""Multi-view PLCS model implementation.

This module provides a multi-view player localization model using
separate sequential and camera attention mechanisms.

Architecture:
    1. KeypointEncoder encodes each (frame, camera) pair into tokens
    2. Alternating Sequential Attention and Camera Attention blocks
    3. Aggregate camera tokens and project to head input dimension
    4. PositionHead and RotationHead produce final outputs
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


class SequentialAttentionBlock(nn.Module):
    """Self-attention over the temporal/sequential dimension.

    For each camera, applies attention across time steps.
    Input shape: (B, T, N, D) - processes attention over T for each camera N.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """Initialize sequential attention block.

        Args:
            hidden_dim: Hidden dimension.
            num_heads: Number of attention heads.
            dropout: Dropout probability.

        """
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Apply sequential attention.

        Args:
            x: Input tensor, shape (B, T, N, D).
            mask: Optional mask, shape (B, T, N), True = valid.

        Returns:
            Tensor: Output tensor, shape (B, T, N, D).

        """
        batch_size, seq_len, n_cameras, hidden_dim = x.shape

        # Reshape to (B*N, T, D) for attention over sequence
        x_reshaped = x.permute(0, 2, 1, 3).reshape(
            batch_size * n_cameras, seq_len, hidden_dim
        )

        # Build attention mask if provided
        key_padding_mask = None
        fully_masked: Tensor | None = None
        if mask is not None:
            # mask: (B, T, N) -> (B*N, T)
            key_padding_mask = ~mask.permute(0, 2, 1).reshape(
                batch_size * n_cameras, seq_len
            )
            fully_masked = key_padding_mask.all(dim=1)
            if fully_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[fully_masked] = False
                x_reshaped = x_reshaped.clone()
                x_reshaped[fully_masked] = 0.0

        # Self-attention
        x_norm = self.norm1(x_reshaped)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask
        )
        x_reshaped = x_reshaped + attn_out

        # FFN
        x_reshaped = x_reshaped + self.ffn(self.norm2(x_reshaped))

        # Reshape back to (B, T, N, D)
        out = x_reshaped.reshape(batch_size, n_cameras, seq_len, hidden_dim).permute(
            0, 2, 1, 3
        )
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(dtype=out.dtype)
        return out


class CameraAttentionBlock(nn.Module):
    """Self-attention over the camera dimension.

    For each time step, applies attention across cameras.
    Input shape: (B, T, N, D) - processes attention over N for each time T.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """Initialize camera attention block.

        Args:
            hidden_dim: Hidden dimension.
            num_heads: Number of attention heads.
            dropout: Dropout probability.

        """
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Apply camera attention.

        Args:
            x: Input tensor, shape (B, T, N, D).
            mask: Optional mask, shape (B, T, N), True = valid.

        Returns:
            Tensor: Output tensor, shape (B, T, N, D).

        """
        batch_size, seq_len, n_cameras, hidden_dim = x.shape

        # Reshape to (B*T, N, D) for attention over cameras
        x_reshaped = x.reshape(batch_size * seq_len, n_cameras, hidden_dim)

        # Build attention mask if provided
        key_padding_mask = None
        fully_masked: Tensor | None = None
        if mask is not None:
            # mask: (B, T, N) -> (B*T, N)
            key_padding_mask = ~mask.reshape(batch_size * seq_len, n_cameras)
            fully_masked = key_padding_mask.all(dim=1)
            if fully_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[fully_masked] = False
                x_reshaped = x_reshaped.clone()
                x_reshaped[fully_masked] = 0.0

        # Self-attention
        x_norm = self.norm1(x_reshaped)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask
        )
        x_reshaped = x_reshaped + attn_out

        # FFN
        x_reshaped = x_reshaped + self.ffn(self.norm2(x_reshaped))

        # Reshape back to (B, T, N, D)
        out = x_reshaped.reshape(batch_size, seq_len, n_cameras, hidden_dim)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(dtype=out.dtype)
        return out


class PLCSMultiViewModel(nn.Module):
    """Multi-view PLCS model with alternating attention architecture.

    This model takes 2D keypoints from multiple camera views and time steps,
    processes them with alternating Sequential and Camera attention blocks,
    and predicts the player's 3D position and rotation.

    Input:
        - human_kp: Human 2D keypoints, shape (B, T, N, 17, 2) or (B, N, 17, 2)
        - court_kp: Court 2D keypoints, shape (B, T, N, 20, 2) or (B, N, 20, 2)
        - human_vis: Human visibility masks, shape (B, T, N, 17) or (B, N, 17)
        - court_vis: Court visibility masks, shape (B, T, N, 20) or (B, N, 20)
        - num_views: Number of valid views per sample, shape (B,)

    Output:
        - position: Normalized (x, y, z), shape (B, T, 3) or (B, 3)
        - rotation: (sin(yaw), cos(yaw)), shape (B, T, 2) or (B, 2)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_views: int = 8,
        max_seq_len: int = 120,
        encoder_layers: int = 2,
    ) -> None:
        """Initialize the multi-view PLCS model.

        Args:
            hidden_dim: Hidden dimension for all components.
            num_layers: Number of alternating attention layer pairs.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            max_views: Maximum number of camera views.
            max_seq_len: Maximum sequence length.
            encoder_layers: Number of MLP layers in KeypointEncoder.

        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_views = max_views
        self.max_seq_len = max_seq_len
        self.num_human_kp = NUM_HUMAN_KP
        self.num_court_kp = NUM_COURT_KP

        # KeypointEncoder for generating per-(frame, camera) tokens
        self.keypoint_encoder = KeypointEncoder(
            human_kp_dim=NUM_HUMAN_KP * 2,
            court_kp_dim=NUM_COURT_KP * 2,
            hidden_dim=hidden_dim,
            num_layers=encoder_layers,
            dropout=dropout,
        )

        # Positional embeddings
        self.time_embed = nn.Embedding(max_seq_len, hidden_dim)
        self.camera_embed = nn.Embedding(max_views, hidden_dim)

        # Alternating Sequential and Camera Attention blocks
        self.seq_attn_blocks = nn.ModuleList(
            [
                SequentialAttentionBlock(hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.cam_attn_blocks = nn.ModuleList(
            [
                CameraAttentionBlock(hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        # Aggregation MLP: concat camera tokens -> project to head input dim
        self.aggregate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * max_views, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
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
    def from_config(cls, config: DictConfig) -> PLCSMultiViewModel:
        """Create model from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            PLCSMultiViewModel: Initialized model.

        """
        model_cfg = config.get("model", {})
        return cls(
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_layers=model_cfg.get("num_layers", 4),
            num_heads=model_cfg.get("num_heads", 8),
            dropout=model_cfg.get("dropout", 0.1),
            max_views=model_cfg.get("max_views", 8),
            max_seq_len=model_cfg.get("max_seq_len", 120),
            encoder_layers=model_cfg.get("encoder_layers", 2),
        )

    def forward(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None = None,
        court_vis: Tensor | None = None,
        num_views: Tensor | None = None,
        camera_params: list[list[dict]] | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            human_kp: Human keypoints, shape (B, T, N, 17, 2) or (B, N, 17, 2).
            court_kp: Court keypoints, shape (B, T, N, 20, 2) or (B, N, 20, 2).
            human_vis: Human visibility mask, shape (B, T, N, 17) or (B, N, 17).
            court_vis: Court visibility mask, shape (B, T, N, 20) or (B, N, 20).
            num_views: Number of valid views per sample, shape (B,). Optional.
            camera_params: Camera parameters per view. Optional (unused).

        Returns:
            dict: Dictionary with 'position' and 'rotation'.
                For temporal input: (B, T, 3) and (B, T, 2).
                For single-frame input: (B, 3) and (B, 2).

        """
        # Handle both temporal (B, T, N, K, 2) and single-frame (B, N, K, 2) inputs
        is_temporal = human_kp.dim() == 5
        if not is_temporal:
            # Add temporal dimension: (B, N, K, 2) -> (B, 1, N, K, 2)
            human_kp = human_kp.unsqueeze(1)
            court_kp = court_kp.unsqueeze(1)
            if human_vis is not None:
                human_vis = human_vis.unsqueeze(1)
            if court_vis is not None:
                court_vis = court_vis.unsqueeze(1)

        batch_size, seq_len, n_cameras = human_kp.shape[:3]
        device = human_kp.device

        # Flatten keypoints: (B, T, N, K, 2) -> (B, T, N, K*2)
        human_flat = human_kp.flatten(start_dim=3)  # (B, T, N, 34)
        court_flat = court_kp.flatten(start_dim=3)  # (B, T, N, 40)

        # Encode each (frame, camera) pair with KeypointEncoder
        # Reshape to (B*T*N, kp_dim) for encoder
        human_enc_input = human_flat.reshape(batch_size * seq_len * n_cameras, -1)
        court_enc_input = court_flat.reshape(batch_size * seq_len * n_cameras, -1)
        tokens_flat = self.keypoint_encoder(
            human_enc_input, court_enc_input
        )  # (B*T*N, D)
        tokens = tokens_flat.reshape(
            batch_size, seq_len, n_cameras, self.hidden_dim
        )  # (B, T, N, D)

        # Add positional embeddings
        time_ids = torch.arange(seq_len, device=device)
        camera_ids = torch.arange(n_cameras, device=device)
        tokens = tokens + self.time_embed(time_ids).view(1, seq_len, 1, -1)
        tokens = tokens + self.camera_embed(camera_ids).view(1, 1, n_cameras, -1)

        # Create view mask if num_views provided
        view_mask: Tensor | None = None
        if num_views is not None:
            # (B, N) -> (B, T, N)
            camera_idx = torch.arange(n_cameras, device=device).view(1, 1, n_cameras)
            view_mask = camera_idx < num_views.view(batch_size, 1, 1)
            view_mask = view_mask.expand(batch_size, seq_len, n_cameras)

        # Combine with visibility mask
        token_mask = view_mask
        if human_vis is not None:
            # Frame/camera valid if any keypoint visible
            frame_camera_valid = human_vis.sum(dim=-1) > 0  # (B, T, N)
            if token_mask is not None:
                token_mask = token_mask & frame_camera_valid
            else:
                token_mask = frame_camera_valid

        # Apply alternating Sequential and Camera Attention
        for seq_attn, cam_attn in zip(
            self.seq_attn_blocks, self.cam_attn_blocks, strict=True
        ):
            tokens = seq_attn(tokens, token_mask)
            tokens = cam_attn(tokens, token_mask)

        # Aggregate camera tokens: (B, T, N, D) -> (B, T, N*D)
        # Pad to max_views if necessary
        if n_cameras < self.max_views:
            pad_size = self.max_views - n_cameras
            padding = torch.zeros(
                batch_size, seq_len, pad_size, self.hidden_dim, device=device
            )
            tokens = torch.cat([tokens, padding], dim=2)

        tokens_cat = tokens.reshape(
            batch_size, seq_len, self.max_views * self.hidden_dim
        )

        # Project to head input dimension
        aggregated = self.aggregate_mlp(tokens_cat)  # (B, T, D)

        # Apply output heads
        aggregated_flat = aggregated.reshape(batch_size * seq_len, self.hidden_dim)
        position_flat = self.position_head(aggregated_flat)  # (B*T, 3)
        rotation_flat = self.rotation_head(aggregated_flat)  # (B*T, 2)

        position = position_flat.view(batch_size, seq_len, 3)
        rotation = rotation_flat.view(batch_size, seq_len, 2)

        # Remove temporal dimension if input was single-frame
        if not is_temporal:
            position = position.squeeze(1)  # (B, 3)
            rotation = rotation.squeeze(1)  # (B, 2)

        return {
            "position": position,
            "rotation": rotation,
        }
