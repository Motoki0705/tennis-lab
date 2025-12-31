"""Multi-view BLCS model implementation with alternating attention architecture.

This module provides a multi-view ball trajectory estimation model using
separate sequential and camera attention mechanisms, similar to the PLCS
multiview model architecture.

Architecture:
    1. BallCourtEncoder encodes each (frame, camera) pair into tokens
    2. Alternating Sequential Attention and Camera Attention blocks
    3. Aggregate camera tokens and project to head input dimension
    4. Trajectory3DHead produces final 3D position outputs
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from src.blcs.models.components.encoders import BallCourtEncoder
from src.blcs.models.components.heads import Trajectory3DHead, VelocityHead
from src.utils.geometry import NUM_COURT_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig


class SequentialAttentionBlock(nn.Module):
    """Self-attention over the temporal/sequential dimension.

    For each camera, applies attention across time steps.
    Input shape: (B, N, T, D) - processes attention over T for each camera N.
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
            x: Input tensor, shape (B, N, T, D).
            mask: Optional mask, shape (B, N, T), True = valid.

        Returns:
            Tensor: Output tensor, shape (B, N, T, D).

        """
        batch_size, n_cameras, seq_len, hidden_dim = x.shape

        # Reshape to (B*N, T, D) for attention over sequence
        x_reshaped = x.reshape(
            batch_size * n_cameras, seq_len, hidden_dim
        )

        # Build attention mask if provided
        key_padding_mask = None
        fully_masked: Tensor | None = None
        if mask is not None:
            # mask: (B, N, T) -> (B*N, T)
            key_padding_mask = ~mask.reshape(
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

        # Reshape back to (B, N, T, D)
        out = x_reshaped.reshape(batch_size, n_cameras, seq_len, hidden_dim)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(dtype=out.dtype)
        return out


class CameraAttentionBlock(nn.Module):
    """Self-attention over the camera dimension.

    For each time step, applies attention across cameras.
    Input shape: (B, N, T, D) - processes attention over N for each time T.
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
            x: Input tensor, shape (B, N, T, D).
            mask: Optional mask, shape (B, N, T), True = valid.

        Returns:
            Tensor: Output tensor, shape (B, N, T, D).

        """
        batch_size, n_cameras, seq_len, hidden_dim = x.shape

        # Reshape to (B*T, N, D) for attention over cameras
        x_reshaped = x.permute(0, 2, 1, 3).reshape(batch_size * seq_len, n_cameras, hidden_dim)

        # Build attention mask if provided
        key_padding_mask = None
        fully_masked: Tensor | None = None
        if mask is not None:
            # mask: (B, N, T) -> (B*T, N)
            key_padding_mask = ~mask.permute(0, 2, 1).reshape(batch_size * seq_len, n_cameras)
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

        # Reshape back to (B, N, T, D)
        out = x_reshaped.reshape(batch_size, seq_len, n_cameras, hidden_dim).permute(0, 2, 1, 3)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(dtype=out.dtype)
        return out


class BLCSMultiViewModel(nn.Module):
    """Multi-view BLCS model with alternating attention architecture.

    This model takes ball 2D trajectories and court keypoints from multiple
    camera views and time steps, processes them with alternating Sequential
    and Camera attention blocks, and predicts the ball's 3D trajectory.

    Input:
        - ball_uv: Ball 2D positions, shape (B, T, N, 2)
        - court_kp: Court 2D keypoints, shape (B, T, N, 20, 2)
        - ball_mask: Ball visibility masks, shape (B, T, N)
        - court_vis: Court visibility masks, shape (B, T, N, 20)
        - num_views: Number of valid views per sample, shape (B,)
        - seq_len: Sequence lengths, shape (B,)

    Output:
        - position: Normalized (x, y, z) trajectory, shape (B, T, 3)
        - velocity: Velocities (optional), shape (B, T, 3)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 120,
        max_views: int = 8,
        predict_velocity: bool = False,
        encoder_layers: int = 2,
    ) -> None:
        """Initialize the multi-view BLCS model.

        Args:
            hidden_dim: Hidden dimension for all components.
            num_layers: Number of alternating attention layer pairs.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            max_seq_len: Maximum sequence length.
            max_views: Maximum number of camera views.
            predict_velocity: Also predict velocities.
            encoder_layers: Number of MLP layers in BallCourtEncoder.

        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.max_views = max_views
        self.predict_velocity = predict_velocity
        self.num_court_kp = NUM_COURT_KP

        # BallCourtEncoder for generating per-(frame, camera) tokens
        self.ball_court_encoder = BallCourtEncoder(
            ball_input_dim=2,
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
        self.position_head = Trajectory3DHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            output_dim=3,
            num_layers=2,
            dropout=dropout,
        )

        if predict_velocity:
            self.velocity_head = VelocityHead(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim // 2,
                output_dim=3,
                num_layers=2,
                dropout=dropout,
            )
        else:
            self.velocity_head = None

    @classmethod
    def from_config(cls, config: DictConfig) -> BLCSMultiViewModel:
        """Create model from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            BLCSMultiViewModel: Initialized model.

        """
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})
        return cls(
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_layers=model_cfg.get("num_layers", 4),
            num_heads=model_cfg.get("num_heads", 8),
            dropout=model_cfg.get("dropout", 0.1),
            max_seq_len=model_cfg.get("max_seq_len", data_cfg.get("max_seq_len", 120)),
            max_views=model_cfg.get("max_views", 8),
            predict_velocity=model_cfg.get("predict_velocity", False),
            encoder_layers=model_cfg.get("encoder_layers", 2),
        )

    def forward(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
        num_views: Tensor | None = None,
        seq_len: Tensor | None = None,
        camera_params: list[list[dict]] | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            ball_uv: Ball 2D positions, shape (B, N, T, 2).
            court_kp: Court keypoints, shape (B, N, T, 20, 2).
            ball_mask: Ball visibility mask, shape (B, N, T). Optional.
            court_vis: Court visibility mask, shape (B, N, T, 20). Optional.
            num_views: Number of valid views, shape (B,). Optional.
            seq_len: Sequence lengths, shape (B,). Optional.
            camera_params: Camera parameters per view. Optional (unused).

        Returns:
            dict: Dictionary with 'position' (B, T, 3) and optionally 'velocity'.

        """
        batch_size, n_cameras, seq_length = ball_uv.shape[:3]
        device = ball_uv.device

        # Flatten keypoints and ball for encoder: (B, N, T, ...) -> (B*N*T, ...)
        ball_flat = ball_uv.reshape(batch_size * n_cameras * seq_length, 2)
        court_flat = court_kp.flatten(start_dim=3).reshape(
            batch_size * n_cameras * seq_length, -1
        )

        # Encode each (frame, camera) pair
        tokens_flat = self.ball_court_encoder(ball_flat, court_flat)  # (B*N*T, D)
        tokens = tokens_flat.reshape(
            batch_size, n_cameras, seq_length, self.hidden_dim
        )  # (B, N, T, D)

        # Add positional embeddings
        time_ids = torch.arange(seq_length, device=device)
        camera_ids = torch.arange(n_cameras, device=device)
        tokens = tokens + self.time_embed(time_ids).view(1, 1, seq_length, -1)
        tokens = tokens + self.camera_embed(camera_ids).view(1, n_cameras, 1, -1)

        # Create view mask if num_views provided: (B, N, T)
        view_mask: Tensor | None = None
        if num_views is not None:
            camera_idx = torch.arange(n_cameras, device=device).view(1, n_cameras, 1)
            view_mask = camera_idx < num_views.view(batch_size, 1, 1)
            view_mask = view_mask.expand(batch_size, n_cameras, seq_length)

        # Combine with ball visibility mask
        token_mask = view_mask
        if ball_mask is not None:
            # ball_mask: (B, N, T)
            frame_camera_valid = ball_mask > 0
            if token_mask is not None:
                token_mask = token_mask & frame_camera_valid
            else:
                token_mask = frame_camera_valid

        # Apply sequence length mask
        if seq_len is not None:
            time_idx = torch.arange(seq_length, device=device).view(1, 1, seq_length)
            time_mask = time_idx < seq_len.view(batch_size, 1, 1)
            time_mask = time_mask.expand(batch_size, n_cameras, seq_length)
            token_mask = token_mask & time_mask if token_mask is not None else time_mask

        # Apply alternating Sequential and Camera Attention
        for seq_attn, cam_attn in zip(
            self.seq_attn_blocks, self.cam_attn_blocks, strict=True
        ):
            tokens = seq_attn(tokens, token_mask)
            tokens = cam_attn(tokens, token_mask)

        # Aggregate camera tokens: (B, N, T, D) -> (B, T, N*D)
        # First permute to (B, T, N, D)
        tokens = tokens.permute(0, 2, 1, 3)  # (B, N, T, D) -> (B, T, N, D)
        
        # Pad to max_views if necessary
        if n_cameras < self.max_views:
            pad_size = self.max_views - n_cameras
            padding = torch.zeros(
                batch_size, seq_length, pad_size, self.hidden_dim, device=device
            )
            tokens = torch.cat([tokens, padding], dim=2)

        tokens_cat = tokens.reshape(
            batch_size, seq_length, self.max_views * self.hidden_dim
        )

        # Project to head input dimension
        aggregated = self.aggregate_mlp(tokens_cat)  # (B, T, D)

        # Predict 3D positions
        position = self.position_head(aggregated)  # (B, T, 3)

        outputs = {"position": position}

        # Optionally predict velocities
        if self.predict_velocity and self.velocity_head is not None:
            velocity = self.velocity_head(aggregated)  # (B, T, 3)
            outputs["velocity"] = velocity

        return outputs
