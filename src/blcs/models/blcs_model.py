"""Main BLCS model implementation.

Ball Localization in Court System: estimates ball 3D trajectory
in tennis court coordinates from 2D ball observations and court keypoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from src.blcs.models.components.encoders import (
    BallTrajectoryEncoder,
    CourtBallCrossAttention,
    CourtContextEncoder,
)
from src.blcs.models.components.heads import Trajectory3DHead, VelocityHead
from src.blcs.utils.constants import MAX_SEQ_LEN, NUM_COURT_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BLCSModel(nn.Module):
    """BLCS: Ball Localization in Court System.

    This model takes 2D ball trajectory and court keypoints from a
    camera view and predicts the ball's 3D trajectory in the court
    coordinate system.

    Input:
        - ball_uv: Ball 2D positions (u, v), shape (B, T, 2)
        - court_kp: Court 2D keypoints (20 landmarks), shape (B, 40) or (B, 20, 2)
        - ball_mask: Ball visibility mask, shape (B, T). Optional.
        - court_vis: Court keypoint visibility, shape (B, 20). Optional.

    Output:
        - position: Normalized (x, y, z) trajectory, shape (B, T, 3)
        - velocity: Normalized velocities (optional), shape (B, T, 3)

    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = MAX_SEQ_LEN,
        use_cross_attention: bool = True,
        predict_velocity: bool = False,
    ) -> None:
        """Initialize the BLCS model.

        Args:
            hidden_dim: Hidden dimension for encoder and heads.
            num_layers: Number of layers in trajectory encoder.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            max_seq_len: Maximum sequence length.
            use_cross_attention: Use cross-attention with court context.
            predict_velocity: Also predict velocities (for auxiliary loss).

        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.use_cross_attention = use_cross_attention
        self.predict_velocity = predict_velocity

        # Court context encoder
        self.court_encoder = CourtContextEncoder(
            num_court_kp=NUM_COURT_KP,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=2,  # Lighter encoder for court
            dropout=dropout,
        )

        # Ball trajectory encoder
        self.ball_encoder = BallTrajectoryEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        # Cross-attention (optional)
        if use_cross_attention:
            self.cross_attention = CourtBallCrossAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
            self.cross_attention = None

            # Alternative: concatenate court context to each frame
            self.context_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        # Output head for 3D positions
        self.position_head = Trajectory3DHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            output_dim=3,
            num_layers=2,
            dropout=dropout,
        )

        # Optional velocity head
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
    def from_config(cls, config: DictConfig) -> BLCSModel:
        """Create model from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            BLCSModel: Initialized model.

        """
        model_cfg = config.get("model", {})
        return cls(
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_layers=model_cfg.get("num_layers", 6),
            num_heads=model_cfg.get("num_heads", 8),
            dropout=model_cfg.get("dropout", 0.1),
            max_seq_len=model_cfg.get("max_seq_len", MAX_SEQ_LEN),
            use_cross_attention=model_cfg.get("use_cross_attention", True),
            predict_velocity=model_cfg.get("predict_velocity", False),
        )

    def forward(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            ball_uv: Ball 2D positions, shape (B, T, 2).
            court_kp: Court keypoints, shape (B, 40) or (B, 20, 2).
            ball_mask: Ball visibility mask, shape (B, T). Optional.
            court_vis: Court visibility mask, shape (B, 20). Optional.

        Returns:
            dict: Dictionary with 'position' (B, T, 3) and optionally 'velocity'.

        """
        batch_size, seq_len, _ = ball_uv.shape

        # Encode court context
        court_context = self.court_encoder(court_kp, court_vis)  # (B, D)

        # Encode ball trajectory
        ball_feat = self.ball_encoder(ball_uv, ball_mask)  # (B, T, D)

        # Fuse with court context
        if self.use_cross_attention and self.cross_attention is not None:
            fused_feat = self.cross_attention(ball_feat, court_context, ball_mask)
        else:
            # Expand court context and concatenate
            court_expanded = court_context.unsqueeze(1).expand(-1, seq_len, -1)
            combined = torch.cat([ball_feat, court_expanded], dim=-1)
            fused_feat = self.context_fusion(combined)

        # Predict 3D positions
        position = self.position_head(fused_feat)  # (B, T, 3)

        outputs = {"position": position}

        # Optionally predict velocities
        if self.predict_velocity and self.velocity_head is not None:
            velocity = self.velocity_head(fused_feat)  # (B, T, 3)
            outputs["velocity"] = velocity

        return outputs

    def predict(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Inference mode prediction.

        Same as forward but ensures eval mode and no gradients.

        Args:
            ball_uv: Ball 2D positions.
            court_kp: Court keypoints.
            ball_mask: Ball visibility mask.
            court_vis: Court visibility mask.

        Returns:
            dict: Predictions with position (and optionally velocity).

        """
        self.eval()
        with torch.no_grad():
            return self.forward(ball_uv, court_kp, ball_mask, court_vis)

    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
