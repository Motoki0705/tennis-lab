"""Main PLCS model implementation.

Player Localization in Court System: estimates player position and
rotation in tennis court coordinates from 2D pose observations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
from torch import Tensor

from src.plcs.models.components.encoders import (
    KeypointEncoder,
    TransformerKeypointEncoder,
)
from src.plcs.models.components.heads import CombinedHead, PositionHead, RotationHead
from src.plcs.utils.constants import NUM_COURT_KP, NUM_HUMAN_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig


class PLCSModel(nn.Module):
    """PLCS: Player Localization in Court System.

    This model takes 2D keypoints (human pose + court landmarks) from a
    camera view and predicts the player's 3D position and rotation in
    the court coordinate system.

    Input:
        - human_kp: Human 2D keypoints (COCO 17), shape (B, 34) or (B, 17, 2)
        - court_kp: Court 2D keypoints (20 landmarks), shape (B, 40) or (B, 20, 2)

    Output:
        - position: Normalized (x, y, z) in court coordinates, shape (B, 3)
        - rotation: (sin(yaw), cos(yaw)), shape (B, 2)

    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_transformer: bool = True,
        use_combined_head: bool = False,
    ) -> None:
        """Initialize the PLCS model.

        Args:
            hidden_dim: Hidden dimension for encoder and heads.
            num_layers: Number of layers in encoder.
            num_heads: Number of attention heads (for transformer encoder).
            dropout: Dropout probability.
            use_transformer: Use transformer encoder instead of MLP.
            use_combined_head: Use combined head instead of separate heads.

        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_transformer = use_transformer
        self.use_combined_head = use_combined_head

        # Encoder
        if use_transformer:
            self.encoder = TransformerKeypointEncoder(
                num_human_kp=NUM_HUMAN_KP,
                num_court_kp=NUM_COURT_KP,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:
            self.encoder = KeypointEncoder(
                human_kp_dim=NUM_HUMAN_KP * 2,
                court_kp_dim=NUM_COURT_KP * 2,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )

        # Output heads
        if use_combined_head:
            self.combined_head = CombinedHead(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_layers=2,
                dropout=dropout,
            )
            self.position_head = None
            self.rotation_head = None
        else:
            self.combined_head = None
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
            dropout=model_cfg.get("dropout", 0.1),
            use_transformer=model_cfg.get("use_transformer", True),
            use_combined_head=model_cfg.get("use_combined_head", False),
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
        # Encode keypoints
        if self.use_transformer:
            features = self.encoder(human_kp, court_kp, human_vis, court_vis)
        else:
            # Flatten if needed
            if human_kp.dim() == 3:
                human_kp = human_kp.flatten(1)
            if court_kp.dim() == 3:
                court_kp = court_kp.flatten(1)
            features = self.encoder(human_kp, court_kp)

        # Decode outputs
        if self.use_combined_head:
            position, rotation = self.combined_head(features)
        else:
            position = self.position_head(features)
            rotation = self.rotation_head(features)

        return {
            "position": position,
            "rotation": rotation,
        }
