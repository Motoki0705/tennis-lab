"""Output head modules for PLCS models."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from src.utils.models.components.norm import RMSNorm
from src.utils.models.heads import MLPHead


class PositionHead(MLPHead):
    """Predict 3D position from latent representation."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Predict position from features."""
        return cast("Tensor", self.mlp(x))


class RotationHead(MLPHead):
    """Predict (cos(yaw), sin(yaw)) from latent representation."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=2,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Predict unit-normalized (cos, sin)."""
        out = self.mlp(x)
        return torch.nn.functional.normalize(out, dim=-1)


class CanonicalPoseHead(MLPHead):
    """Predict canonical 3D player joints from latent representation."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_keypoints: int,
    ) -> None:
        n_kp = int(num_keypoints)
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=n_kp * 3,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.num_keypoints = n_kp

    def forward(self, x: Tensor) -> Tensor:
        """Predict canonical joints with shape ``(..., K, 3)``."""
        out = cast("Tensor", self.mlp(x))
        return out.reshape(*x.shape[:-1], self.num_keypoints, 3)


class TemporalDecomposedCanonicalPoseHead(torch.nn.Module):
    """Decode a sequence into a static pose plus zero-mean motion residual.

    The direct canonical head can spend almost all of its capacity on the
    sequence-average body geometry because articulation is a comparatively
    small component of the metre-space target. This head gives that small
    component its own normalized path: the temporal mean feature predicts the
    static pose, while RMS-normalized deviations predict a residual whose mean
    is constrained to zero over valid frames.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_keypoints: int,
    ) -> None:
        super().__init__()
        self.static_head = CanonicalPoseHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_keypoints=num_keypoints,
        )
        self.motion_norm = RMSNorm(input_dim)
        self.motion_head = CanonicalPoseHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_keypoints=num_keypoints,
        )

    def forward(self, features: Tensor, frame_valid: Tensor) -> Tensor:
        """Predict ``(B,T,J,3)`` canonical pose from valid sequence features."""
        if features.ndim != 3:
            raise ValueError(
                "TemporalDecomposedCanonicalPoseHead features must have shape "
                f"(B,T,D), got {tuple(features.shape)}."
            )
        if frame_valid.dtype is not torch.bool:
            raise TypeError("frame_valid must have dtype torch.bool.")
        if frame_valid.shape != features.shape[:2]:
            raise ValueError(
                "frame_valid must match the feature (B,T) axes, got "
                f"{tuple(frame_valid.shape)} for {tuple(features.shape)}."
            )
        if bool((~frame_valid.any(dim=1)).any().item()):
            raise ValueError("Every sequence must contain at least one valid frame.")

        feature_weight = frame_valid.to(dtype=features.dtype).unsqueeze(-1)
        valid_count = feature_weight.sum(dim=1, keepdim=True)
        mean_features = (features * feature_weight).sum(
            dim=1, keepdim=True
        ) / valid_count

        static_pose = self.static_head(mean_features.squeeze(1)).unsqueeze(1)
        centered_features = (features - mean_features) * feature_weight
        motion_pose = self.motion_head(self.motion_norm(centered_features))

        pose_weight = feature_weight.unsqueeze(-1)
        motion_mean = (motion_pose * pose_weight).sum(
            dim=1, keepdim=True
        ) / valid_count.unsqueeze(-1)
        centered_motion = (motion_pose - motion_mean) * pose_weight
        return cast("Tensor", static_pose + centered_motion)
