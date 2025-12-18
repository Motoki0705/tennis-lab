"""Predictor class for BLCS inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor

from src.base.api.predictor import BasePredictor
from src.blcs.models.blcs_model import BLCSModel
from src.blcs.training.lightning_module import BLCSLightningModule
from src.utils.geometry.constants import BLCS_NORM_SCALE_XYZ


class BLCSPredictor(BasePredictor):
    """BLCS model inference predictor.

    Predicts 3D ball trajectory from 2D ball trajectory and court keypoints.

    Attributes:
        model: The BLCS model.
        device: The inference device.

    Example:
        >>> predictor = BLCSPredictor.load_from_checkpoint("model.ckpt", device="cuda")
        >>> results = predictor.predict(ball_uv, court_kp)
        >>> print(results["position"].shape)  # (B, T, 3)

    """

    def __init__(
        self,
        model: BLCSModel,
        device: torch.device,
        norm_scale_xyz: tuple[float, float, float] = BLCS_NORM_SCALE_XYZ,
    ) -> None:
        """Initialize the predictor.

        Use load_from_checkpoint to create instances in most cases.

        Args:
            model: Initialized BLCS model.
            device: Inference device.

        """
        self.model = model.to(device)
        self.device = device
        self.norm_scale_xyz = norm_scale_xyz
        self.model.eval()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> Self:
        """Create a BLCSPredictor from a checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file (.ckpt).
            device: Inference device.
            **kwargs: Unused (for compatibility).

        Returns:
            Initialized BLCSPredictor instance.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.

        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        device = torch.device(device)
        lightning_module = BLCSLightningModule.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
        )

        return cls(model=lightning_module.model, device=device)

    @torch.no_grad()
    def predict(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
        denormalize: bool = True,
    ) -> dict[str, Tensor]:
        """Predict 3D ball trajectory.

        Args:
            ball_uv: Ball 2D trajectory. Shape (B, T, 2) or (T, 2).
            court_kp: Court 2D keypoints. Shape (B, 20, 2) or (20, 2).
            ball_mask: Ball visibility mask. Shape (B, T) or (T,).
            court_vis: Court keypoint visibility. Shape (B, 20) or (20,).
            denormalize: If True, convert positions to meters.

        Returns:
            Inference results dictionary:
                - position: 3D position (B, T, 3)
                - velocity: Velocity (B, T, 3) (if model outputs it)

        """
        # Add batch dimension if needed
        if ball_uv.dim() == 2:
            ball_uv = ball_uv.unsqueeze(0)
        if court_kp.dim() == 2:
            court_kp = court_kp.unsqueeze(0)
        if ball_mask is not None and ball_mask.dim() == 1:
            ball_mask = ball_mask.unsqueeze(0)
        if court_vis is not None and court_vis.dim() == 1:
            court_vis = court_vis.unsqueeze(0)

        # Move to device
        ball_uv = ball_uv.to(self.device)
        court_kp = court_kp.to(self.device)
        if ball_mask is not None:
            ball_mask = ball_mask.to(self.device)
        if court_vis is not None:
            court_vis = court_vis.to(self.device)

        # Forward pass
        outputs = self.model.predict(ball_uv, court_kp, ball_mask, court_vis)

        # Denormalize if requested
        if denormalize:
            outputs["position"] = self._denormalize_position(outputs["position"])
            if "velocity" in outputs:
                outputs["velocity"] = self._denormalize_velocity(outputs["velocity"])

        return outputs

    def _denormalize_position(self, position: Tensor) -> Tensor:
        """Convert normalized position to meters."""
        scale = torch.tensor(
            list(self.norm_scale_xyz),
            device=position.device,
            dtype=position.dtype,
        )
        return position * scale

    def _denormalize_velocity(self, velocity: Tensor) -> Tensor:
        """Convert normalized velocity to m/s."""
        scale = torch.tensor(
            list(self.norm_scale_xyz),
            device=velocity.device,
            dtype=velocity.dtype,
        )
        return velocity * scale
