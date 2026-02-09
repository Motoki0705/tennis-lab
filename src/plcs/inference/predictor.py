"""PLCS inference predictor for tennis analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor

from src.base.inference.predictor import BasePredictor
from src.plcs.models.plcs_model import PLCSModel
from src.plcs.training.lightning_module import PLCSLightningModule
from src.utils.geometry.constants import COURT_COORD_SCALE_XYZ


class PLCSPredictor(BasePredictor):
    """PLCS model inference predictor.

    Predicts 3D position and orientation from human and court keypoints.

    Attributes:
        model: The PLCS model.
        device: The inference device.

    Example:
        >>> predictor = PLCSPredictor.load_from_checkpoint("model.ckpt", device="cuda")
        >>> results = predictor.predict(human_kp, court_kp)
        >>> print(results["position"].shape)  # (B, 3)

    """

    def __init__(
        self,
        model: PLCSModel,
        device: torch.device,
    ) -> None:
        """Initialize the predictor.

        Use load_from_checkpoint to create instances in most cases.

        Args:
            model: Initialized PLCS model.
            device: Inference device.

        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

        self._norm_scale_xyz = COURT_COORD_SCALE_XYZ

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> Self:
        """Create a PLCSPredictor from a checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file (.ckpt).
            device: Inference device.
            **kwargs: Unused (for compatibility).

        Returns:
            Initialized PLCSPredictor instance.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.

        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        device = torch.device(device)
        lightning_module = PLCSLightningModule.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
        )

        return cls(model=lightning_module.model, device=device)

    @torch.no_grad()
    def predict(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None = None,
        court_vis: Tensor | None = None,
        denormalize: bool = True,
    ) -> dict[str, Tensor]:
        """Predict player 3D position and orientation.

        Args:
            human_kp: Human keypoints. Shape (B, 34) or (B, 17, 2).
            court_kp: Court keypoints. Shape (B, 40) or (B, 20, 2).
            human_vis: Human keypoint visibility mask. Shape (B, 17).
            court_vis: Court keypoint visibility mask. Shape (B, 20).
            denormalize: If True, convert positions to meters.

        Returns:
            Inference results dictionary (CPU tensors):
                - position: Normalized position (B, 3)
                - rotation: (cos, sin) representation (B, 2)
                - position_meters: Position in meters (B, 3) (if denormalize=True)
                - yaw_radians: Yaw angle in radians (B,) (if denormalize=True)

        """
        # Move to device
        human_kp = human_kp.to(self.device)
        court_kp = court_kp.to(self.device)
        if human_vis is not None:
            human_vis = human_vis.to(self.device)
        if court_vis is not None:
            court_vis = court_vis.to(self.device)

        # Forward pass
        outputs = self.model(human_kp, court_kp, human_vis, court_vis)

        position = outputs["position"].cpu()
        rotation = outputs["rotation"].cpu()

        result: dict[str, Tensor] = {
            "position": position,
            "rotation": rotation,
        }

        if denormalize:
            scale = torch.tensor(
                list(self._norm_scale_xyz),
                dtype=position.dtype,
            )
            result["position_meters"] = position * scale
            result["yaw_radians"] = torch.atan2(rotation[:, 1], rotation[:, 0])

        return result
