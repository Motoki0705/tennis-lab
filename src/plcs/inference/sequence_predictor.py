"""PLCS sequence inference predictor for tennis analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor

from src.base.api.predictor import BasePredictor
from src.plcs.models.plcs_sequence_model import PLCSSequenceModel
from src.plcs.training.sequence_lightning_module import PLCSSequenceLightningModule
from src.utils.geometry.constants import PLCS_NORM_SCALE_XYZ


class PLCSSequencePredictor(BasePredictor):
    """PLCS sequence model inference predictor.

    Predicts 3D position and orientation sequences from temporal keypoints.

    Attributes:
        model: The PLCS sequence model.
        device: The inference device.

    Example:
        >>> predictor = PLCSSequencePredictor.load_from_checkpoint("model.ckpt")
        >>> results = predictor.predict(human_kp, court_kp)
        >>> print(results["position"].shape)  # (B, T, 3)

    """

    def __init__(
        self,
        model: PLCSSequenceModel,
        device: torch.device,
    ) -> None:
        """Initialize the predictor.

        Use load_from_checkpoint to create instances in most cases.

        Args:
            model: Initialized PLCS sequence model.
            device: Inference device.

        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

        self._norm_scale_xyz = PLCS_NORM_SCALE_XYZ

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> Self:
        """Create a PLCSSequencePredictor from a checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file (.ckpt).
            device: Inference device.
            **kwargs: Unused (for compatibility).

        Returns:
            Initialized PLCSSequencePredictor instance.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.

        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        device = torch.device(device)
        lightning_module = PLCSSequenceLightningModule.load_from_checkpoint(
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
        """Predict player 3D position and orientation sequences.

        Args:
            human_kp: Human keypoints. Shape (B, T, 34) or (B, T, 17, 2).
            court_kp: Court keypoints. Shape (B, T, 40) or (B, T, 20, 2).
            human_vis: Human keypoint visibility mask. Shape (B, T, 17).
            court_vis: Court keypoint visibility mask. Shape (B, T, 20).
            denormalize: If True, convert positions to meters.

        Returns:
            Inference results dictionary:
                - position: Normalized position (B, T, 3)
                - position_meters: Position in meters (B, T, 3) (if denormalize=True)
                - rotation: (sin, cos) (B, T, 2)
                - yaw_radians: Yaw angle in radians (B, T) (if denormalize=True)

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

        position = outputs["position"].cpu()  # (B, T, 3)
        rotation = outputs["rotation"].cpu()  # (B, T, 2)

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
            result["yaw_radians"] = torch.atan2(rotation[..., 0], rotation[..., 1])

        return result
