"""PLCS multi-view inference predictor for tennis analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor

from src.base.api.predictor import BasePredictor
from src.plcs.models.plcs_multiview_model import PLCSMultiViewModel
from src.plcs.training.multiview_lightning_module import PLCSMultiViewLightningModule
from src.utils.geometry.constants import COURT_COORD_SCALE_XYZ


class PLCSMultiViewPredictor(BasePredictor):
    """PLCS multi-view model inference predictor.

    Predicts 3D position and orientation from multiple camera views.
    Supports both single-frame and sequential inputs.

    Attributes:
        model: The PLCS multi-view model.
        device: The inference device.

    Example:
        >>> predictor = PLCSMultiViewPredictor.load_from_checkpoint(
        ...     "model.ckpt", device="cuda"
        ... )
        >>> # Single frame: (B, N, 17, 2) -> (B, 3)
        >>> results = predictor.predict(human_kp, court_kp)
        >>> # Sequential: (B, N, T, 17, 2) -> (B, T, 3)
        >>> results = predictor.predict(human_kp_seq, court_kp_seq)

    """

    def __init__(
        self,
        model: PLCSMultiViewModel,
        device: torch.device,
    ) -> None:
        """Initialize the predictor.

        Use load_from_checkpoint to create instances in most cases.

        Args:
            model: Initialized PLCS multi-view model.
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
        """Create a PLCSMultiViewPredictor from a checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file (.ckpt).
            device: Inference device.
            **kwargs: Unused (for compatibility).

        Returns:
            Initialized PLCSMultiViewPredictor instance.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.

        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        device = torch.device(device)
        lightning_module = PLCSMultiViewLightningModule.load_from_checkpoint(
            strict=False,
            checkpoint_path=checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        return cls(model=lightning_module.model, device=device)

    @torch.inference_mode()
    def predict(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_kp_mask: Tensor | None = None,
        court_kp_mask: Tensor | None = None,
        view_mask: Tensor | None = None,
        denormalize: bool = True,
    ) -> dict[str, Tensor]:
        """Predict player 3D position and orientation from multiple views.

        Supports both single-frame and sequential inputs:
            - Single frame: (B, N, 17, 2) or (N, 17, 2) -> (B, 3)
            - Sequential: (B, N, T, 17, 2) or (N, T, 17, 2) -> (B, T, 3)

        Args:
            human_kp: Human keypoints.
                - Single frame: (B, N, 17, 2) or (N, 17, 2)
                - Sequential: (B, N, T, 17, 2) or (N, T, 17, 2)
            court_kp: Court keypoints.
                - Single frame: (B, N, 20, 2) or (N, 20, 2)
                - Sequential: (B, N, T, 20, 2) or (N, T, 20, 2)
            human_kp_mask: Human keypoint visibility mask.
            court_kp_mask: Court keypoint visibility mask.
            view_mask: Valid view mask. Shape (B, N) or (N,).
            denormalize: If True, convert positions to meters.

        Returns:
            Inference results dictionary:
                - position: Normalized position (B, 3) or (B, T, 3)
                - position_meters: Position in meters (if denormalize=True)
                - rotation: (sin, cos) (B, 2) or (B, T, 2)
                - yaw_radians: Yaw angle in radians (if denormalize=True)

        """
        # Detect if sequential input: dim == 5 means (B, N, T, K, 2)
        is_sequential = human_kp.dim() == 5 or (
            human_kp.dim() == 4 and human_kp.shape[-1] != 2
        )

        # Add batch dimension if needed
        if human_kp.dim() == 3:
            human_kp = human_kp.unsqueeze(0)
        if court_kp.dim() == 3:
            court_kp = court_kp.unsqueeze(0)
        if human_kp.dim() == 4 and is_sequential:
            human_kp = human_kp.unsqueeze(0)
            court_kp = court_kp.unsqueeze(0)
        if human_kp_mask is not None and human_kp_mask.dim() == 2:
            human_kp_mask = human_kp_mask.unsqueeze(0)
        if court_kp_mask is not None and court_kp_mask.dim() == 2:
            court_kp_mask = court_kp_mask.unsqueeze(0)
        if view_mask is not None and view_mask.dim() == 1:
            view_mask = view_mask.unsqueeze(0)

        # Move to device
        human_kp = human_kp.to(self.device)
        court_kp = court_kp.to(self.device)
        if human_kp_mask is not None:
            human_kp_mask = human_kp_mask.to(self.device)
        if court_kp_mask is not None:
            court_kp_mask = court_kp_mask.to(self.device)
        if view_mask is not None:
            view_mask = view_mask.to(self.device)

        if is_sequential:
            # Sequential input: (B, N, T, K, 2)
            # Process each frame through the model
            B, N, T = human_kp.shape[:3]
            positions = []
            rotations = []

            for t in range(T):
                frame_human = human_kp[:, :, t]  # (B, N, 17, 2)
                frame_court = court_kp[:, :, t]  # (B, N, 20, 2)
                frame_hm = (
                    human_kp_mask[:, :, t] if human_kp_mask is not None else None
                )
                frame_cm = (
                    court_kp_mask[:, :, t] if court_kp_mask is not None else None
                )

                outputs = self.model(
                    human_kp=frame_human,
                    court_kp=frame_court,
                    human_vis=frame_hm,
                    court_vis=frame_cm,
                    num_views=None,
                    camera_params=None,
                )
                positions.append(outputs["position"])
                rotations.append(outputs["rotation"])

            position = torch.stack(positions, dim=1).cpu()  # (B, T, 3)
            rotation = torch.stack(rotations, dim=1).cpu()  # (B, T, 2)
        else:
            # Single frame input: (B, N, K, 2)
            outputs = self.model(
                human_kp=human_kp,
                court_kp=court_kp,
                human_vis=human_kp_mask,
                court_vis=court_kp_mask,
                num_views=None,
                camera_params=None,
            )

            position = outputs["position"].cpu()  # (B, 3)
            rotation = outputs["rotation"].cpu()  # (B, 2)

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
            result["yaw_radians"] = torch.atan2(
                rotation[..., 0], rotation[..., 1]
            )

        return result
