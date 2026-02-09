"""PLCS multi-view inference predictor for tennis analysis."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Self, TypeVar, cast

import torch
from torch import Tensor

from src.base.inference.predictor import BasePredictor
from src.plcs.models.plcs_multiview_model import PLCSMultiViewModel
from src.plcs.training.multiview_lightning_module import PLCSMultiViewLightningModule
from src.utils.geometry.constants import COURT_COORD_SCALE_XYZ

F = TypeVar("F", bound=Callable[..., Any])


def inference_mode(fn: F) -> F:
    """Apply torch.inference_mode with typing preserved."""
    return cast(F, torch.inference_mode()(fn))


class PLCSMultiViewPredictor(BasePredictor):
    """PLCS multi-view sequential model inference predictor.

    Predicts 3D position and orientation sequences from multiple camera views
    over time. Uses camera-time ordering: (B, N, T, ...) where N=cameras, T=time.

    Attributes:
        model: The PLCS multi-view model.
        device: The inference device.

    Example:
        >>> predictor = PLCSMultiViewPredictor.load_from_checkpoint(
        ...     "model.ckpt", device="cuda"
        ... )
        >>> # Input: (B, N, T, 17, 2) -> Output: (B, T, 3)
        >>> results = predictor.predict(human_kp, court_kp)
        >>> print(results["position"].shape)  # (B, T, 3)
        >>> print(results["rotation"].shape)  # (B, T, 2)

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

    @inference_mode
    def predict(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None = None,
        court_vis: Tensor | None = None,
        view_mask: Tensor | None = None,
        seq_mask: Tensor | None = None,
        denormalize: bool = True,
    ) -> dict[str, Tensor]:
        """Predict player 3D position and orientation from multi-view sequences.

        Uses camera-time ordering: (B, N, T, K, 2) where:
        - B: Batch size
        - N: Number of camera views
        - T: Sequence length (temporal dimension)
        - K: Number of keypoints (17 for human, 20 for court)
        - 2: (u, v) coordinates

        Args:
            human_kp: Human keypoints. Shape (B, N, T, 17, 2) or (N, T, 17, 2).
            court_kp: Court keypoints. Shape (B, N, T, 20, 2) or (N, T, 20, 2).
            human_vis: Human keypoint visibility mask. Shape (B, N, T, 17).
            court_vis: Court keypoint visibility mask. Shape (B, N, T, 20).
            view_mask: Valid view mask. Shape (B, N) where True = valid view.
            seq_mask: Valid sequence mask. Shape (B, T) where True = valid frame.
            denormalize: If True, convert positions to meters.

        Returns:
            Inference results dictionary:
                - position: Normalized position (B, T, 3)
                - position_meters: Position in meters (B, T, 3) (if denormalize=True)
                - rotation: (cos, sin) (B, T, 2)
                - yaw_radians: Yaw angle in radians (B, T) (if denormalize=True)

        """
        # Add batch dimension if needed: (N, T, K, 2) -> (1, N, T, K, 2)
        if human_kp.dim() == 4:
            human_kp = human_kp.unsqueeze(0)
        if court_kp.dim() == 4:
            court_kp = court_kp.unsqueeze(0)
        if human_vis is not None and human_vis.dim() == 3:
            human_vis = human_vis.unsqueeze(0)
        if court_vis is not None and court_vis.dim() == 3:
            court_vis = court_vis.unsqueeze(0)
        if view_mask is not None and view_mask.dim() == 1:
            view_mask = view_mask.unsqueeze(0)
        if seq_mask is not None and seq_mask.dim() == 1:
            seq_mask = seq_mask.unsqueeze(0)

        # Move to device
        human_kp = human_kp.to(self.device)
        court_kp = court_kp.to(self.device)
        if human_vis is not None:
            human_vis = human_vis.to(self.device)
        if court_vis is not None:
            court_vis = court_vis.to(self.device)
        if view_mask is not None:
            view_mask = view_mask.to(self.device)
        if seq_mask is not None:
            seq_mask = seq_mask.to(self.device)

        # Input shape: (B, N, T, K, 2) - camera-time order
        # Pass directly to model (model handles permutation internally)
        outputs = self.model(
            human_kp=human_kp,
            court_kp=court_kp,
            human_vis=human_vis,
            court_vis=court_vis,
            view_mask=view_mask,
            seq_mask=seq_mask,
            camera_params=None,
        )

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
            result["yaw_radians"] = torch.atan2(rotation[..., 1], rotation[..., 0])

        return result
