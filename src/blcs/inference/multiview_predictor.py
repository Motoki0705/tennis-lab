"""BLCS multi-view inference predictor for tennis analysis."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, ParamSpec, Self, TypeVar, cast

import torch
from torch import Tensor

from src.base.api.predictor import BasePredictor
from src.blcs.models.blcs_multiview_model import BLCSMultiViewModel
from src.blcs.training.multiview_lightning_module import BLCSMultiViewLightningModule
from src.utils.geometry.constants import COURT_COORD_SCALE_XYZ

P = ParamSpec("P")
R = TypeVar("R")


def _no_grad(func: Callable[P, R]) -> Callable[P, R]:
    return cast(Callable[P, R], torch.no_grad()(func))


class BLCSMultiViewPredictor(BasePredictor):
    """BLCS multi-view model inference predictor.

    Predicts 3D ball trajectory from multiple camera views.

    Attributes:
        model: The BLCS multi-view model.
        device: The inference device.

    Example:
        >>> predictor = BLCSMultiViewPredictor.load_from_checkpoint(
        ...     "model.ckpt", device="cuda"
        ... )
        >>> results = predictor.predict(ball_uv, court_kp)
        >>> print(results["position"].shape)  # (B, T, 3)

    """

    def __init__(
        self,
        model: BLCSMultiViewModel,
        device: torch.device,
        norm_scale_xyz: tuple[float, float, float] = COURT_COORD_SCALE_XYZ,
    ) -> None:
        """Initialize the predictor.

        Use load_from_checkpoint to create instances in most cases.

        Args:
            model: Initialized BLCS multi-view model.
            device: Inference device.
            norm_scale_xyz: Normalization scale for xyz coordinates.

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
        """Create a BLCSMultiViewPredictor from a checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file (.ckpt).
            device: Inference device.
            **kwargs: Unused (for compatibility).

        Returns:
            Initialized BLCSMultiViewPredictor instance.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.

        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        device = torch.device(device)
        lightning_module = BLCSMultiViewLightningModule.load_from_checkpoint(
            strict=False,
            checkpoint_path=checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        return cls(model=lightning_module.model, device=device)

    @_no_grad
    def predict(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_mask: Tensor | None = None,
        court_kp_mask: Tensor | None = None,
        view_mask: Tensor | None = None,
        seq_len: Tensor | None = None,
        denormalize: bool = True,
    ) -> dict[str, Tensor]:
        """Predict 3D ball trajectory from multiple views.

        Args:
            ball_uv: Ball 2D trajectory. Shape (B, N, T, 2) or (N, T, 2).
            court_kp: Court 2D keypoints. Shape (B, N, 20, 2) or (N, 20, 2).
            ball_mask: Ball visibility mask. Shape (B, N, T) or (N, T).
            court_kp_mask: Court keypoint visibility. Shape (B, N, 20) or (N, 20).
            view_mask: Valid view mask. Shape (B, N) or (N,).
            seq_len: Sequence lengths. Shape (B,) or scalar.
            denormalize: If True, convert positions to meters.

        Returns:
            Inference results dictionary:
                - position: 3D position (B, T, 3)
                - position_meters: Position in meters (if denormalize=True)

        """
        # Add batch dimension if needed
        if ball_uv.dim() == 3:
            ball_uv = ball_uv.unsqueeze(0)
        if court_kp.dim() == 3:
            court_kp = court_kp.unsqueeze(0)
        if ball_mask is not None and ball_mask.dim() == 2:
            ball_mask = ball_mask.unsqueeze(0)
        if court_kp_mask is not None and court_kp_mask.dim() == 2:
            court_kp_mask = court_kp_mask.unsqueeze(0)
        if view_mask is not None and view_mask.dim() == 1:
            view_mask = view_mask.unsqueeze(0)

        # Move to device
        ball_uv = ball_uv.to(self.device)
        court_kp = court_kp.to(self.device)
        if ball_mask is not None:
            ball_mask = ball_mask.to(self.device)
        if court_kp_mask is not None:
            court_kp_mask = court_kp_mask.to(self.device)
        if view_mask is not None:
            view_mask = view_mask.to(self.device)
        if seq_len is not None:
            seq_len = seq_len.to(self.device)

        # Forward pass
        outputs = self.model(
            ball_uv=ball_uv,
            court_kp=court_kp,
            ball_mask=ball_mask,
            court_kp_mask=court_kp_mask,
            view_mask=view_mask,
        )

        position = outputs["position"].cpu()

        result: dict[str, Tensor] = {
            "position": position,
        }

        if "velocity" in outputs:
            result["velocity"] = outputs["velocity"].cpu()

        if denormalize:
            result["position_meters"] = self._denormalize_position(position)
            if "velocity" in result:
                result["velocity_meters"] = self._denormalize_velocity(result["velocity"])

        return result

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
