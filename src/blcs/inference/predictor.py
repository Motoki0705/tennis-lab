"""Unified predictor class for BLCS inference."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor, nn

from src.base.inference.predictor import BasePredictor
from src.blcs.training.lightning_module import BLCSLightningModule
from src.utils.schema.keypoint_schema import COURT_COORD_SCALE_XYZ


class BLCSPredictor(BasePredictor):
    """Unified BLCS model inference predictor.

    Supports:
    - `blcs` (single-view)
    - `blcs_query` (single-view query-based)
    - `blcs_multiview` (multi-view)

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
        model: nn.Module,
        device: torch.device,
        norm_scale_xyz: tuple[float, float, float] = COURT_COORD_SCALE_XYZ,
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
        checkpoint_path: str | Path | Iterable[str | Path],
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> Self:
        """Create a BLCSPredictor from a checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file (.ckpt).
            device: Inference device.
            **kwargs: Forwarded to `BLCSLightningModule.load_from_checkpoint`.

        Returns:
            Initialized BLCSPredictor instance.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.

        """
        checkpoints = cls._ensure_checkpoint(checkpoint_path)
        if len(checkpoints) != 1:
            raise ValueError(
                "BLCSPredictor expects a single checkpoint, "
                f"got {len(checkpoints)} checkpoints."
            )
        checkpoint = checkpoints[0]
        device = cls._resolve_device(device)
        lightning_module = BLCSLightningModule.load_from_checkpoint(
            checkpoint_path=checkpoint,
            map_location=device,
            strict=bool(kwargs.pop("strict", False)),
            weights_only=bool(kwargs.pop("weights_only", False)),
            **kwargs,
        )
        return cls(model=lightning_module.model, device=device)

    def predict(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_vis: Tensor | None = None,
        ball_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
        denormalize: bool = True,
    ) -> dict[str, Tensor]:
        """Predict 3D ball trajectory.

        Args:
            ball_uv: Ball 2D trajectory tensor accepted by the loaded model.
            court_kp: Court keypoint tensor accepted by the loaded model.
            ball_vis: Optional ball visibility tensor.
            ball_mask: Optional ball validity/padding mask tensor.
            court_vis: Optional court keypoint visibility tensor.
            denormalize: If True, convert positions to meters.

        Returns:
            Inference results dictionary (CPU tensors):
                - position: 3D position (B, T, 3) in meters if denormalize=True,
                           else in normalized coordinates
                - velocity: Velocity (B, T, 3) in m/s if denormalize=True and
                           model outputs it, else in normalized units

        """
        ball_uv = ball_uv.to(self.device)
        court_kp = court_kp.to(self.device)
        ball_vis = ball_vis.to(self.device) if ball_vis is not None else None
        ball_mask = ball_mask.to(self.device) if ball_mask is not None else None
        court_vis = court_vis.to(self.device) if court_vis is not None else None

        with torch.no_grad():
            outputs = self.model(
                ball_uv=ball_uv,
                court_kp=court_kp,
                ball_vis=ball_vis,
                ball_mask=ball_mask,
                court_vis=court_vis,
            )

        if denormalize:
            outputs["position"] = self._denormalize_position(outputs["position"])
            if "velocity" in outputs:
                outputs["velocity"] = self._denormalize_velocity(outputs["velocity"])

        outputs = {k: v.cpu() if isinstance(v, Tensor) else v for k, v in outputs.items()}
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
