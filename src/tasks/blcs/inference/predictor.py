"""Unified predictor class for BLCS inference."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor, nn

from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.blcs.training.lightning_module import BLCSLightningModule
from src.utils.configuration import PathResolver
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


class BLCSPredictor(BasePredictor):
    """Unified BLCS model inference predictor.

    Supports:
    - `blcs` (single-view)
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
        *,
        resolver: PathResolver,
        device: str | torch.device,
        allow_device_fallback: bool,
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
        model, resolved_device = cls._load_single_lightning_checkpoint(
            checkpoint_path,
            BLCSLightningModule,
            resolver=resolver,
            device=device,
            allow_device_fallback=allow_device_fallback,
            **kwargs,
        )
        return cls(model=model, device=resolved_device)

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
        ball_vis = None if ball_vis is None else ball_vis.to(self.device)
        ball_mask = None if ball_mask is None else ball_mask.to(self.device)
        court_vis = None if court_vis is None else court_vis.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                ball_uv=ball_uv,
                court_kp=court_kp,
                ball_vis=ball_vis,
                ball_mask=ball_mask,
                court_vis=court_vis,
            )

        if denormalize:
            outputs["position"] = self._denormalize_coords(
                outputs["position"], self.norm_scale_xyz
            )
            if "velocity" in outputs:
                outputs["velocity"] = self._denormalize_coords(
                    outputs["velocity"], self.norm_scale_xyz
                )

        outputs = {
            k: v.cpu() if isinstance(v, Tensor) else v for k, v in outputs.items()
        }
        return outputs
