"""Unified predictor class for BLCS inference."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Self, cast

import torch
from torch import Tensor, nn

from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.blcs.training.lightning_module import BLCSLightningModule
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
        model, resolved_device = cls._load_single_lightning_checkpoint(
            checkpoint_path,
            BLCSLightningModule,
            device,
            **kwargs,
        )
        return cls(model=model, device=resolved_device)

    def predict(
        self,
        ball_uv: Tensor,
        court_kp: Tensor | None = None,
        ball_vis: Tensor | None = None,
        ball_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
        denormalize: bool = True,
        court_line_map: Tensor | None = None,
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
        moved = self._to_device(
            self.device,
            ball_uv,
            court_kp,
            ball_vis,
            ball_mask,
            court_vis,
            court_line_map,
        )
        ball_uv = cast(Tensor, moved[0])
        court_kp, ball_vis, ball_mask, court_vis, court_line_map = moved[1:]

        with torch.no_grad():
            court_input_type = str(getattr(self.model, "court_input_type", "kp"))
            if court_input_type == "line":
                if (
                    court_line_map is None
                    or court_kp is not None
                    or court_vis is not None
                ):
                    raise ValueError(
                        "Line-based BLCS inference requires court_line_map and rejects "
                        "court_kp/court_vis."
                    )
                outputs = self.model(
                    ball_uv=ball_uv,
                    court_line_map=court_line_map,
                    ball_vis=ball_vis,
                    ball_mask=ball_mask,
                )
            elif court_input_type == "kp":
                if court_kp is None or court_line_map is not None:
                    raise ValueError(
                        "KP-based BLCS inference requires court_kp and rejects "
                        "court_line_map."
                    )
                outputs = self.model(
                    ball_uv=ball_uv,
                    court_kp=court_kp,
                    ball_vis=ball_vis,
                    ball_mask=ball_mask,
                    court_vis=court_vis,
                )
            else:
                raise ValueError(f"Unsupported court_input_type={court_input_type!r}.")

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
