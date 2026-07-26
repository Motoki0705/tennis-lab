"""Inference wrapper for lifecycle-aware multi-person track queries."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Self, cast

import torch
from torch import Tensor, nn

from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.plcs.training.tracking_lightning_module import (
    PLCSTrackingLightningModule,
)
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


class PLCSTrackingPredictor(BasePredictor):
    """Predict fixed lifecycle queries from ID-ordered per-camera observations."""

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.device = device

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path | Iterable[str | Path],
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> Self:
        model, resolved_device = cls._load_single_lightning_checkpoint(
            checkpoint_path,
            PLCSTrackingLightningModule,
            device,
            **kwargs,
        )
        return cls(model=model, device=resolved_device)

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def predict(
        self,
        *,
        human_kp: Tensor,
        detection_mask: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        frame_mask: Tensor,
        view_mask: Tensor,
        presence_threshold: float = 0.5,
        denormalize: bool = True,
    ) -> dict[str, Tensor]:
        """Return query position/rotation and lifecycle presence outputs."""
        if not 0.0 < presence_threshold < 1.0:
            raise ValueError("presence_threshold must be in (0, 1).")
        inputs = {
            "human_kp": human_kp,
            "detection_mask": detection_mask,
            "court_kp": court_kp,
            "court_vis": court_vis,
            "frame_mask": frame_mask,
            "view_mask": view_mask,
        }
        outputs = cast(
            dict[str, Tensor],
            self.model(**{key: value.to(self.device) for key, value in inputs.items()}),
        )
        position = outputs["position"]
        rotation = outputs["rotation"]
        result = {
            "position": position,
            "rotation": rotation,
            "presence_logits": outputs["presence_logits"],
        }
        probability = result["presence_logits"].sigmoid()
        result["presence_probability"] = probability
        result["presence"] = probability >= presence_threshold
        if denormalize:
            result["position_meters"] = self._denormalize_coords(
                position, COURT_COORD_SCALE_XYZ
            )
            result["yaw_radians"] = torch.atan2(rotation[..., 1], rotation[..., 0])
        return {key: value.detach().cpu() for key, value in result.items()}


__all__ = ["PLCSTrackingPredictor"]
