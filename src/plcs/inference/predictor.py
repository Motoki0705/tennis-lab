"""Unified PLCS inference predictor."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor, nn

from src.base.inference.predictor import BasePredictor
from src.plcs.training.lightning_module import PLCSLightningModule
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


class PLCSPredictor(BasePredictor):
    """Unified PLCS model inference predictor."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        self._norm_scale_xyz = COURT_COORD_SCALE_XYZ

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path | Iterable[str | Path],
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> Self:
        checkpoints = cls._ensure_checkpoint(checkpoint_path)
        if len(checkpoints) != 1:
            raise ValueError(
                "PLCSPredictor expects a single checkpoint, "
                f"got {len(checkpoints)} checkpoints."
            )
        checkpoint = checkpoints[0]
        device = cls._resolve_device(device)
        lightning_module = PLCSLightningModule.load_from_checkpoint(
            checkpoint_path=checkpoint,
            map_location=device,
            strict=bool(kwargs.pop("strict", False)),
            weights_only=bool(kwargs.pop("weights_only", False)),
            **kwargs,
        )
        return cls(model=lightning_module.model, device=device)

    @torch.no_grad()
    def predict(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None = None,
        human_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
        denormalize: bool = True,
    ) -> dict[str, Tensor]:
        """Predict player 3D position and orientation from caller-provided tensors."""

        human_kp = human_kp.to(self.device)
        court_kp = court_kp.to(self.device)
        if human_vis is not None:
            human_vis = human_vis.to(self.device)
        if human_mask is not None:
            human_mask = human_mask.to(self.device)
        if court_vis is not None:
            court_vis = court_vis.to(self.device)

        outputs = self.model(
            human_kp=human_kp,
            court_kp=court_kp,
            human_vis=human_vis,
            human_mask=human_mask,
            court_vis=court_vis,
        )

        position = outputs["position"]
        rotation = outputs["rotation"]

        result: dict[str, Tensor] = {
            "position": position,
            "rotation": rotation,
        }

        if denormalize:
            scale = torch.tensor(
                list(self._norm_scale_xyz),
                device=position.device,
                dtype=position.dtype,
            )
            result["position_meters"] = position * scale
            result["yaw_radians"] = torch.atan2(rotation[..., 0], rotation[..., 1])

        return {k: v.detach().cpu() for k, v in result.items()}
