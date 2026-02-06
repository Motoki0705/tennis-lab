"""PLCS keypoint-3D inference predictor."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor

from src.base.inference.predictor import BasePredictor
from src.plcs.models.plcs_kp3d_model import PLCSKeypoint3DModel
from src.plcs.training.lightning_module_kp3d import PLCSKeypoint3DLightningModule


class PLCSKeypoint3DPredictor(BasePredictor):
    """Predict per-keypoint 3D coordinates from 2D keypoints."""

    def __init__(self, model: PLCSKeypoint3DModel, device: torch.device) -> None:
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> Self:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        device = torch.device(device)
        lightning_module = PLCSKeypoint3DLightningModule.load_from_checkpoint(
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
    ) -> dict[str, Tensor]:
        human_kp = human_kp.to(self.device)
        court_kp = court_kp.to(self.device)
        if human_vis is not None:
            human_vis = human_vis.to(self.device)
        if court_vis is not None:
            court_vis = court_vis.to(self.device)

        outputs = self.model(human_kp, court_kp, human_vis, court_vis)
        return {"player_kp_3d": outputs["player_kp_3d"].cpu()}
