"""Unified PLCS inference predictor."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor, nn

from src.base.inference.predictor import BasePredictor
from src.plcs.training.lightning_module import PLCSLightningModule
from src.utils.schema.keypoint_schema import COURT_COORD_SCALE_XYZ


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

    def _normalize_inputs(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None,
        human_mask: Tensor | None,
        court_vis: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None]:
        # human_kp -> (B,N,T,17,2)
        if human_kp.dim() == 3:  # (T,17,2)
            human_kp = human_kp.unsqueeze(0).unsqueeze(0)
        elif human_kp.dim() == 4:  # (N,T,17,2)
            human_kp = human_kp.unsqueeze(0)
        elif human_kp.dim() != 5:
            raise ValueError(
                "human_kp must be (T,17,2), (N,T,17,2), or (B,N,T,17,2), "
                f"got {tuple(human_kp.shape)}"
            )

        # court_kp -> (B,N,T,20,2)
        if court_kp.dim() == 3:  # (T,20,2)
            court_kp = court_kp.unsqueeze(0).unsqueeze(0)
        elif court_kp.dim() == 4:  # (N,T,20,2)
            court_kp = court_kp.unsqueeze(0)
        elif court_kp.dim() != 5:
            raise ValueError(
                "court_kp must be (T,20,2), (N,T,20,2), or (B,N,T,20,2), "
                f"got {tuple(court_kp.shape)}"
            )

        B, N, T = human_kp.shape[:3]

        if human_vis is not None:
            if human_vis.dim() == 2:  # (T,17)
                human_vis = human_vis.unsqueeze(0).unsqueeze(0)
            elif human_vis.dim() == 3:  # (N,T,17)
                human_vis = human_vis.unsqueeze(0)
            elif human_vis.dim() != 4:
                raise ValueError(
                    "human_vis must be (T,17), (N,T,17), or (B,N,T,17), "
                    f"got {tuple(human_vis.shape)}"
                )

        if court_vis is not None:
            if court_vis.dim() == 2:  # (T,20)
                court_vis = court_vis.unsqueeze(0).unsqueeze(0)
            elif court_vis.dim() == 3:  # (N,T,20)
                court_vis = court_vis.unsqueeze(0)
            elif court_vis.dim() != 4:
                raise ValueError(
                    "court_vis must be (T,20), (N,T,20), or (B,N,T,20), "
                    f"got {tuple(court_vis.shape)}"
                )

        if human_mask is not None:
            if human_mask.dim() == 1:  # (T,)
                human_mask = human_mask.unsqueeze(0).unsqueeze(0)
            elif human_mask.dim() == 2:  # (N,T)
                human_mask = human_mask.unsqueeze(0)
            elif human_mask.dim() != 3:
                raise ValueError(
                    "human_mask must be (T,), (N,T), or (B,N,T), "
                    f"got {tuple(human_mask.shape)}"
                )
            if human_mask.shape != (B, N, T):
                raise ValueError(
                    f"human_mask shape must be {(B, N, T)}, got {tuple(human_mask.shape)}"
                )

        return human_kp, court_kp, human_vis, human_mask, court_vis

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
        """Predict player 3D position and orientation from unified PLCS input."""
        human_kp, court_kp, human_vis, human_mask, court_vis = self._normalize_inputs(
            human_kp,
            court_kp,
            human_vis,
            human_mask,
            court_vis,
        )

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
