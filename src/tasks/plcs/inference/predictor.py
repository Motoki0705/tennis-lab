"""Unified PLCS inference predictor."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Self, cast

import torch
from torch import Tensor, nn

from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.plcs.training.lightning_module import PLCSLightningModule
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
        model, resolved_device = cls._load_single_lightning_checkpoint(
            checkpoint_path,
            PLCSLightningModule,
            device,
            **kwargs,
        )
        return cls(model=model, device=resolved_device)

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def predict(
        self,
        human_kp: Tensor,
        court_kp: Tensor | None = None,
        human_vis: Tensor | None = None,
        human_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
        denormalize: bool = True,
        court_line_map: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Predict player 3D position and orientation from caller-provided tensors."""

        moved = self._to_device(
            self.device,
            human_kp,
            court_kp,
            human_vis,
            human_mask,
            court_vis,
            court_line_map,
        )
        human_kp = cast(Tensor, moved[0])
        court_kp, human_vis, human_mask, court_vis, court_line_map = moved[1:]

        court_input_type = str(getattr(self.model, "court_input_type", "kp"))
        if court_input_type == "line":
            if (
                court_line_map is None
                or court_kp is not None
                or court_vis is not None
            ):
                raise ValueError(
                    "Line-based PLCS inference requires court_line_map and rejects "
                    "court_kp/court_vis."
                )
            outputs = self.model(
                human_kp=human_kp,
                court_line_map=court_line_map,
                human_vis=human_vis,
                human_mask=human_mask,
            )
        elif court_input_type == "kp":
            if court_kp is None or court_line_map is not None:
                raise ValueError(
                    "KP-based PLCS inference requires court_kp and rejects "
                    "court_line_map."
                )
            outputs = self.model(
                human_kp=human_kp,
                court_kp=court_kp,
                human_vis=human_vis,
                human_mask=human_mask,
                court_vis=court_vis,
            )
        else:
            raise ValueError(f"Unsupported court_input_type={court_input_type!r}.")

        position = outputs["position"]
        rotation = outputs["rotation"]

        result: dict[str, Tensor] = {
            "position": position,
            "rotation": rotation,
        }

        canonical_pose = outputs.get("canonical_pose")
        if canonical_pose is not None:
            result["canonical_pose"] = canonical_pose

        if denormalize:
            result["position_meters"] = self._denormalize_coords(
                position, self._norm_scale_xyz
            )
            result["yaw_radians"] = torch.atan2(rotation[..., 1], rotation[..., 0])

        return {k: v.detach().cpu() for k, v in result.items()}
