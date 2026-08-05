"""Unified PLCS inference predictor."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, ParamSpec, Self, TypeVar, cast

import torch
from torch import Tensor, nn

from src.tasks.base.inference.grad_mode import no_grad
from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.plcs.training.lightning_module import PLCSLightningModule
from src.utils.configuration import PathResolver
from src.utils.schema.court import COURT_COORD_SCALE_XYZ

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _typed_no_grad(function: Callable[_P, _R]) -> Callable[_P, _R]:
    decorator: Callable[[Callable[_P, _R]], Callable[_P, _R]] = no_grad
    return decorator(function)


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
        *,
        resolver: PathResolver,
        device: str | torch.device,
        allow_device_fallback: bool,
        **kwargs: Any,
    ) -> Self:
        model, resolved_device = cls._load_single_lightning_checkpoint(
            checkpoint_path,
            PLCSLightningModule,
            resolver=resolver,
            device=device,
            allow_device_fallback=allow_device_fallback,
            **kwargs,
        )
        return cls(model=model, device=resolved_device)

    @_typed_no_grad
    def predict(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None = None,
        human_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
        *,
        denormalize: bool,
    ) -> dict[str, Tensor]:
        """Predict player 3D position and orientation from caller-provided tensors."""

        moved = self._to_device(
            self.device,
            human_kp,
            court_kp,
            human_vis,
            human_mask,
            court_vis,
        )
        moved_human_kp, moved_court_kp, human_vis, human_mask, court_vis = moved
        if moved_human_kp is None or moved_court_kp is None:
            raise AssertionError("Required predictor inputs became None.")

        outputs = cast(
            "dict[str, Tensor]",
            self.model(
                human_kp=moved_human_kp,
                court_kp=moved_court_kp,
                human_vis=human_vis,
                human_mask=human_mask,
                court_vis=court_vis,
            ),
        )

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
