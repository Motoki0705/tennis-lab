"""Inference boundary for a once-bound PLCS track-query model."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor, nn

from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.base.model_io import ModelCall, ModelInputContractError
from src.tasks.base.training.tracking_metrics import TrackingMetricConfig
from src.tasks.plcs.model_io import (
    PLCSPreparedBatch,
    PLCSTrackingBoundModelIO,
    PLCSTrackQueryIOAdapter,
    bind_plcs_model_io,
)
from src.tasks.plcs.training.tracking_lightning_module import (
    PLCSTrackingLightningModule,
)
from src.utils.configuration import PathResolver
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


class PLCSTrackingPredictor(BasePredictor):
    """Predict fixed lifecycle queries through the track-query adapter."""

    def __init__(
        self,
        *,
        model: nn.Module,
        adapter: PLCSTrackQueryIOAdapter,
        device: torch.device,
    ) -> None:
        bound = bind_plcs_model_io(model, adapter)
        self.model_io: PLCSTrackingBoundModelIO = bound
        self.model = self.model_io.model.to(device).eval()
        self.io_adapter = adapter
        self.device = device

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path | Iterable[str | Path],
        *,
        resolver: PathResolver,
        device: str | torch.device,
        **kwargs: Any,
    ) -> Self:
        lightning_module, resolved_device = cls._load_single_lightning_module(
            checkpoint_path,
            PLCSTrackingLightningModule,
            resolver=resolver,
            device=device,
            strict=bool(kwargs.pop("strict", True)),
            weights_only=bool(kwargs.pop("weights_only", False)),
            **kwargs,
        )
        adapter = lightning_module.io_adapter
        if not isinstance(adapter, PLCSTrackQueryIOAdapter):
            raise ModelInputContractError(
                "Loaded checkpoint does not contain a PLCS track-query adapter."
            )
        return cls(
            model=lightning_module.model,
            adapter=adapter,
            device=resolved_device,
        )

    def predict(
        self,
        *,
        human_kp: Tensor,
        human_vis: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        padding_mask: Tensor,
        tracking_metrics: TrackingMetricConfig,
        denormalize: bool,
    ) -> dict[str, Tensor]:
        """Return query position/rotation and lifecycle presence outputs."""
        with torch.no_grad():
            prepared = PLCSPreparedBatch(
                call=self.io_adapter.build_call(
                    {
                        "human_kp": human_kp,
                        "human_vis": human_vis,
                        "court_kp": court_kp,
                        "court_vis": court_vis,
                        "padding_mask": padding_mask,
                    }
                )
            )
            moved = ModelCall(
                kwargs={
                    key: value.to(self.device) if isinstance(value, Tensor) else None
                    for key, value in prepared.call.kwargs.items()
                }
            )
            raw_output = self.model_io.execute_call(moved)
            decoded = self.model_io.decode_output(raw_output)
            presence_logits = decoded.presence_logits
            probability = presence_logits.sigmoid()
            result = {
                "position": decoded.position,
                "rotation": decoded.rotation,
                "presence_logits": presence_logits,
                "presence_probability": probability,
                "presence": probability >= tracking_metrics.presence_threshold,
            }
            if denormalize:
                result["position_meters"] = self._denormalize_coords(
                    decoded.position, COURT_COORD_SCALE_XYZ
                )
                result["yaw_radians"] = torch.atan2(
                    decoded.rotation[..., 1], decoded.rotation[..., 0]
                )
            return {key: value.detach().cpu() for key, value in result.items()}


__all__ = ["PLCSTrackingPredictor"]
