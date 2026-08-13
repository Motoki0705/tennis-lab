"""Inference wrapper for lifecycle-aware multi-ball track queries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor

from src.tasks.base.data.court_peaks import CourtPeakFrame
from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.base.model_io import ModelCall
from src.tasks.blcs.model_io import (
    BLCSTrackQueryPrediction,
    TrackQueryBoundModelIO,
    compose_blcs_track_query_model_io,
)
from src.tasks.blcs.model_io.checkpoints import load_checkpoint_config
from src.tasks.blcs.training.tracking_lightning_module import (
    BLCSTrackingLightningModule,
)
from src.utils.configuration import PathResolver
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


class BLCSTrackingPredictor(BasePredictor[BLCSTrackQueryPrediction]):
    """Predict fixed lifecycle queries from ID-ordered per-camera observations."""

    def __init__(
        self,
        model_io: TrackQueryBoundModelIO,
        device: torch.device,
    ) -> None:
        self.model_io = model_io
        self.model = model_io.model.to(device).eval()
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
        checkpoints = cls._ensure_checkpoint(checkpoint_path, resolver=resolver)
        if len(checkpoints) != 1:
            raise ValueError(
                f"{cls.__name__} expects exactly one checkpoint, got {len(checkpoints)}."
            )
        binding = compose_blcs_track_query_model_io(
            load_checkpoint_config(checkpoints[0])
        )
        lightning_module, resolved_device = cls._load_single_lightning_module(
            checkpoints[0],
            BLCSTrackingLightningModule,
            resolver=resolver,
            device=device,
            model_io=binding,
            strict=True,
            weights_only=False,
            **kwargs,
        )
        return cls(model_io=lightning_module.model_io, device=resolved_device)

    def predict_batch(
        self,
        batch: Mapping[str, object],
        *,
        denormalize: bool,
    ) -> BLCSTrackQueryPrediction:
        """Run one validated tracking call and return its typed decode."""
        with torch.no_grad():
            call = self.model_io.build_call(batch)
            moved = ModelCall(
                args=tuple(
                    value.to(self.device) if isinstance(value, Tensor) else value
                    for value in call.args
                ),
                kwargs={
                    key: value.to(self.device) if isinstance(value, Tensor) else value
                    for key, value in call.kwargs.items()
                },
            )
            prediction = self.model_io.decode_output(
                self.model_io.execute_call(moved)
            )
            position = prediction.position
            if denormalize:
                position = self._denormalize_coords(position, COURT_COORD_SCALE_XYZ)
            return BLCSTrackQueryPrediction(
                position=position.detach().cpu(),
                presence_logits=prediction.presence_logits.detach().cpu(),
                presence_probability=prediction.presence_probability.detach().cpu(),
                presence=prediction.presence.detach().cpu(),
            )

    def predict(
        self,
        *,
        ball_uv: Tensor,
        ball_score: Tensor,
        ball_visible: Tensor,
        candidate_mask: Tensor,
        frame_mask: Tensor,
        view_mask: Tensor,
        reference_view_index: Tensor,
        court_kp: Tensor | None = None,
        court_vis: Tensor | None = None,
        court_peak_uv: Tensor | None = None,
        court_peak_score: Tensor | None = None,
        court_peak_covariance: Tensor | None = None,
        court_peak_valid: Tensor | None = None,
        court_peak_frames: Sequence[CourtPeakFrame] | None = None,
        denormalize: bool,
    ) -> BLCSTrackQueryPrediction:
        """Return the adapter's typed query-position/presence decode."""
        inputs: dict[str, object] = {
            "ball_uv": ball_uv,
            "ball_score": ball_score,
            "ball_visible": ball_visible,
            "candidate_mask": candidate_mask,
            "frame_mask": frame_mask,
            "view_mask": view_mask,
            "reference_view_index": reference_view_index,
        }
        optional = {
            "court_kp": court_kp,
            "court_vis": court_vis,
            "court_peak_uv": court_peak_uv,
            "court_peak_score": court_peak_score,
            "court_peak_covariance": court_peak_covariance,
            "court_peak_valid": court_peak_valid,
            "court_peak_frames": court_peak_frames,
        }
        inputs.update({key: value for key, value in optional.items() if value is not None})
        return self.predict_batch(
            inputs,
            denormalize=denormalize,
        )


__all__ = ["BLCSTrackingPredictor"]
