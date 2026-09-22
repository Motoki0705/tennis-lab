"""Lazy task-owned inference on unassociated camera-local observations."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from src.tasks.base.data.observation_tracking import ObservationTrackingConfig
from src.tasks.base.model_io.association_contracts import (
    AssociationInferencePolicy,
    AssociationObservationRequest,
    AssociationObservationResult,
)
from src.tasks.blcs.inference.association_predictor import BLCSAssociationPredictor
from src.tasks.plcs.inference.association_predictor import PLCSAssociationPredictor


class ViewAssociationModule:
    """Inference on unassociated camera-local player/ball observations."""

    def __init__(self, checkpoint: Path, *, task: Literal["plcs", "blcs"], device: str = "cpu") -> None:
        if task not in ("plcs", "blcs"):
            raise ValueError("Unknown association task")
        self.checkpoint, self.task, self.device = checkpoint, task, device
        self.predictor: PLCSAssociationPredictor | BLCSAssociationPredictor | None = None
        self.tracking: ObservationTrackingConfig | None = None

    def load(self) -> None:
        if self.predictor is None:
            self.predictor = (PLCSAssociationPredictor.load(self.checkpoint, device=self.device) if self.task == "plcs" else BLCSAssociationPredictor.load(self.checkpoint, device=self.device))
            self.tracking = ObservationTrackingConfig.from_mapping(self.predictor.module.config.data.association)

    def unload(self) -> None:
        if self.predictor is None:
            return
        self.predictor = None
        import gc
        gc.collect()
        if torch.device(self.device).type == "cuda":
            torch.cuda.empty_cache()

    def process_observations(
        self, request: AssociationObservationRequest, *, policy: AssociationInferencePolicy,
    ) -> AssociationObservationResult | None:
        if not bool(request.object_vis.any()):
            return None
        self.load()
        assert self.predictor is not None
        try:
            return self.predictor.predict_observations(request, policy=policy)
        finally:
            self.unload()
