"""Lazy task-owned inference on unassociated camera-local observations."""

from __future__ import annotations

from pathlib import Path

import torch

from src.tasks.base.data.observation_tracking import ObservationTrackingConfig
from src.tasks.base.model_io.association_contracts import (
    AssociationInferencePolicy,
    AssociationObservationRequest,
    AssociationObservationResult,
)
from src.tasks.plcs.inference.association_predictor import PLCSAssociationPredictor


class ViewAssociationModule:
    """Inference on unassociated camera-local player observations."""

    def __init__(self, checkpoint: Path, *, device: str = "cpu") -> None:
        self.checkpoint, self.device = checkpoint, device
        self.predictor: PLCSAssociationPredictor | None = None
        self.tracking: ObservationTrackingConfig | None = None

    def load(self) -> None:
        if self.predictor is None:
            self.predictor = PLCSAssociationPredictor.load(self.checkpoint, device=self.device)
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
