"""Independently loaded PLCS player Re-ID and court-side pipeline components."""

from __future__ import annotations

import gc
from pathlib import Path

import torch

from src.tasks.plcs.inference.person_predictor import (
    CourtSidePredictor,
    PlayerReIDPredictor,
)
from src.tasks.plcs.model_io.person_association import (
    CourtSideResult,
    PersonInferencePolicy,
    PersonObservationRequest,
    PersonReIDResult,
)


class PlayerReIDModule:
    def __init__(self, checkpoint: Path, *, device: str = "cpu") -> None:
        self.checkpoint, self.device = checkpoint, device

    def process_observations(self, request: PersonObservationRequest, *, policy: PersonInferencePolicy) -> PersonReIDResult | None:
        if not bool(request.human_vis.any()):
            return None
        predictor = PlayerReIDPredictor.load(self.checkpoint, device=self.device)
        try:
            return predictor.predict_observations(request, policy=policy)
        finally:
            del predictor
            gc.collect()
            if torch.device(self.device).type == "cuda":
                torch.cuda.empty_cache()


class CourtSideModule:
    def __init__(self, checkpoint: Path, *, device: str = "cpu") -> None:
        self.checkpoint, self.device = checkpoint, device

    def process_observations(self, request: PersonObservationRequest, *, policy: PersonInferencePolicy) -> CourtSideResult:
        predictor = CourtSidePredictor.load(self.checkpoint, device=self.device)
        try:
            return predictor.predict_observations(request, policy=policy)
        finally:
            del predictor
            gc.collect()
            if torch.device(self.device).type == "cuda":
                torch.cuda.empty_cache()
