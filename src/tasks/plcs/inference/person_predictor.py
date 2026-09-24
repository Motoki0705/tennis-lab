"""Separate PLCS predictors/checkpoints for track embeddings and court side."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import torch
from torch import Tensor

from src.tasks.plcs.inference.person_inputs import prepare_person_observations
from src.tasks.plcs.model_io.person_association import (
    REID_MODEL,
    SIDE_MODEL,
    CourtSideResult,
    PersonInferencePolicy,
    PersonObservationRequest,
    PersonReIDResult,
)
from src.tasks.plcs.model_io.track_matching import match_track_embeddings
from src.tasks.plcs.training.association_lightning_module import (
    PLCSAssociationLightningModule,
)


class _PersonPredictor:
    model_name: str

    def __init__(self, module: PLCSAssociationLightningModule, *, device: str = "cpu") -> None:
        if str(module.config.model.name) != self.model_name:
            raise ValueError(f"This predictor requires a {self.model_name} checkpoint")
        self.device = torch.device(device)
        self.module = module.to(self.device).eval()

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> Self:
        return cls(PLCSAssociationLightningModule.load_from_checkpoint(path, map_location="cpu", weights_only=False), device=device)

    def predict(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        with torch.no_grad(), torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            output = self.module.model_io.run({key: value.to(self.device) for key, value in batch.items()})
        return {key: value.detach().cpu() for key, value in output.items()}


class PlayerReIDPredictor(_PersonPredictor):
    model_name = REID_MODEL

    def predict_observations(self, request: PersonObservationRequest, *, policy: PersonInferencePolicy) -> PersonReIDResult:
        batch, packed = prepare_person_observations(request, policy=policy, num_slots=int(self.module.config.model.num_slots))
        output = self.predict(batch)
        views = len(request.camera_ids)
        embeddings, valid = output["track_embedding"][0, :views], output["track_valid"][0, :views]
        probability = output["is_player_logit"][0, :views].sigmoid()
        threshold = float(self.module.matching_threshold.cpu()) if policy.cosine_threshold is None else policy.cosine_threshold
        global_ids = match_track_embeddings(embeddings, valid & probability.ge(policy.min_player_probability), threshold=threshold)
        raw = torch.full(request.local_track_ids.shape, -1, dtype=torch.int64)
        for view in range(views):
            for slot, local_id in enumerate(packed.local_track_ids[view].tolist()):
                if local_id >= 0:
                    raw[view, request.local_track_ids[view].eq(local_id)] = global_ids[view, slot]
        return PersonReIDResult(raw, global_ids, packed.local_track_ids, embeddings, valid, probability, threshold)


class CourtSidePredictor(_PersonPredictor):
    model_name = SIDE_MODEL

    def predict_observations(self, request: PersonObservationRequest, *, policy: PersonInferencePolicy) -> CourtSideResult:
        batch, _ = prepare_person_observations(request, policy=policy, num_slots=int(self.module.config.model.num_slots))
        logits = self.predict(batch)["side_logits"][0, :len(request.camera_ids)].float()
        half_turn = logits.sigmoid().ge(float(self.module.config.metrics.side_threshold))
        half_turn[request.camera_ids.index(request.reference_camera)] = False
        return CourtSideResult(logits, half_turn)
