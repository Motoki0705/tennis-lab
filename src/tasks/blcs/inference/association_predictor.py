"""BLCS association inference with strict task checkpoint ownership."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from src.tasks.base.data.observation_tracking import ObservationTrackingConfig
from src.tasks.base.inference.association import predict_association_observations
from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.base.model_io.association_contracts import (
    INPUT_KEYS,
    AssociationInferencePolicy,
    AssociationObservationRequest,
    AssociationObservationResult,
)
from src.tasks.base.model_io.association_decoding import decode_association
from src.tasks.blcs.training.association_lightning_module import (
    BLCSAssociationLightningModule,
)


class BLCSAssociationPredictor(BasePredictor[dict[str, Tensor]]):
    def __init__(
        self, module: BLCSAssociationLightningModule, *, device: str = "cpu"
    ) -> None:
        self.device = torch.device(device)
        self.module = module.to(self.device).eval()
        self.model = module.model

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> BLCSAssociationPredictor:
        module = BLCSAssociationLightningModule.load_from_checkpoint(
            path, map_location="cpu", weights_only=False
        )
        return cls(module, device=device)

    def predict(self, observations: dict[str, Tensor]) -> dict[str, Tensor]:
        if set(observations) != set(INPUT_KEYS):
            raise ValueError("Expected exactly the six observation/reference tensors")
        with torch.no_grad(), torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
        ):
            inputs = {k: v.to(self.device) for k, v in observations.items()}
            raw = self.module.model_io.run(inputs)
            cpu = {k: v.detach().cpu() for k, v in raw.items()}
            return decode_association(
                cpu,
                observed=(
                    inputs["object_vis"].any(-1) & ~inputs["padding_mask"][..., None]
                ).cpu(),
                reference=inputs["reference_view_index"].cpu(),
                view_valid=(~inputs["padding_mask"]).any(-1).cpu(),
                side_threshold=float(self.module.config.metrics.side_threshold),
            )

    def predict_observations(
        self,
        request: AssociationObservationRequest,
        *,
        policy: AssociationInferencePolicy | None = None,
    ) -> AssociationObservationResult:
        if int(self.module.config.model.num_slots) != 4 or int(self.module.config.model.max_identities) != 10:
            raise ValueError("Integrated association requires four slots and ten identity classes")
        return predict_association_observations(
            self.predict, request,
            tracking=ObservationTrackingConfig.from_mapping(self.module.config.data.association),
            policy=AssociationInferencePolicy() if policy is None else policy, joints=1,
        )
