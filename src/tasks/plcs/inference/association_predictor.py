"""PLCS association inference with strict task checkpoint ownership."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.base.model_io.association_contracts import INPUT_KEYS
from src.tasks.base.model_io.association_decoding import decode_association
from src.tasks.plcs.training.association_lightning_module import (
    PLCSAssociationLightningModule,
)


class PLCSAssociationPredictor(BasePredictor[dict[str, Tensor]]):
    def __init__(
        self, module: PLCSAssociationLightningModule, *, device: str = "cpu"
    ) -> None:
        self.device = torch.device(device)
        self.module = module.to(self.device).eval()
        self.model = module.model

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> PLCSAssociationPredictor:
        module = PLCSAssociationLightningModule.load_from_checkpoint(
            path, map_location="cpu", weights_only=False
        )
        return cls(module, device=device)

    def predict(self, observations: dict[str, Tensor]) -> dict[str, Tensor]:
        if set(observations) != set(INPUT_KEYS):
            raise ValueError("Expected exactly the six observation/reference tensors")
        with torch.no_grad():
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
