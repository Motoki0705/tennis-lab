"""Tensor and checkpoint contracts for view-side / clip identity association."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from src.tasks.base.model_io import ModelCall, ModelInputContractError
from src.tasks.base.models import validate_reference_context_mask

ASSOCIATION_CONTRACT = "camera_local_global_mha_mhc_association_v2"
INPUT_KEYS = (
    "object_uv",
    "object_vis",
    "court_kp",
    "court_vis",
    "padding_mask",
    "reference_view_index",
)


class AssociationIOAdapter:
    def __init__(
        self, model_type: type[nn.Module], *, joints: int, slots: int, identities: int
    ) -> None:
        self._model_type = model_type
        self.joints, self.slots, self.identities = joints, slots, identities

    @property
    def model_type(self) -> type[nn.Module]:
        return self._model_type

    def build_call(self, batch: Mapping[str, Tensor]) -> ModelCall:
        values = {key: batch[key] for key in INPUT_KEYS}
        uv, vis, court, cv, padding, reference = (values[k] for k in INPUT_KEYS)
        if uv.ndim != 6 or uv.shape[-3:] != (self.slots, self.joints, 2):
            raise ModelInputContractError("object_uv must match (B,V,T,P,J,2)")
        b, v, t = uv.shape[:3]
        if (
            min(b, v, t) <= 0
            or vis.shape != uv.shape[:-1]
            or court.shape != (b, v, t, 14, 2)
            or cv.shape != (b, v, t, 14)
            or padding.shape != (b, v, t)
        ):
            raise ModelInputContractError("Association tensor shapes disagree")
        if any(x.dtype != torch.bool for x in (vis, cv, padding)):
            raise ModelInputContractError("Association masks must be boolean")
        if not uv.is_floating_point() or not court.is_floating_point():
            raise ModelInputContractError(
                "Association coordinates must be floating point"
            )
        if any(value.device != uv.device for value in values.values()):
            raise ModelInputContractError("All association inputs must share a device")
        validate_reference_context_mask(reference, ~padding)
        return ModelCall(kwargs=values)

    def decode_output(self, output: Mapping[str, Tensor]) -> dict[str, Tensor]:
        side, identities = output["side_logits"], output["object_id_logits"]
        if (
            side.ndim != 2
            or identities.ndim != 5
            or identities.shape[:2] != side.shape
            or identities.shape[-2:] != (self.slots, self.identities + 1)
        ):
            raise ValueError("Invalid association model output")
        return dict(output)


def validate_association_checkpoint(
    checkpoint: Mapping[str, Any], *, model_name: str
) -> None:
    if (
        checkpoint.get("association_contract") != ASSOCIATION_CONTRACT
        or checkpoint.get("association_model") != model_name
    ):
        raise ValueError(
            "Association checkpoint contract/task mismatch; retraining is required"
        )
