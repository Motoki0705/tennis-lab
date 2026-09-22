"""Strict BLCS axial reference boundaries for forward and checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import cast

import torch
from torch import nn

from src.tasks.base.model_io import (
    ModelCall,
    ModelInputContractError,
    TensorSpec,
    require_tensor,
)
from src.tasks.base.models import validate_reference_context_mask
from src.tasks.blcs.axial_reference_contract import AXIAL_REFERENCE_CONTRACT
from src.tasks.blcs.model_io.adapters import (
    AxialTrajectoryModelIOAdapter,
)
from src.tasks.blcs.model_io.contracts import blcs_reference_metadata_from_batch
from src.tasks.blcs.models.blcs_multiview_axial_reference_model import (
    BLCSMultiViewAxialReferenceModel,
)


class AxialReferenceTrajectoryModelIOAdapter(AxialTrajectoryModelIOAdapter):
    """Require validated stable identity and a non-padding reference view."""

    @property
    def model_type(self) -> type[nn.Module]:
        return cast("type[nn.Module]", BLCSMultiViewAxialReferenceModel)

    def _validate_inference_observations(self, batch: Mapping[str, object]) -> None:
        # Array builders precede the explicit reference metadata binding in predictor.
        super().build_call(batch)

    def build_call(self, batch: Mapping[str, object]) -> ModelCall:
        call = super().build_call(batch)
        padding = cast(torch.Tensor, call.kwargs["padding_mask"])
        if self.court_keypoint_contract.selector != "camera_view_v2":
            raise ModelInputContractError("Axial reference requires camera_view_v2.")
        counts = (~padding.all(dim=-1)).sum(dim=-1)
        if not bool(((counts >= 3) & (counts <= 4)).all()):
            raise ModelInputContractError(
                "Axial reference requires 3 or 4 non-padding cameras."
            )
        reference = require_tensor(
            batch,
            "reference_view_index",
            spec=TensorSpec(shape=(padding.shape[0],), dtypes=frozenset({torch.int64})),
        )
        try:
            validate_reference_context_mask(reference, ~padding)
            if "reference_view_selection" in batch:
                metadata = blcs_reference_metadata_from_batch(batch)
                if metadata is None or not torch.equal(
                    reference, metadata.reference_view_index
                ):
                    raise ValueError(
                        "Reference index and geometry provenance disagree."
                    )
        except (TypeError, ValueError) as error:
            raise ModelInputContractError(str(error)) from error
        return ModelCall(kwargs={**call.kwargs, "reference_view_index": reference})


def write_axial_reference_checkpoint(
    checkpoint: MutableMapping[str, object], *, model_name: str
) -> None:
    if model_name == "blcs_multiview_axial_reference":
        checkpoint["axial_reference"] = dict(AXIAL_REFERENCE_CONTRACT)


def validate_axial_reference_checkpoint(
    checkpoint: Mapping[str, object], *, model_name: str
) -> None:
    marker = checkpoint.get("axial_reference")
    if model_name == "blcs_multiview_axial_reference":
        if marker != AXIAL_REFERENCE_CONTRACT:
            raise ValueError(
                "Missing or mismatched BLCS axial reference checkpoint contract."
            )
    elif marker is not None:
        raise ValueError(
            "Axial reference checkpoint requires its exact reference model."
        )
