"""Strict axial reference input and checkpoint contracts."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import cast

import torch

from src.tasks.base.model_io import (
    ModelCall,
    ModelInputContractError,
    TensorSpec,
    require_tensor,
)
from src.tasks.base.models import validate_reference_context_mask
from src.tasks.plcs.axial_reference_contract import AXIAL_REFERENCE_CONTRACT
from src.tasks.plcs.model_io.adapters import PLCSModelIOAdapter, _court_context
from src.tasks.plcs.model_io.contracts import plcs_reference_metadata_from_batch


class PLCSAxialReferenceIOAdapter(PLCSModelIOAdapter):
    """Validate identity and non-padding reference context outside compiled forward."""

    def build_call(self, batch: Mapping[str, object]) -> ModelCall:
        call = super().build_call(batch)
        padding = cast(torch.Tensor, call.kwargs["padding_mask"])
        batch_size = padding.shape[0]
        valid_views = (~padding.all(dim=-1)).sum(dim=-1)
        if not bool(((valid_views >= 3) & (valid_views <= 4)).all().item()):
            raise ModelInputContractError("Axial reference requires 3 or 4 non-padding cameras per sample.")
        reference = require_tensor(
            batch,
            "reference_view_index",
            spec=TensorSpec(
                shape=(batch_size,),
                dtypes=frozenset({torch.int64}),
            ),
        )
        try:
            validate_reference_context_mask(reference, ~padding)
            provenance = _court_context(
                batch, self.court_keypoint_contract, batch_size=batch_size
            )
            expected = [p.reference_camera_local_index for p in provenance]
            if len(expected) == 1:
                expected *= batch_size
            if len(expected) != batch_size or reference.tolist() != expected:
                raise ValueError(
                    "reference_view_index does not match the explicit reference provenance."
                )
            metadata = (
                plcs_reference_metadata_from_batch(batch)
                if any(
                    key in batch
                    for key in (
                        "view_camera_ids",
                        "reference_view_selection",
                        "reference_camera_id",
                        "stable_camera_id_table",
                        "reference_from_physical",
                        "physical_from_reference",
                    )
                )
                else None
            )
            if metadata is not None and not torch.equal(
                metadata.view_camera_ids.ge(0), ~padding.all(dim=-1)
            ):
                raise ValueError("Reference camera IDs do not match padding_mask.")
            if metadata is not None:
                contexts = (
                    provenance
                    if len(provenance) == batch_size
                    else provenance * batch_size
                )
                if any(
                    selection.provenance != context
                    for selection, context in zip(
                        metadata.selections, contexts, strict=True
                    )
                ):
                    raise ValueError(
                        "Reference metadata and court provenance disagree."
                    )
            if self.court_keypoint_contract.selector != "camera_view_v2":
                raise ValueError("Axial reference requires camera_view_v2.")
        except (TypeError, ValueError) as error:
            raise ModelInputContractError(str(error)) from error
        return ModelCall(kwargs={**call.kwargs, "reference_view_index": reference})


def write_axial_reference_checkpoint(
    checkpoint: MutableMapping[str, object], *, model_name: str
) -> None:
    if model_name == "plcs_multiview_axial_reference":
        checkpoint["axial_reference"] = dict(AXIAL_REFERENCE_CONTRACT)


def validate_axial_reference_checkpoint(
    checkpoint: Mapping[str, object], *, model_name: str
) -> None:
    marker = checkpoint.get("axial_reference")
    if model_name == "plcs_multiview_axial_reference":
        if marker != AXIAL_REFERENCE_CONTRACT:
            raise ValueError(
                "Missing or mismatched PLCS axial reference checkpoint contract."
            )
    elif marker is not None:
        raise ValueError(
            "Axial reference checkpoint requires its exact reference model."
        )
