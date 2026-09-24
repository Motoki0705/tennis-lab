"""Shared orchestration behind task-owned raw-observation predictors."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from src.tasks.base.data.association_observations import (
    prepare_association_observations,
)
from src.tasks.base.data.observation_tracking import ObservationTrackingConfig
from src.tasks.base.model_io.association_contracts import (
    AssociationInferencePolicy,
    AssociationObservationRequest,
    AssociationObservationResult,
    IdentityReason,
)
from src.tasks.base.model_io.association_decoding import decode_reliable_identities


def predict_association_observations(
    predict: Callable[[dict[str, Tensor]], dict[str, Tensor]],
    request: AssociationObservationRequest,
    *,
    tracking: ObservationTrackingConfig,
    policy: AssociationInferencePolicy,
    joints: int,
) -> AssociationObservationResult:
    inputs, indices = prepare_association_observations(request, tracking=tracking, policy=policy, joints=joints)
    views, length, carriers = request.object_uv.shape[:3]
    output = predict(inputs)
    logits = output["object_id_logits"][0, :views, :length]
    observed = inputs["object_vis"][0, :views, :length].any(-1)
    decoded = decode_reliable_identities(logits, observed, policy=policy)
    raw_ids = torch.full((views, length, carriers), -1, dtype=torch.int64)
    raw_reasons = torch.full(raw_ids.shape, int(IdentityReason.MISSING), dtype=torch.uint8)
    v, t, slot = (indices >= 0).nonzero(as_tuple=True)
    carrier = indices[v, t, slot]
    raw_ids[v, t, carrier] = decoded["ids"][v, t, slot]
    raw_reasons[v, t, carrier] = decoded["reasons"][v, t, slot]
    return AssociationObservationResult(
        camera_ids=request.camera_ids,
        reference_camera=request.reference_camera,
        frame_indices=request.frame_indices.clone(),
        raw_ids=raw_ids,
        raw_reasons=raw_reasons,
        slot_ids=decoded["ids"],
        detection_indices=indices,
        side_logits=output["side_logits"][0, :views].float().cpu(),
        view_half_turns=output["view_half_turns"][0, :views].cpu(),
        object_id_logits=logits.float().cpu(),
        assignment_probabilities=decoded["probabilities"],
        assignment_gaps=decoded["gaps"],
    )
