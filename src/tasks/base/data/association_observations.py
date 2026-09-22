"""Training-equivalent local tracking for unbatched association inference."""

from __future__ import annotations

import torch
from torch import Tensor

from src.tasks.base.data.observation_tracking import (
    ObservationTrackingConfig,
    track_multiview_observations,
)
from src.tasks.base.model_io.association_contracts import (
    AssociationInferencePolicy,
    AssociationObservationRequest,
)


def prepare_association_observations(
    request: AssociationObservationRequest,
    *,
    tracking: ObservationTrackingConfig,
    policy: AssociationInferencePolicy,
    joints: int,
) -> tuple[dict[str, Tensor], Tensor]:
    views, length = request.object_uv.shape[:2]
    if request.object_uv.shape[-2] != joints:
        raise ValueError(f"Association task requires J={joints}")
    if length > policy.max_frames:
        raise ValueError(f"Association clip length {length} exceeds {policy.max_frames}; no implicit crop/windowing")
    tracked = track_multiview_observations(
        request.object_uv.masked_fill(~request.object_vis[..., None], 0),
        request.object_vis,
        num_slots=4,
        config=tracking,
    )
    size = max(length, policy.min_frames)
    uv = torch.zeros(1, policy.padded_views, size, 4, joints, 2)
    visible = torch.zeros(uv.shape[:-1], dtype=torch.bool)
    court = torch.zeros(1, policy.padded_views, size, 14, 2)
    court_visible = torch.zeros(court.shape[:-1], dtype=torch.bool)
    padding = torch.ones(1, policy.padded_views, size, dtype=torch.bool)
    uv[0, :views, :length] = tracked.values
    visible[0, :views, :length] = tracked.visibility
    court[0, :views, :length] = request.court_kp.masked_fill(~request.court_vis[..., None], 0)
    court_visible[0, :views, :length] = request.court_vis
    padding[0, :views, :length] = False
    return {
        "object_uv": uv,
        "object_vis": visible,
        "court_kp": court,
        "court_vis": court_visible,
        "padding_mask": padding,
        "reference_view_index": torch.tensor([request.camera_ids.index(request.reference_camera)], dtype=torch.int64),
    }, tracked.detection_indices
