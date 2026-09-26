"""Strict tracker-ID slot packing and explicit temporal/view padding."""

from __future__ import annotations

import torch
from torch import Tensor

from src.tasks.plcs.data.tracked_slots import FixedTrackRegistry, TrackedSlots
from src.tasks.plcs.model_io.person_association import (
    PersonInferencePolicy,
    PersonObservationRequest,
)


def prepare_person_observations(request: PersonObservationRequest, *, policy: PersonInferencePolicy, num_slots: int) -> tuple[dict[str, Tensor], TrackedSlots]:
    views, length = request.human_kp.shape[:2]
    if length > policy.max_frames or views > policy.padded_views:
        raise ValueError("Person observations exceed the explicit inference budget")
    packed = FixedTrackRegistry(request.camera_ids, num_slots=num_slots).pack(request.human_kp, request.human_vis, request.local_track_ids)
    frames = max(policy.min_frames, length)
    batch = {
        "human_kp": torch.zeros(1, policy.padded_views, frames, num_slots, 17, 2),
        "human_vis": torch.zeros(1, policy.padded_views, frames, num_slots, 17, dtype=torch.bool),
        "court_kp": torch.zeros(1, policy.padded_views, frames, 14, 2),
        "court_vis": torch.zeros(1, policy.padded_views, frames, 14, dtype=torch.bool),
        "padding_mask": torch.ones(1, policy.padded_views, frames, dtype=torch.bool),
        "reference_view_index": torch.tensor([request.camera_ids.index(request.reference_camera)]),
    }
    batch["human_kp"][0, :views, :length] = packed.keypoints
    batch["human_vis"][0, :views, :length] = packed.visibility
    batch["court_kp"][0, :views, :length] = request.court_kp.masked_fill(~request.court_vis[..., None], 0)
    batch["court_vis"][0, :views, :length] = request.court_vis
    batch["padding_mask"][0, :views, :length] = False
    return batch, packed
