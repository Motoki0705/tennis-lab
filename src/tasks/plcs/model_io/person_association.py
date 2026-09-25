"""PLCS-only fixed-track Re-ID and independent court-side contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from src.tasks.base.model_io import ModelCall, ModelInputContractError
from src.tasks.base.models import validate_reference_context_mask

REID_MODEL = "plcs_player_reid"
SIDE_MODEL = "plcs_court_side"
MODEL_CONTRACTS = {REID_MODEL: "plcs_fixed_track_reid_v2", SIDE_MODEL: "plcs_independent_court_side_v1"}
OBSERVATION_KEYS = ("human_kp", "human_vis", "court_kp", "court_vis", "padding_mask")


@dataclass(frozen=True)
class PersonInferencePolicy:
    cosine_threshold: float | None = None
    min_frames: int = 512
    max_frames: int = 1024
    padded_views: int = 5

    def __post_init__(self) -> None:
        if self.cosine_threshold is not None and (not math.isfinite(self.cosine_threshold) or not -1 < self.cosine_threshold < 1):
            raise ValueError("Cosine threshold must be finite in (-1,1)")
        if not 1 <= self.min_frames <= self.max_frames or not 2 <= self.padded_views <= 5:
            raise ValueError("Invalid person inference frame/view budget")


@dataclass(frozen=True)
class PersonObservationRequest:
    human_kp: Tensor  # CPU (V,T,D,17,2); raw tracker carrier order
    human_vis: Tensor
    local_track_ids: Tensor  # (V,D), scene-lifetime IDs, -1 for padding
    court_kp: Tensor
    court_vis: Tensor
    camera_ids: tuple[str, ...]
    reference_camera: str
    frame_indices: Tensor

    def __post_init__(self) -> None:
        kp, visible = self.human_kp, self.human_vis
        if kp.ndim != 5 or kp.shape[-2:] != (17, 2) or visible.shape != kp.shape[:-1]:
            raise ValueError("Person requests require (V,T,D,17,2) and joint visibility")
        v, t, d = kp.shape[:3]
        if not 2 <= v <= 5 or t < 1 or self.local_track_ids.shape != (v, d) or self.local_track_ids.dtype != torch.int64:
            raise ValueError("Invalid view/timeline/scene-local track ID contract")
        if self.court_kp.shape != (v, t, 14, 2) or self.court_vis.shape != (v, t, 14):
            raise ValueError("Court and person timelines must agree")
        if len(self.camera_ids) != v or len(set(self.camera_ids)) != v or any(not x for x in self.camera_ids) or self.reference_camera not in self.camera_ids:
            raise ValueError("Invalid camera/reference IDs")
        if self.frame_indices.shape != (t,) or self.frame_indices.dtype != torch.int64 or bool((self.frame_indices < 0).any()) or bool((self.frame_indices.diff() <= 0).any()):
            raise ValueError("Source frame indices must be nonnegative strictly increasing int64")
        for value in (kp, visible, self.local_track_ids, self.court_kp, self.court_vis, self.frame_indices):
            if value.device.type != "cpu":
                raise ValueError("Person requests must be CPU tensors")
        for coordinates, mask in ((kp, visible), (self.court_kp, self.court_vis)):
            if coordinates.dtype != torch.float32 or mask.dtype != torch.bool:
                raise TypeError("Coordinates must be float32 and masks boolean")
            selected = coordinates[mask]
            if not bool(torch.isfinite(selected).all()) or bool(((selected < 0) | (selected > 1)).any()):
                raise ValueError("Visible inputs must be finite normalized UV")


@dataclass(frozen=True)
class PersonReIDResult:
    raw_track_ids: Tensor  # (V,D), global IDs or -1 for unobserved/explicitly excluded tracks
    slot_global_ids: Tensor  # (V,P)
    local_track_ids: Tensor  # (V,P)
    track_embedding: Tensor  # (V,P,D)
    track_valid: Tensor  # (V,P)
    cosine_threshold: float


@dataclass(frozen=True)
class CourtSideResult:
    side_logits: Tensor
    view_half_turns: Tensor


class PersonModelIOAdapter:
    def __init__(self, model_type: type[nn.Module], *, name: str, slots: int) -> None:
        if name not in MODEL_CONTRACTS:
            raise ValueError("Unknown PLCS person model")
        self._model_type, self.name, self.slots = model_type, name, slots

    @property
    def model_type(self) -> type[nn.Module]:
        return self._model_type

    def build_call(self, batch: Mapping[str, Tensor]) -> ModelCall:
        keys = OBSERVATION_KEYS + (("reference_view_index",) if self.name == SIDE_MODEL else ())
        values = {key: batch[key] for key in keys}
        kp, visible, court, cv, padding = (values[key] for key in OBSERVATION_KEYS)
        if kp.ndim != 6 or kp.shape[-3:] != (self.slots, 17, 2):
            raise ModelInputContractError("PLCS expects (B,V,T,P,17,2)")
        b, v, t = kp.shape[:3]
        if min(b, v, t) < 1 or visible.shape != kp.shape[:-1] or court.shape != (b, v, t, 14, 2) or cv.shape != (b, v, t, 14) or padding.shape != (b, v, t):
            raise ModelInputContractError("Person/court/mask dimensions disagree")
        if any(mask.dtype != torch.bool for mask in (visible, cv, padding)) or not kp.is_floating_point() or not court.is_floating_point():
            raise ModelInputContractError("PLCS masks must be boolean and coordinates floating point")
        if any(value.device != kp.device for value in values.values()):
            raise ModelInputContractError("PLCS inputs must share a device")
        if self.name == SIDE_MODEL:
            validate_reference_context_mask(values["reference_view_index"], ~padding)
        return ModelCall(kwargs=values)

    def decode_output(self, output: Mapping[str, Tensor]) -> dict[str, Tensor]:
        if self.name == REID_MODEL:
            if set(output) != {"track_embedding", "track_valid"}:
                raise ValueError("Re-ID output requires exactly embeddings and observation validity")
            z, valid = output["track_embedding"], output["track_valid"]
            if z.ndim != 4 or z.shape[-2] != self.slots or valid.shape != z.shape[:-1] or valid.dtype != torch.bool:
                raise ValueError("Invalid PLCS track embedding output")
        elif output["side_logits"].ndim != 2:
            raise ValueError("Side output must be (B,V)")
        return dict(output)


def validate_person_checkpoint(checkpoint: Mapping[str, Any], *, model_name: str) -> None:
    if checkpoint.get("person_association_contract") != MODEL_CONTRACTS[model_name] or checkpoint.get("person_association_model") != model_name:
        raise ValueError("PLCS person checkpoint contract/model mismatch; use a matching checkpoint or an explicit export")
