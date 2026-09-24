"""Tensor and checkpoint contracts for view-side / clip identity association."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
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


class IdentityReason(IntEnum):
    ACCEPTED = 0
    MISSING = 1
    FALSE_POSITIVE = 2
    AMBIGUOUS = 3


@dataclass(frozen=True)
class AssociationInferencePolicy:
    """Operating thresholds, independent of the trained architecture/loss."""

    min_probability: float = 0.5
    min_assignment_gap: float = math.log(2)
    min_frames: int = 512
    max_frames: int = 1024
    padded_views: int = 5

    def __post_init__(self) -> None:
        if not 0 < self.min_probability <= 1:
            raise ValueError("Identity probability must be in (0,1]")
        if not math.isfinite(self.min_assignment_gap) or self.min_assignment_gap <= 0:
            raise ValueError("Identity assignment gap must be finite and positive")
        if not 1 <= self.min_frames <= self.max_frames or self.padded_views != 5:
            raise ValueError("Association requires a valid frame budget and five padded views")


@dataclass(frozen=True)
class AssociationObservationRequest:
    """Untracked CPU observations, normalized by image (width,height).

    Carriers D have no cross-view identity. Source frame indices are provenance,
    never model features. CourtKP14 is in each camera's unchanged local order.
    """

    object_uv: Tensor  # (V,L,D,J,2)
    object_vis: Tensor  # (V,L,D,J)
    court_kp: Tensor  # (V,L,14,2)
    court_vis: Tensor  # (V,L,14)
    camera_ids: tuple[str, ...]
    reference_camera: str
    frame_indices: Tensor  # (L,) source indices

    def __post_init__(self) -> None:
        uv, visible = self.object_uv, self.object_vis
        if uv.ndim != 5 or uv.shape[-1] != 2:
            raise ValueError("Raw objects must have shape (V,L,D,J,2)")
        v, length = uv.shape[:2]
        if not 3 <= v <= 5 or length < 1 or visible.shape != uv.shape[:-1]:
            raise ValueError("Association requires 3..5 views and a nonempty timeline")
        if self.court_kp.shape != (v, length, 14, 2) or self.court_vis.shape != (v, length, 14):
            raise ValueError("Court and object timelines must agree")
        if len(self.camera_ids) != v or len(set(self.camera_ids)) != v or any(not x for x in self.camera_ids):
            raise ValueError("Unique camera IDs must match the view axis")
        if self.reference_camera not in self.camera_ids:
            raise ValueError("Reference camera is absent")
        if self.frame_indices.shape != (length,) or self.frame_indices.dtype != torch.int64:
            raise ValueError("Source frame indices must be int64 (L,)")
        if bool((self.frame_indices < 0).any()) or bool((self.frame_indices.diff() <= 0).any()):
            raise ValueError("Source frame indices must increase strictly")
        for value in (uv, visible, self.court_kp, self.court_vis, self.frame_indices):
            if value.device.type != "cpu":
                raise ValueError("Observation requests must be on CPU")
        for coordinates, mask in ((uv, visible), (self.court_kp, self.court_vis)):
            if coordinates.dtype != torch.float32 or mask.dtype != torch.bool:
                raise TypeError("Coordinates must be float32 and visibility boolean")
            selected = coordinates[mask]
            if not bool(torch.isfinite(selected).all()) or bool(((selected < 0) | (selected > 1)).any()):
                raise ValueError("Visible observations must be finite normalized UV")


@dataclass(frozen=True)
class AssociationObservationResult:
    camera_ids: tuple[str, ...]
    reference_camera: str
    frame_indices: Tensor
    raw_ids: Tensor  # (V,L,D), accepted IDs only
    raw_reasons: Tensor  # uint8 (V,L,D), IdentityReason
    slot_ids: Tensor  # (V,L,4)
    detection_indices: Tensor  # (V,L,4), slot -> raw carrier
    side_logits: Tensor  # (V,)
    view_half_turns: Tensor  # bool (V,)
    object_id_logits: Tensor  # (V,L,4,11)
    assignment_probabilities: Tensor  # (V,L,4)
    assignment_gaps: Tensor  # (V,L,4)

    def __post_init__(self) -> None:
        views, length = len(self.camera_ids), len(self.frame_indices)
        if self.reference_camera not in self.camera_ids or self.raw_ids.ndim != 3 or self.raw_ids.shape[:2] != (views, length):
            raise ValueError("Association result camera/timeline mismatch")
        if self.raw_reasons.shape != self.raw_ids.shape or self.raw_ids.dtype != torch.int64 or self.raw_reasons.dtype != torch.uint8:
            raise ValueError("Invalid raw association IDs/reasons")
        if bool(((self.raw_ids < -1) | (self.raw_ids >= 10)).any()) or not torch.equal(self.raw_ids >= 0, self.raw_reasons == int(IdentityReason.ACCEPTED)):
            raise ValueError("Accepted IDs and rejection reasons disagree")
        if self.slot_ids.shape != (views, length, 4) or self.detection_indices.shape != self.slot_ids.shape:
            raise ValueError("Association tracking mapping shape mismatch")
        if self.side_logits.shape != (views,) or self.view_half_turns.shape != (views,) or self.view_half_turns.dtype != torch.bool:
            raise ValueError("Invalid side output")
        if bool(self.view_half_turns[self.camera_ids.index(self.reference_camera)]):
            raise ValueError("Reference side must be false")
        if self.object_id_logits.shape != (views, length, 4, 11):
            raise ValueError("Invalid identity logit dimensions")
        if not all(bool(torch.isfinite(value).all()) for value in (self.side_logits, self.object_id_logits, self.assignment_probabilities, self.assignment_gaps)):
            raise ValueError("Non-finite association output")


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
