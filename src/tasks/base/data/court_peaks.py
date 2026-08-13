"""CourtKP7 semantic multi-peak contracts shared by tracking tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, runtime_checkable

import torch
from torch import Tensor

from src.utils.schema.court import (
    COURT_PHYSICAL_INDICES_BY_SEMANTIC_CLASS,
    COURT_SEMANTIC_CLASS_NAMES,
)

COURT_PHYSICAL_INDICES_BY_CLASS = COURT_PHYSICAL_INDICES_BY_SEMANTIC_CLASS
NUM_COURT_SEMANTIC_CLASSES = len(COURT_SEMANTIC_CLASS_NAMES)
SYNTHETIC_GT_COVARIANCE_SIGMA = 0.01
CourtObservationProfile: TypeAlias = Literal[
    "kp14_reference_baseline", "kp7_no_reference", "kp7_reference"
]


def parse_court_observation_profile(value: object) -> CourtObservationProfile:
    """Parse the three incompatible tracking observation architectures."""
    if value not in {
        "kp14_reference_baseline",
        "kp7_no_reference",
        "kp7_reference",
    }:
        raise ValueError(
            "court_observation_profile must be kp14_reference_baseline, "
            "kp7_no_reference, or kp7_reference."
        )
    return value


@dataclass(frozen=True, slots=True)
class CourtPeakBatch:
    """Validated padded semantic peaks at a BLCS/PLCS model boundary."""

    uv: Tensor
    score: Tensor
    covariance: Tensor
    valid: Tensor

    def __post_init__(self) -> None:
        if self.uv.ndim != 6 or self.uv.shape[-1] != 2:
            raise ValueError("court_peak_uv must have shape (B,V,T,7,N,2).")
        if self.uv.shape[3] != NUM_COURT_SEMANTIC_CLASSES:
            raise ValueError("court_peak_uv must use exactly seven semantic classes.")
        if self.uv.shape[4] <= 0:
            raise ValueError("court peak capacity N must be positive.")
        if self.score.shape != self.uv.shape[:-1]:
            raise ValueError("court_peak_score must match court_peak_uv without XY.")
        if self.covariance.shape != (*self.uv.shape[:-1], 2, 2):
            raise ValueError(
                "court_peak_covariance must have shape (B,V,T,7,N,2,2)."
            )
        if self.valid.shape != self.score.shape or self.valid.dtype != torch.bool:
            raise ValueError("court_peak_valid must be boolean and match score.")
        if not self.uv.is_floating_point() or not self.score.is_floating_point():
            raise TypeError("court peak coordinates and scores must be floating.")
        if not self.covariance.is_floating_point():
            raise TypeError("court peak covariance must be floating.")
        values = (self.uv, self.score, self.covariance)
        if len({value.dtype for value in values}) != 1:
            raise ValueError("court peak floating tensors must share one dtype.")
        if len({value.device for value in (*values, self.valid)}) != 1:
            raise ValueError("court peak tensors must share one device.")
        if any(not bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("court peak tensors must contain only finite values.")
        if bool(((self.uv < 0.0) | (self.uv > 1.0)).any()):
            raise ValueError("court_peak_uv must be normalized within [0,1].")
        if bool(((self.score < 0.0) | (self.score > 1.0)).any()):
            raise ValueError("court_peak_score must be within [0,1].")
        covariance = self.covariance
        if not bool(torch.allclose(covariance, covariance.transpose(-1, -2))):
            raise ValueError("court peak covariance matrices must be symmetric.")
        if bool((torch.linalg.eigvalsh(covariance) < 0.0).any()):
            raise ValueError("court peak covariance matrices must be positive semidefinite.")

    def model_fields(self) -> dict[str, Tensor]:
        """Return the four canonical tracking field names without compatibility data."""
        return {
            "court_peak_uv": self.uv,
            "court_peak_score": self.score,
            "court_peak_covariance": self.covariance,
            "court_peak_valid": self.valid,
        }


@runtime_checkable
class CourtPeakPredictionSource(Protocol):
    """Structural Court predictor output accepted by the tracking source adapter."""

    keypoints: Tensor
    scores: Tensor
    valid: Tensor
    covariance: Tensor
    semantic_class_names: tuple[str, ...] | None
    image_size_hw: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class CourtPeakFrame:
    """One explicitly indexed pixel-space KP7 predictor/dataset result."""

    batch_index: int
    view_index: int
    frame_index: int
    keypoints_pixels: Tensor
    scores: Tensor
    covariance_pixels: Tensor
    valid: Tensor
    image_size_hw: tuple[int, int]
    semantic_class_names: tuple[str, ...]

    @classmethod
    def from_prediction(
        cls,
        prediction: CourtPeakPredictionSource,
        *,
        batch_index: int,
        view_index: int,
        frame_index: int,
    ) -> CourtPeakFrame:
        """Index a typed Court predictor result without guessing missing metadata."""
        if not isinstance(prediction, CourtPeakPredictionSource):
            raise TypeError(
                "Court prediction must expose keypoints, scores, valid, covariance, "
                "semantic_class_names, and image_size_hw."
            )
        if prediction.image_size_hw is None:
            raise ValueError("Court prediction is missing image_size_hw metadata.")
        if prediction.semantic_class_names is None:
            raise ValueError("Court prediction is missing its semantic class schema.")
        return cls(
            batch_index=batch_index,
            view_index=view_index,
            frame_index=frame_index,
            keypoints_pixels=prediction.keypoints,
            scores=prediction.scores,
            covariance_pixels=prediction.covariance,
            valid=prediction.valid,
            image_size_hw=prediction.image_size_hw,
            semantic_class_names=prediction.semantic_class_names,
        )

    @classmethod
    def from_dataset_output(
        cls,
        output: Mapping[str, object],
        *,
        batch_index: int,
        view_index: int,
        frame_index: int,
    ) -> CourtPeakFrame:
        """Index an explicit dataset KP7 record; covariance is never synthesized."""
        required = {
            "keypoints",
            "scores",
            "valid",
            "covariance",
            "image_size",
            "semantic_class_names",
        }
        missing = required - set(output)
        if missing:
            raise ValueError(
                f"Court dataset KP7 output is missing fields: {sorted(missing)}."
            )
        tensors: dict[str, Tensor] = {}
        for name in ("keypoints", "scores", "valid", "covariance"):
            value = output[name]
            if not isinstance(value, Tensor):
                raise TypeError(f"Court dataset field {name!r} must be a Tensor.")
            tensors[name] = value
        image_size = output["image_size"]
        if not isinstance(image_size, Tensor) or image_size.shape != (2,):
            raise ValueError("Court dataset image_size must be an integer (H,W) tensor.")
        if image_size.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError("Court dataset image_size must use an integer dtype.")
        raw_names = output["semantic_class_names"]
        if (
            not isinstance(raw_names, Sequence)
            or isinstance(raw_names, (str, bytes))
            or any(not isinstance(name, str) for name in raw_names)
        ):
            raise TypeError("Court dataset semantic_class_names must be a string sequence.")
        return cls(
            batch_index=batch_index,
            view_index=view_index,
            frame_index=frame_index,
            keypoints_pixels=tensors["keypoints"],
            scores=tensors["scores"],
            covariance_pixels=tensors["covariance"],
            valid=tensors["valid"],
            image_size_hw=(int(image_size[0].item()), int(image_size[1].item())),
            semantic_class_names=tuple(raw_names),
        )


def assemble_court_peak_batch(
    frames: Sequence[CourtPeakFrame],
    *,
    expected_shape_bvt: tuple[int, int, int],
) -> CourtPeakBatch:
    """Normalize, align, and pad indexed Court KP7 frames across ``B/V/T``.

    Every expected frame must occur exactly once. Peak capacity is the maximum
    source ``P`` and is padding only; no peak index or court-instance identity is
    introduced.
    """
    batch_size, views, time = expected_shape_bvt
    if min(expected_shape_bvt) <= 0:
        raise ValueError("expected Court peak B/V/T axes must all be positive.")
    expected_count = batch_size * views * time
    if len(frames) != expected_count:
        raise ValueError(
            f"Court peak source has {len(frames)} frames; expected {expected_count} "
            f"for B/V/T={expected_shape_bvt}."
        )

    indexed: dict[tuple[int, int, int], CourtPeakFrame] = {}
    capacity = 0
    dtype: torch.dtype | None = None
    device: torch.device | None = None
    for frame in frames:
        if not isinstance(frame, CourtPeakFrame):
            raise TypeError("court_peak_frames must contain only CourtPeakFrame values.")
        if any(
            type(value) is not int
            for value in (frame.batch_index, frame.view_index, frame.frame_index)
        ):
            raise TypeError("Court peak frame indices must be exact integers.")
        index = (frame.batch_index, frame.view_index, frame.frame_index)
        if not (
            0 <= index[0] < batch_size
            and 0 <= index[1] < views
            and 0 <= index[2] < time
        ):
            raise ValueError(f"Court peak frame index {index} is outside {expected_shape_bvt}.")
        if index in indexed:
            raise ValueError(f"Court peak frame index {index} occurs more than once.")
        if frame.semantic_class_names != COURT_SEMANTIC_CLASS_NAMES:
            raise ValueError(
                "Court peak source semantic class schema must exactly match "
                f"{COURT_SEMANTIC_CLASS_NAMES}."
            )
        if frame.keypoints_pixels.ndim != 3:
            raise ValueError("Court source keypoints must have shape (7,P,2).")
        capacity = max(capacity, int(frame.keypoints_pixels.shape[1]))
        if dtype is None:
            dtype = frame.keypoints_pixels.dtype
            device = frame.keypoints_pixels.device
        elif (
            frame.keypoints_pixels.dtype != dtype
            or frame.keypoints_pixels.device != device
        ):
            raise ValueError("Court peak frames must share one coordinate dtype/device.")
        indexed[index] = frame

    expected_indices = {
        (batch_index, view_index, frame_index)
        for batch_index in range(batch_size)
        for view_index in range(views)
        for frame_index in range(time)
    }
    missing = expected_indices - set(indexed)
    if missing:
        raise ValueError(f"Court peak source is missing indexed frames: {sorted(missing)}.")
    if capacity <= 0 or dtype is None or device is None:
        raise ValueError("Court peak source must declare a positive peak capacity P.")

    uv = torch.zeros(
        batch_size,
        views,
        time,
        NUM_COURT_SEMANTIC_CLASSES,
        capacity,
        2,
        dtype=dtype,
        device=device,
    )
    score = torch.zeros(uv.shape[:-1], dtype=dtype, device=device)
    covariance = torch.zeros(*score.shape, 2, 2, dtype=dtype, device=device)
    valid = torch.zeros(score.shape, dtype=torch.bool, device=device)
    for index in sorted(indexed):
        frame = indexed[index]
        normalized = predicted_peaks_to_normalized(
            frame.keypoints_pixels,
            frame.scores,
            frame.covariance_pixels,
            frame.valid,
            image_size_hw=frame.image_size_hw,
        )
        frame_uv, frame_score, frame_covariance, frame_valid = normalized
        frame_capacity = int(frame_uv.shape[1])
        target = (*index, slice(None), slice(0, frame_capacity))
        uv[*target, :] = frame_uv
        score[target] = frame_score
        covariance[*target, :, :] = frame_covariance
        valid[target] = frame_valid
    return CourtPeakBatch(uv=uv, score=score, covariance=covariance, valid=valid)


def court_peak_batch_from_model_input(
    batch: Mapping[str, object],
    *,
    expected_shape_bvt: tuple[int, int, int],
) -> CourtPeakBatch:
    """Resolve exactly one tensor or indexed-frame KP7 representation."""
    field_names = {
        "court_peak_uv",
        "court_peak_score",
        "court_peak_covariance",
        "court_peak_valid",
    }
    present = field_names & set(batch)
    has_frames = "court_peak_frames" in batch
    if has_frames and present:
        raise ValueError(
            "Provide court_peak_frames or the four court_peak tensors, not both."
        )
    if has_frames:
        value = batch["court_peak_frames"]
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError("court_peak_frames must be a CourtPeakFrame sequence.")
        return assemble_court_peak_batch(
            value,
            expected_shape_bvt=expected_shape_bvt,
        )
    if present != field_names:
        missing = field_names - present
        raise ValueError(f"Court peak model input is missing fields: {sorted(missing)}.")
    tensors: dict[str, Tensor] = {}
    for name in field_names:
        value = batch[name]
        if not isinstance(value, Tensor):
            raise TypeError(f"Court peak model field {name!r} must be a Tensor.")
        tensors[name] = value
    peaks = CourtPeakBatch(
        uv=tensors["court_peak_uv"],
        score=tensors["court_peak_score"],
        covariance=tensors["court_peak_covariance"],
        valid=tensors["court_peak_valid"],
    )
    if peaks.uv.shape[:3] != expected_shape_bvt:
        raise ValueError(
            "Court peak B/V/T axes must match object observations; "
            f"got {tuple(peaks.uv.shape[:3])} and {expected_shape_bvt}."
        )
    return peaks


def ordered_court_to_semantic_peaks(
    court_kp: Tensor,
    court_visible: Tensor,
    *,
    covariance_sigma: float = SYNTHETIC_GT_COVARIANCE_SIGMA,
) -> CourtPeakBatch:
    """Group an authoritative physical KP14 source into seven unordered classes.

    This source-side operation never accepts KP7 and never reconstructs KP14. It is
    used only while materializing synthetic tracking samples whose annotations are
    authored as physical court points.
    """
    if court_kp.ndim != 5 or court_kp.shape[-2:] != (14, 2):
        raise ValueError("physical court source must have shape (B,V,T,14,2).")
    if court_visible.shape != court_kp.shape[:-1] or court_visible.dtype != torch.bool:
        raise ValueError("physical court visibility must be boolean (B,V,T,14).")
    if covariance_sigma <= 0.0:
        raise ValueError("covariance_sigma must be positive.")
    class_indices = torch.tensor(
        COURT_PHYSICAL_INDICES_BY_CLASS,
        dtype=torch.long,
        device=court_kp.device,
    )
    uv = court_kp[..., class_indices, :]
    valid = court_visible[..., class_indices]
    uv = uv.masked_fill(~valid.unsqueeze(-1), 0.0)
    score = valid.to(dtype=court_kp.dtype)
    eye = torch.eye(2, dtype=court_kp.dtype, device=court_kp.device)
    covariance = eye * covariance_sigma**2
    covariance = covariance.view(*(1 for _ in range(5)), 2, 2).expand(
        *uv.shape[:-1], 2, 2
    )
    covariance = covariance.masked_fill(~valid.unsqueeze(-1).unsqueeze(-1), 0.0)
    return CourtPeakBatch(uv=uv, score=score, covariance=covariance, valid=valid)


def predicted_peaks_to_normalized(
    keypoints_pixels: Tensor,
    scores: Tensor,
    covariance_pixels: Tensor,
    valid: Tensor,
    *,
    image_size_hw: tuple[int, int],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Normalize a Court predictor's explicit KP7 peak and covariance output."""
    if keypoints_pixels.ndim != 3 or keypoints_pixels.shape[-1] != 2:
        raise ValueError("predicted keypoints must have shape (7,N,2).")
    if keypoints_pixels.shape[0] != NUM_COURT_SEMANTIC_CLASSES:
        raise ValueError("predicted keypoints must use the seven-class schema.")
    if scores.shape != keypoints_pixels.shape[:-1]:
        raise ValueError("predicted scores must match keypoints without XY.")
    if valid.shape != scores.shape or valid.dtype != torch.bool:
        raise ValueError("predicted validity must be boolean and match scores.")
    if covariance_pixels.shape != (*scores.shape, 2, 2):
        raise ValueError("predicted covariance must have shape (7,N,2,2).")
    if not (
        keypoints_pixels.is_floating_point()
        and scores.is_floating_point()
        and covariance_pixels.is_floating_point()
    ):
        raise TypeError("predicted keypoints, scores, and covariance must be floating.")
    if len(
        {
            keypoints_pixels.device,
            scores.device,
            covariance_pixels.device,
            valid.device,
        }
    ) != 1:
        raise ValueError("predicted peak tensors must share one device.")
    if scores.dtype != keypoints_pixels.dtype or covariance_pixels.dtype != keypoints_pixels.dtype:
        raise ValueError("predicted peak floating tensors must share one dtype.")
    if any(
        not bool(torch.isfinite(value).all())
        for value in (keypoints_pixels, scores, covariance_pixels)
    ):
        raise ValueError("predicted peak tensors must contain only finite values.")
    if bool(((scores < 0.0) | (scores > 1.0)).any()):
        raise ValueError("predicted peak scores must be within [0,1].")
    if not bool(torch.allclose(covariance_pixels, covariance_pixels.transpose(-1, -2))):
        raise ValueError("predicted covariance matrices must be symmetric.")
    if bool((torch.linalg.eigvalsh(covariance_pixels) < 0.0).any()):
        raise ValueError("predicted covariance matrices must be positive semidefinite.")
    if any(type(value) is not int for value in image_size_hw):
        raise TypeError("image_size_hw must contain two integers.")
    height, width = image_size_hw
    if height <= 1 or width <= 1:
        raise ValueError("image_size_hw must contain dimensions greater than one.")
    scale = keypoints_pixels.new_tensor((float(width - 1), float(height - 1)))
    uv = keypoints_pixels / scale
    covariance = (
        covariance_pixels
        / scale.view(1, 1, 2, 1)
        / scale.view(1, 1, 1, 2)
    )
    uv = uv.masked_fill(~valid.unsqueeze(-1), 0.0)
    score = scores.masked_fill(~valid, 0.0)
    covariance = covariance.masked_fill(
        ~valid.unsqueeze(-1).unsqueeze(-1), 0.0
    )
    return uv, score, covariance, valid


def reference_view_mask(reference_view_index: Tensor, view_mask: Tensor) -> Tensor:
    """Validate exactly one non-padded reference view and return its mask."""
    if view_mask.ndim != 2 or view_mask.dtype != torch.bool:
        raise ValueError("view_mask must be boolean (B,V).")
    if reference_view_index.shape != (view_mask.shape[0],):
        raise ValueError("reference_view_index must have shape (B,).")
    if reference_view_index.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError("reference_view_index must use an integer dtype.")
    if bool(
        ((reference_view_index < 0) | (reference_view_index >= view_mask.shape[1])).any()
    ):
        raise ValueError("reference_view_index is outside the padded view axis.")
    result = torch.zeros_like(view_mask)
    result.scatter_(1, reference_view_index.long().unsqueeze(1), True)
    if not bool((result & view_mask).sum(dim=1).eq(1).all()):
        raise ValueError("the reference view must be inside view_mask.")
    return result


def reference_context_validity(
    detection_valid: Tensor,
    *,
    frame_mask: Tensor,
    view_mask: Tensor,
    reference_mask: Tensor,
    mask_invisible_observations: bool,
) -> Tensor:
    """Keep D-slot 0 only when a reference camera-time has no detection."""
    if detection_valid.ndim != 4 or detection_valid.dtype != torch.bool:
        raise ValueError("detection_valid must be boolean (B,V,T,D).")
    batch_size, views, frames, detections = detection_valid.shape
    if detections <= 0:
        raise ValueError("the declared D axis must contain at least one slot.")
    if frame_mask.shape != (batch_size, frames) or frame_mask.dtype != torch.bool:
        raise ValueError("frame_mask must be boolean (B,T).")
    if view_mask.shape != (batch_size, views) or view_mask.dtype != torch.bool:
        raise ValueError("view_mask must be boolean (B,V).")
    if reference_mask.shape != view_mask.shape or reference_mask.dtype != torch.bool:
        raise ValueError("reference_mask must be boolean (B,V).")
    if not bool(reference_mask.sum(dim=1).eq(1).all()) or bool(
        (reference_mask & ~view_mask).any()
    ):
        raise ValueError("reference_mask must select exactly one valid view.")
    context = (
        view_mask[:, :, None, None] & frame_mask[:, None, :, None]
    ).expand_as(detection_valid)
    valid = context & detection_valid if mask_invisible_observations else context
    missing_reference = (
        reference_mask[:, :, None]
        & frame_mask[:, None, :]
        & ~detection_valid.any(dim=-1)
    )
    valid = valid.clone()
    valid[..., 0] |= missing_reference
    return valid.permute(0, 2, 1, 3)


__all__ = [
    "COURT_PHYSICAL_INDICES_BY_CLASS",
    "COURT_SEMANTIC_CLASS_NAMES",
    "CourtObservationProfile",
    "CourtPeakBatch",
    "CourtPeakFrame",
    "CourtPeakPredictionSource",
    "NUM_COURT_SEMANTIC_CLASSES",
    "SYNTHETIC_GT_COVARIANCE_SIGMA",
    "assemble_court_peak_batch",
    "court_peak_batch_from_model_input",
    "ordered_court_to_semantic_peaks",
    "parse_court_observation_profile",
    "predicted_peaks_to_normalized",
    "reference_context_validity",
    "reference_view_mask",
]
