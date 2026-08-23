"""Strict tensor contracts for the Court query-encoder architecture."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import Tensor

from src.tasks.court_detection.data.contracts import CourtTargetKind
from src.tasks.court_detection.geometry.pose import POSE10D_RAW_ORDER

COURT_POSE10D_RAW_ORDER = POSE10D_RAW_ORDER


def _positive_hw(value: tuple[int, int], *, name: str) -> tuple[int, int]:
    if (
        len(value) != 2
        or any(type(item) is not int or item <= 0 for item in value)
    ):
        raise ValueError(f"{name} must contain exactly two positive integers.")
    return value


def _require_floating_finite(value: Tensor, *, name: str) -> None:
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values.")


@dataclass(frozen=True, slots=True)
class PatchTokenBatch:
    """Patch-only DINO output and its explicit padded image geometry."""

    tokens: Tensor
    original_hw: tuple[int, int]
    padded_hw: tuple[int, int]
    padding_hw: tuple[int, int]
    grid_hw: tuple[int, int]
    patch_size: int

    def __post_init__(self) -> None:
        if self.tokens.ndim != 3 or any(size <= 0 for size in self.tokens.shape):
            raise ValueError("Patch tokens must have shape (B,N,C) with positive axes.")
        _require_floating_finite(self.tokens, name="Patch tokens")
        original_h, original_w = _positive_hw(self.original_hw, name="original_hw")
        padded_h, padded_w = _positive_hw(self.padded_hw, name="padded_hw")
        grid_h, grid_w = _positive_hw(self.grid_hw, name="grid_hw")
        if type(self.patch_size) is not int or self.patch_size <= 0:
            raise ValueError("patch_size must be a positive integer.")
        if (
            len(self.padding_hw) != 2
            or any(type(item) is not int or item < 0 for item in self.padding_hw)
        ):
            raise ValueError("padding_hw must contain two non-negative integers.")
        pad_h, pad_w = self.padding_hw
        if (padded_h, padded_w) != (original_h + pad_h, original_w + pad_w):
            raise ValueError("padded_hw must exactly equal original_hw + padding_hw.")
        if pad_h >= self.patch_size or pad_w >= self.patch_size:
            raise ValueError("Patch padding must be the minimal next-patch padding.")
        if (grid_h * self.patch_size, grid_w * self.patch_size) != self.padded_hw:
            raise ValueError("Patch grid, padded shape, and patch size disagree.")
        if self.tokens.shape[1] != grid_h * grid_w:
            raise ValueError(
                "Patch token count must exactly equal grid height times grid width; "
                "CLS/register tokens are excluded."
            )

    @property
    def batch_size(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def embed_dim(self) -> int:
        return int(self.tokens.shape[2])


@dataclass(frozen=True, slots=True)
class CourtEncoderTap:
    """Patch-only output captured after one declared task-encoder layer."""

    layer_index: int
    patch_tokens: Tensor
    grid_hw: tuple[int, int]

    def __post_init__(self) -> None:
        if type(self.layer_index) is not int or self.layer_index < 0:
            raise ValueError("Encoder tap layer_index must be non-negative.")
        grid_h, grid_w = _positive_hw(self.grid_hw, name="tap grid_hw")
        if self.patch_tokens.ndim != 3:
            raise ValueError("Encoder tap patch_tokens must have shape (B,N,C).")
        if self.patch_tokens.shape[1] != grid_h * grid_w:
            raise ValueError("Encoder tap token count and grid disagree.")
        _require_floating_finite(
            self.patch_tokens,
            name=f"Encoder tap {self.layer_index}",
        )


@dataclass(frozen=True, slots=True)
class CourtTaskEncoderOutput:
    pose_query: Tensor
    taps: tuple[CourtEncoderTap, ...]

    def __post_init__(self) -> None:
        if self.pose_query.ndim != 2 or any(
            size <= 0 for size in self.pose_query.shape
        ):
            raise ValueError("Final pose query must have shape (B,C).")
        _require_floating_finite(self.pose_query, name="Final pose query")
        if not self.taps:
            raise ValueError("Task encoder must expose at least one patch tap.")
        indices = tuple(tap.layer_index for tap in self.taps)
        if len(set(indices)) != len(indices):
            raise ValueError("Task encoder tap indices must be unique.")
        for tap in self.taps:
            if tap.patch_tokens.shape[0] != self.pose_query.shape[0]:
                raise ValueError("Pose query and encoder taps must share batch size.")
            if tap.patch_tokens.shape[2] != self.pose_query.shape[1]:
                raise ValueError("Pose query and encoder taps must share hidden width.")


@dataclass(frozen=True, slots=True)
class CourtPose10DRaw:
    """Raw camera head in the immutable camera-pose scalar order."""

    values: Tensor

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or self.values.shape[1] != len(
            COURT_POSE10D_RAW_ORDER
        ):
            raise ValueError("Raw Court pose must have exact shape (B,10).")
        if self.values.shape[0] <= 0:
            raise ValueError("Raw Court pose batch size must be positive.")
        _require_floating_finite(self.values, name="Raw Court pose10d")


@dataclass(frozen=True, slots=True)
class CourtQueryRawOutput:
    """Raw query-model output before pose decoding or training losses."""

    pose: CourtPose10DRaw
    dense_logits: Mapping[CourtTargetKind, Tensor]

    def __post_init__(self) -> None:
        logits = dict(self.dense_logits)
        if not logits:
            raise ValueError("Query model requires a non-empty dense output mapping.")
        for kind, value in logits.items():
            if kind not in {"kp", "seg", "line"}:
                raise ValueError(f"Unknown Court dense output kind: {kind!r}.")
            if value.ndim != 4 or value.shape[0] != self.pose.values.shape[0]:
                raise ValueError(
                    f"Court {kind} logits must have shape (B,C,H,W) matching pose."
                )
            _require_floating_finite(value, name=f"Court {kind} logits")
        object.__setattr__(self, "dense_logits", MappingProxyType(logits))


__all__ = [
    "COURT_POSE10D_RAW_ORDER",
    "CourtEncoderTap",
    "CourtPose10DRaw",
    "CourtQueryRawOutput",
    "CourtTaskEncoderOutput",
    "PatchTokenBatch",
]
