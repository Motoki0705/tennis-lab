"""Source-neutral data contracts for composable court detection training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Literal, TypeAlias

import torch
from PIL import Image
from torch import Tensor

CourtTargetKind: TypeAlias = Literal["kp", "seg", "line"]
CourtDenseTargetKind: TypeAlias = Literal["seg", "line"]
CourtSourceKind: TypeAlias = Literal["tennis_court_detector", "synthetic_court"]
CourtSourceSplit: TypeAlias = Literal["train", "val", "test"]


class CourtInputCapability(StrEnum):
    """Canonical payloads that an input layer can supply."""

    KEYPOINT_CHANNELS = "keypoint_channels"
    COURT_INSTANCES = "court_instances"
    SEGMENTATION_REFERENCE = "segmentation_reference"
    LINE_REFERENCE = "line_reference"


@dataclass(frozen=True, slots=True)
class CourtTargetSpec:
    """One resolved model/target contract."""

    kind: CourtTargetKind
    schema: str
    output_channels: int
    channel_names: tuple[str, ...]
    target_dtype: torch.dtype
    precomputed: bool

    def __post_init__(self) -> None:
        if self.kind not in {"kp", "seg", "line"}:
            raise ValueError(f"Unsupported Court target kind: {self.kind!r}.")
        if not self.schema or self.schema != self.schema.strip():
            raise ValueError("Court target schema must be a non-empty trimmed string.")
        if self.output_channels <= 0:
            raise ValueError("Court target output_channels must be positive.")
        if len(self.channel_names) != self.output_channels:
            raise ValueError("Court target channel_names must match output_channels.")
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("Court target channel_names must be unique.")


@dataclass(frozen=True, slots=True)
class CourtTargetBundleSpec:
    """Ordered non-empty collection of target/head contracts."""

    targets: Mapping[CourtTargetKind, CourtTargetSpec]

    def __post_init__(self) -> None:
        copied = dict(self.targets)
        if not copied:
            raise ValueError("Court target bundle must contain at least one target.")
        for kind, spec in copied.items():
            if kind != spec.kind:
                raise ValueError(
                    f"Court target bundle key {kind!r} disagrees with spec {spec.kind!r}."
                )
        object.__setattr__(self, "targets", MappingProxyType(copied))

    @property
    def kinds(self) -> tuple[CourtTargetKind, ...]:
        return tuple(self.targets)

    @property
    def head_channels(self) -> Mapping[CourtTargetKind, int]:
        return MappingProxyType(
            {kind: spec.output_channels for kind, spec in self.targets.items()}
        )

    def require_single(self, kind: CourtTargetKind | None = None) -> CourtTargetSpec:
        if len(self.targets) != 1:
            raise ValueError("This operation requires a single-target Court bundle.")
        spec = next(iter(self.targets.values()))
        if kind is not None and spec.kind != kind:
            raise ValueError(
                f"Court bundle contains {spec.kind!r}, not required target {kind!r}."
            )
        return spec


@dataclass(frozen=True, slots=True)
class CourtInputSpec:
    """Stable source schema and capabilities resolved before workers start."""

    source_kind: CourtSourceKind
    source_schema: str
    capabilities: frozenset[CourtInputCapability]
    keypoint_schema: str | None = None
    keypoint_channel_names: tuple[str, ...] = ()
    keypoint_flip_permutation: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.source_kind not in {"tennis_court_detector", "synthetic_court"}:
            raise ValueError(f"Unsupported Court source kind: {self.source_kind!r}.")
        if not self.source_schema:
            raise ValueError("Court source schema must be non-empty.")
        has_keypoints = CourtInputCapability.KEYPOINT_CHANNELS in self.capabilities
        if has_keypoints != (self.keypoint_schema is not None):
            raise ValueError("Court keypoint capability and schema must be declared together.")
        if has_keypoints:
            channel_count = len(self.keypoint_channel_names)
            if channel_count == 0:
                raise ValueError("Court keypoint schemas require channel names.")
            if tuple(sorted(self.keypoint_flip_permutation)) != tuple(
                range(channel_count)
            ):
                raise ValueError(
                    "Court keypoint horizontal_flip_permutation must be a bijection."
                )


@dataclass(frozen=True, slots=True)
class CourtSampleRecord:
    """A validated source record with authoritative paths and derived refs."""

    sample_id: str
    split: CourtSourceSplit
    image_path: Path
    annotation_path: Path
    derived_key: str
    dense_target_refs: Mapping[CourtDenseTargetKind, Path]
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.sample_id or self.sample_id != self.sample_id.strip():
            raise ValueError("Court sample_id must be non-empty and trimmed.")
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported Court split: {self.split!r}.")
        if not self.derived_key:
            raise ValueError("Court derived target key must be non-empty.")
        object.__setattr__(
            self, "dense_target_refs", MappingProxyType(dict(self.dense_target_refs))
        )
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


@dataclass(frozen=True, slots=True)
class CourtInstance2D:
    """One court instance represented by physical point identity and image geometry."""

    court_instance_id: str
    physical_indices: Tensor  # [N], int64
    points_xy: Tensor  # [N, 2], float32 pixel coordinates
    point_in_front: Tensor  # [N], bool; positive camera-depth half-plane
    point_visible: Tensor  # [N], bool; geometry visibility, not KP supervision

    def __post_init__(self) -> None:
        if not self.court_instance_id:
            raise ValueError("Court instance ID must be non-empty.")
        count = self.physical_indices.numel()
        if self.physical_indices.shape != (count,) or self.physical_indices.dtype != torch.long:
            raise ValueError("physical_indices must be an int64 vector.")
        if self.points_xy.shape != (count, 2) or not self.points_xy.is_floating_point():
            raise ValueError("Court instance points_xy must have shape (N, 2) and float dtype.")
        if self.point_in_front.shape != (count,) or self.point_in_front.dtype != torch.bool:
            raise ValueError("Court instance point_in_front must be a boolean vector.")
        if self.point_visible.shape != (count,) or self.point_visible.dtype != torch.bool:
            raise ValueError("Court instance point_visible must be a boolean vector.")
        if bool(torch.any(self.point_visible & ~self.point_in_front)):
            raise ValueError("Visible Court instance points must be in front of the camera.")
        if not bool(torch.isfinite(self.points_xy).all()):
            raise ValueError("Court instance points_xy must be finite.")
        if count == 0 or len(set(self.physical_indices.tolist())) != count:
            raise ValueError("Court instance physical point IDs must be non-empty and unique.")


@dataclass(frozen=True, slots=True)
class CourtKeypointChannels:
    """Single- or multi-peak keypoint channels shared by both source schemas."""

    channel_names: tuple[str, ...]
    points_xy: Tensor  # [C, P, 2], pixels
    point_visible: Tensor  # [C, P], bool
    physical_indices: Tensor  # [C, P], int64; -1 means padding
    horizontal_flip_permutation: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.points_xy.ndim != 3 or self.points_xy.shape[-1] != 2:
            raise ValueError("Court keypoint points_xy must have shape (C, P, 2).")
        channels, points, _ = self.points_xy.shape
        if channels == 0 or points == 0 or len(self.channel_names) != channels:
            raise ValueError("Court keypoint channels and point capacity must be non-empty.")
        if not self.points_xy.is_floating_point() or not bool(
            torch.isfinite(self.points_xy).all()
        ):
            raise ValueError("Court keypoint points_xy must be finite floating values.")
        if self.point_visible.shape != (channels, points) or self.point_visible.dtype != torch.bool:
            raise ValueError("Court keypoint point_visible must have shape (C, P) and bool dtype.")
        if self.physical_indices.shape != (channels, points) or self.physical_indices.dtype != torch.long:
            raise ValueError("Court keypoint physical_indices must have shape (C, P) and int64 dtype.")
        if tuple(sorted(self.horizontal_flip_permutation)) != tuple(range(channels)):
            raise ValueError("Court keypoint flip permutation must be a channel bijection.")
        if bool(torch.any((self.physical_indices < 0) & self.point_visible)):
            raise ValueError("Padded Court keypoint slots cannot be visible.")


@dataclass(frozen=True, slots=True)
class CourtSampleMetadata:
    """Diagnostic provenance that target builders never branch on."""

    source_kind: CourtSourceKind
    source_schema: str
    source_sample_id: str
    scene_id: str | None
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def to_dict(self) -> dict[str, object]:
        return {
            "source_kind": self.source_kind,
            "source_schema": self.source_schema,
            "source_sample_id": self.source_sample_id,
            "scene_id": self.scene_id,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True, slots=True)
class CourtRawSample:
    """Source-neutral payload loaded by one input implementation."""

    sample_id: str
    image: Image.Image
    keypoint_channels: CourtKeypointChannels | None
    court_instances: tuple[CourtInstance2D, ...]
    dense_target_refs: Mapping[CourtDenseTargetKind, Path]
    metadata: CourtSampleMetadata

    def __post_init__(self) -> None:
        if self.image.mode != "RGB":
            raise ValueError("Court raw images must be RGB PIL images.")
        object.__setattr__(
            self, "dense_target_refs", MappingProxyType(dict(self.dense_target_refs))
        )


@dataclass(frozen=True, slots=True)
class CourtTransformedSample:
    """One shared-geometry result consumed by every selected target builder."""

    sample_id: str
    image_tensor: Tensor  # [3,H,W], ImageNet-normalized
    image_size: Tensor  # [2] = H,W before batch padding
    keypoint_channels: CourtKeypointChannels | None
    court_instances: tuple[CourtInstance2D, ...]
    dense_targets: Mapping[CourtDenseTargetKind, Tensor]
    horizontal_flipped: bool
    metadata: CourtSampleMetadata

    def __post_init__(self) -> None:
        if self.image_tensor.ndim != 3 or self.image_tensor.shape[0] != 3:
            raise ValueError("Court transformed image must have shape (3,H,W).")
        if self.image_size.shape != (2,) or self.image_size.dtype != torch.long:
            raise ValueError("Court transformed image_size must be int64 [H,W].")
        object.__setattr__(
            self, "dense_targets", MappingProxyType(dict(self.dense_targets))
        )


__all__ = [
    "CourtDenseTargetKind",
    "CourtInputCapability",
    "CourtInputSpec",
    "CourtInstance2D",
    "CourtKeypointChannels",
    "CourtRawSample",
    "CourtSampleMetadata",
    "CourtSampleRecord",
    "CourtSourceKind",
    "CourtSourceSplit",
    "CourtTargetBundleSpec",
    "CourtTargetKind",
    "CourtTargetSpec",
    "CourtTransformedSample",
]
