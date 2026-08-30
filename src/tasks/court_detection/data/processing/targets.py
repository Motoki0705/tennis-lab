"""Source-neutral Court target builders."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from src.tasks.court_detection.configuration import CourtTargetConfig
from src.tasks.court_detection.data.contracts import (
    CourtDenseTargetKind,
    CourtInputCapability,
    CourtInputSpec,
    CourtRawSample,
    CourtSampleRecord,
    CourtTargetSpec,
    CourtTransformedSample,
)
from src.tasks.court_detection.data.target_generation.store import (
    validate_derived_target,
)
from src.utils.data.heatmaps import generate_gaussian_heatmaps


class CourtTargetBuilder(Protocol):
    """A target layer that never inspects a concrete source implementation."""

    @property
    def spec(self) -> CourtTargetSpec: ...

    @property
    def required_capabilities(self) -> frozenset[CourtInputCapability]: ...

    def preflight(self, records: tuple[CourtSampleRecord, ...]) -> None: ...

    def load_dense(
        self, raw: CourtRawSample
    ) -> Mapping[CourtDenseTargetKind, Tensor]: ...

    def build(self, sample: CourtTransformedSample) -> object: ...


class KeypointTargetBuilder:
    def __init__(self, input_spec: CourtInputSpec, *, sigma_ratio: float) -> None:
        if input_spec.keypoint_schema is None or not input_spec.keypoint_channel_names:
            raise ValueError("Keypoint target requires an input keypoint schema.")
        self.sigma_ratio = sigma_ratio
        self._spec = CourtTargetSpec(
            kind="kp",
            schema=f"{input_spec.keypoint_schema}:gaussian_max_v1",
            output_channels=len(input_spec.keypoint_channel_names),
            channel_names=input_spec.keypoint_channel_names,
            target_dtype=torch.float32,
            precomputed=False,
        )

    @property
    def spec(self) -> CourtTargetSpec:
        return self._spec

    @property
    def required_capabilities(self) -> frozenset[CourtInputCapability]:
        return frozenset({CourtInputCapability.KEYPOINT_CHANNELS})

    def preflight(self, records: tuple[CourtSampleRecord, ...]) -> None:
        if not records:
            raise ValueError("Keypoint target requires a non-empty split.")

    def load_dense(self, raw: CourtRawSample) -> Mapping[CourtDenseTargetKind, Tensor]:
        _ = raw
        return {}

    def build(self, sample: CourtTransformedSample) -> object:
        channels = sample.keypoint_channels
        if channels is None:
            raise ValueError("Keypoint target received no keypoint channels.")
        height, width = (int(value) for value in sample.image_size.tolist())
        scale = channels.points_xy.new_tensor(
            [
                float(max(width - 1, 1)),
                float(max(height - 1, 1)),
            ]
        )
        normalized = (channels.points_xy / scale).to(dtype=self.spec.target_dtype)
        heatmap = generate_gaussian_heatmaps(
            (height, width),
            normalized,
            self.sigma_ratio,
            visibility=channels.point_visible,
            point_reduction="max",
        )
        if heatmap.shape != (self.spec.output_channels, height, width):
            raise ValueError("Keypoint heatmap shape disagrees with target spec.")
        return {
            "heatmap": heatmap,
            "points_xy": normalized,
            "point_visible": channels.point_visible,
            "physical_indices": channels.physical_indices,
        }


class _PrecomputedDenseTargetBuilder:
    kind: CourtDenseTargetKind
    capability: CourtInputCapability

    def __init__(self, spec: CourtTargetSpec, *, input_spec: CourtInputSpec) -> None:
        self._spec = spec
        self._input_spec = input_spec

    @property
    def spec(self) -> CourtTargetSpec:
        return self._spec

    @property
    def required_capabilities(self) -> frozenset[CourtInputCapability]:
        return frozenset({self.capability})

    def preflight(self, records: tuple[CourtSampleRecord, ...]) -> None:
        if not records:
            raise ValueError(f"{self.kind} target requires a non-empty split.")
        for record in records:
            validate_derived_target(
                record,
                input_spec=self._input_spec,
                target_kind=self.kind,
                target_schema=self.spec.schema,
            )

    def load_dense(self, raw: CourtRawSample) -> Mapping[CourtDenseTargetKind, Tensor]:
        try:
            path = raw.dense_target_refs[self.kind]
        except KeyError as error:
            raise FileNotFoundError(
                f"Court sample {raw.sample_id!r} has no {self.kind} target reference."
            ) from error
        if not path.is_file():
            raise FileNotFoundError(f"Precomputed Court target is missing: {path}")
        with Image.open(path) as handle:
            array = np.asarray(handle.convert("L"), dtype=np.uint8)
        if array.shape != (raw.image.height, raw.image.width):
            raise ValueError(
                f"Precomputed Court {self.kind} target resolution disagrees with RGB."
            )
        return {self.kind: self._decode(array)}

    def _decode(self, array: np.ndarray) -> Tensor:
        raise NotImplementedError


class SegmentationTargetBuilder(_PrecomputedDenseTargetBuilder):
    kind: CourtDenseTargetKind = "seg"
    capability = CourtInputCapability.SEGMENTATION_REFERENCE

    def __init__(self, *, target_schema: str, input_spec: CourtInputSpec) -> None:
        super().__init__(
            CourtTargetSpec(
                kind="seg",
                schema=target_schema,
                output_channels=7,
                channel_names=(
                    "background",
                    "service_left",
                    "service_right",
                    "back_left",
                    "back_right",
                    "doubles_left",
                    "doubles_right",
                ),
                target_dtype=torch.long,
                precomputed=True,
            ),
            input_spec=input_spec,
        )

    def _decode(self, array: np.ndarray) -> Tensor:
        if int(array.max(initial=0)) > 6:
            raise ValueError("Court segmentation labels must be in [0,6].")
        return torch.from_numpy(np.ascontiguousarray(array).copy()).long()

    def build(self, sample: CourtTransformedSample) -> object:
        mask = sample.dense_targets["seg"].long()
        if sample.horizontal_flipped:
            source = mask.clone()
            for left, right in ((1, 2), (3, 4), (5, 6)):
                mask[source == left] = right
                mask[source == right] = left
        return mask


class LineTargetBuilder(_PrecomputedDenseTargetBuilder):
    kind: CourtDenseTargetKind = "line"
    capability = CourtInputCapability.LINE_REFERENCE

    def __init__(self, *, target_schema: str, input_spec: CourtInputSpec) -> None:
        super().__init__(
            CourtTargetSpec(
                kind="line",
                schema=target_schema,
                output_channels=1,
                channel_names=("court_line",),
                target_dtype=torch.float32,
                precomputed=True,
            ),
            input_spec=input_spec,
        )

    def _decode(self, array: np.ndarray) -> Tensor:
        unique = set(np.unique(array).tolist())
        if not unique.issubset({0, 1, 255}):
            raise ValueError("Court line targets must be binary.")
        return torch.from_numpy((array > 0).astype(np.float32)).unsqueeze(0)

    def build(self, sample: CourtTransformedSample) -> object:
        target = sample.dense_targets["line"].float()
        if target.ndim != 3 or target.shape[0] != 1:
            raise ValueError("Court line target must have shape [1,H,W].")
        return target


def build_target_builder(
    config: CourtTargetConfig,
    *,
    input_spec: CourtInputSpec,
) -> CourtTargetBuilder:
    if config.kind == "kp":
        if config.sigma_ratio is None:
            raise ValueError("KP target config requires sigma_ratio.")
        builder: CourtTargetBuilder = KeypointTargetBuilder(
            input_spec, sigma_ratio=config.sigma_ratio
        )
    elif config.kind == "seg":
        if config.target_schema is None:
            raise ValueError("Segmentation target config requires target_schema.")
        builder = SegmentationTargetBuilder(
            target_schema=config.target_schema,
            input_spec=input_spec,
        )
    elif config.kind == "line":
        if config.target_schema is None:
            raise ValueError("Line target config requires target_schema.")
        builder = LineTargetBuilder(
            target_schema=config.target_schema,
            input_spec=input_spec,
        )
    else:  # pragma: no cover - typed configuration rejects this
        raise ValueError(f"Unsupported Court target kind: {config.kind!r}.")
    missing = builder.required_capabilities - input_spec.capabilities
    if missing:
        raise ValueError(
            f"Input {input_spec.source_kind!r} lacks target capabilities "
            f"{sorted(item.value for item in missing)}."
        )
    return builder


__all__ = [
    "CourtTargetBuilder",
    "KeypointTargetBuilder",
    "LineTargetBuilder",
    "SegmentationTargetBuilder",
    "build_target_builder",
]
