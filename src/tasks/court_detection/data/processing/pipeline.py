"""Composable Court source -> shared geometry -> target bundle pipeline."""

from __future__ import annotations

from types import MappingProxyType

import torch
from torch import Tensor

from src.tasks.court_detection.data.contracts import (
    CourtDenseTargetKind,
    CourtSampleRecord,
    CourtTargetBundleSpec,
    CourtTargetKind,
)
from src.tasks.court_detection.data.inputs.contract import CourtInput
from src.tasks.court_detection.data.processing.geometry import CourtProcessingGeometry
from src.tasks.court_detection.data.processing.targets import CourtTargetBuilder
from src.tasks.court_detection.geometry.pose import (
    build_pose_target,
    validate_projection_round_trip,
)


class CourtProcessingPipeline:
    """Process a sample without any source-target cross branching."""

    def __init__(
        self,
        *,
        input_layer: CourtInput,
        geometry: CourtProcessingGeometry,
        target_builders: tuple[CourtTargetBuilder, ...],
        require_pose: bool = False,
    ) -> None:
        if not target_builders:
            raise ValueError("Court processing requires at least one target builder.")
        kinds = tuple(builder.spec.kind for builder in target_builders)
        if len(set(kinds)) != len(kinds):
            raise ValueError("Court processing target builders must be unique.")
        self.input_layer = input_layer
        self.geometry = geometry
        self.target_builders = target_builders
        self.require_pose = require_pose
        self.target_bundle_spec = CourtTargetBundleSpec(
            {builder.spec.kind: builder.spec for builder in target_builders}
        )

    def preflight(self, records: tuple[CourtSampleRecord, ...]) -> None:
        """Validate every selected target before DataLoader workers start."""
        if not records:
            raise ValueError("Court processing split must be non-empty.")
        for builder in self.target_builders:
            builder.preflight(records)
        if self.require_pose:
            for record in records:
                raw = self.input_layer.load(record)
                if raw.pose_authority is None:
                    raise ValueError(
                        f"Court sample {record.sample_id!r} has no V3 pose authority."
                    )
                channels = raw.keypoint_channels
                if channels is None or channels.points_xy.shape != (14, 1, 2):
                    raise ValueError(
                        "Court query requires target-court singleton KP14 with P==1."
                    )
                pose_target = build_pose_target(raw.pose_authority)
                if not torch.equal(
                    channels.physical_indices[:, 0],
                    pose_target.semantic_to_physical,
                ):
                    raise ValueError(
                        "Court query KP14 order disagrees with V3 pose authority."
                    )
                validate_projection_round_trip(
                    pose_target,
                    channels.points_xy[:, 0],
                )

    def process(self, record: CourtSampleRecord) -> dict[str, object]:
        raw = self.input_layer.load(record)
        if raw.sample_id != record.sample_id:
            raise ValueError("Court input changed the stable sample ID.")
        dense: dict[CourtDenseTargetKind, Tensor] = {}
        for builder in self.target_builders:
            for kind, value in builder.load_dense(raw).items():
                if kind in dense:
                    raise ValueError(f"Duplicate prepared dense target {kind!r}.")
                dense[kind] = value
        plan = self.geometry.sample(raw)
        transformed = self.geometry.apply(raw, dense_targets=dense, plan=plan)
        targets: dict[CourtTargetKind, object] = {
            builder.spec.kind: builder.build(transformed)
            for builder in self.target_builders
        }
        if tuple(targets) != self.target_bundle_spec.kinds:
            raise ValueError("Court target order disagrees with the resolved bundle.")
        output: dict[str, object] = {
            "image": transformed.image_tensor,
            "targets": MappingProxyType(targets),
            "image_size": transformed.image_size,
            "sample_id": transformed.sample_id,
            "metadata": transformed.metadata.to_dict(),
        }
        if self.require_pose:
            if transformed.pose_target is None:  # pragma: no cover - geometry owns it
                raise ValueError("Court query processing produced no pose target.")
            output["pose_target"] = transformed.pose_target.to_mapping()
        return output


__all__ = ["CourtProcessingPipeline"]
