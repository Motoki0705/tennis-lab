"""Composable Court source -> shared geometry -> target bundle pipeline."""

from __future__ import annotations

from types import MappingProxyType

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


class CourtProcessingPipeline:
    """Process a sample without any source-target cross branching."""

    def __init__(
        self,
        *,
        input_layer: CourtInput,
        geometry: CourtProcessingGeometry,
        target_builders: tuple[CourtTargetBuilder, ...],
    ) -> None:
        if not target_builders:
            raise ValueError("Court processing requires at least one target builder.")
        kinds = tuple(builder.spec.kind for builder in target_builders)
        if len(set(kinds)) != len(kinds):
            raise ValueError("Court processing target builders must be unique.")
        self.input_layer = input_layer
        self.geometry = geometry
        self.target_builders = target_builders
        self.target_bundle_spec = CourtTargetBundleSpec(
            {builder.spec.kind: builder.spec for builder in target_builders}
        )

    def preflight(self, records: tuple[CourtSampleRecord, ...]) -> None:
        """Validate every selected target before DataLoader workers start."""
        if not records:
            raise ValueError("Court processing split must be non-empty.")
        for builder in self.target_builders:
            builder.preflight(records)

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
        return {
            "image": transformed.image_tensor,
            "targets": MappingProxyType(targets),
            "image_size": transformed.image_size,
            "sample_id": transformed.sample_id,
            "metadata": transformed.metadata.to_dict(),
        }


__all__ = ["CourtProcessingPipeline"]
