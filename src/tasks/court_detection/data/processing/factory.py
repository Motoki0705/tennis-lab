"""Composition root for the orthogonal Court input and target axes."""

from __future__ import annotations

from src.tasks.court_detection.configuration import CourtDataConfig
from src.tasks.court_detection.data.inputs.factory import build_court_input
from src.tasks.court_detection.data.processing.geometry import CourtProcessingGeometry
from src.tasks.court_detection.data.processing.pipeline import CourtProcessingPipeline
from src.tasks.court_detection.data.processing.targets import build_target_builder
from src.tasks.court_detection.data.target_generation.store import (
    CourtDerivedTargetStore,
)


def build_court_processing_pipeline(
    config: CourtDataConfig,
    *,
    is_train: bool,
) -> CourtProcessingPipeline:
    store = CourtDerivedTargetStore(config.processing.derived_target_root)
    input_layer = build_court_input(config.source, target_store=store)
    builders = tuple(
        build_target_builder(target, input_spec=input_layer.spec)
        for target in config.processing.targets
    )
    return CourtProcessingPipeline(
        input_layer=input_layer,
        geometry=CourtProcessingGeometry(config.augmentation, is_train=is_train),
        target_builders=builders,
    )


__all__ = ["build_court_processing_pipeline"]
