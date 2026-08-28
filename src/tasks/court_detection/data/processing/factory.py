"""Composition root for the orthogonal Court input and target axes."""

from __future__ import annotations

from src.tasks.court_detection.configuration import CourtDataConfig
from src.tasks.court_detection.data.contracts import CourtInputCapability
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
    require_pose: bool = False,
) -> CourtProcessingPipeline:
    store = CourtDerivedTargetStore(config.processing.derived_target_root)
    input_layer = build_court_input(config.source, target_store=store)
    if (
        require_pose
        and CourtInputCapability.V3_TARGET_COURT_POSE
        not in input_layer.spec.capabilities
    ):
        raise ValueError(
            "Court query source lacks the V3 target-court pose capability."
        )
    builders = tuple(
        build_target_builder(target, input_spec=input_layer.spec)
        for target in config.processing.targets
    )
    return CourtProcessingPipeline(
        input_layer=input_layer,
        geometry=CourtProcessingGeometry(
            config.augmentation,
            is_train=is_train,
            require_pose=require_pose,
        ),
        target_builders=builders,
        require_pose=require_pose,
    )


__all__ = ["build_court_processing_pipeline"]
