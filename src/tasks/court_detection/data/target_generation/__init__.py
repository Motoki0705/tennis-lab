"""Explicit source-neutral dense Court target materialization."""

from src.tasks.court_detection.data.target_generation.line import (
    generate_line_target,
)
from src.tasks.court_detection.data.target_generation.materializer import (
    CourtMaterializationResult,
    CourtTargetMaterializer,
)
from src.tasks.court_detection.data.target_generation.segmentation import (
    generate_segmentation_target,
)
from src.tasks.court_detection.data.target_generation.store import (
    LINE_TARGET_SCHEMA,
    SEGMENTATION_TARGET_SCHEMA,
    CourtDerivedTargetStore,
)

__all__ = [
    "CourtDerivedTargetStore",
    "CourtMaterializationResult",
    "CourtTargetMaterializer",
    "LINE_TARGET_SCHEMA",
    "SEGMENTATION_TARGET_SCHEMA",
    "generate_line_target",
    "generate_segmentation_target",
]
