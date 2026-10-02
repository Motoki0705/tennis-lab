"""Source-neutral, in-memory Court target rasterization."""

from src.tasks.court_detection.data.target_generation.line import (
    generate_line_target,
)
from src.tasks.court_detection.data.target_generation.online import (
    generate_online_targets,
)
from src.tasks.court_detection.data.target_generation.segmentation import (
    generate_segmentation_target,
)
from src.tasks.court_detection.data.target_generation.semantic_line import (
    generate_semantic_line_target,
)
from src.tasks.court_detection.target_schemas import (
    LINE_TARGET_SCHEMA,
    SEGMENTATION_TARGET_SCHEMA,
)

__all__ = [
    "LINE_TARGET_SCHEMA",
    "SEGMENTATION_TARGET_SCHEMA",
    "generate_line_target",
    "generate_online_targets",
    "generate_segmentation_target",
    "generate_semantic_line_target",
]
