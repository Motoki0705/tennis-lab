"""Parsers that normalize supported raw web datasets."""

from src.tasks.ball_detection.data.components.web.parser.ball_yolo import (
    BallYoloParser,
)
from src.tasks.ball_detection.data.components.web.parser.base import (
    ParsedSource,
    WebDatasetParser,
)
from src.tasks.ball_detection.data.components.web.parser.kaggle import KaggleParser
from src.tasks.ball_detection.data.components.web.parser.racketvision import (
    RacketVisionParser,
)
from src.tasks.ball_detection.data.components.web.parser.roboflow import (
    RoboflowParser,
)
from src.utils.data.splits import GroupSplitConfig

__all__ = [
    "BallYoloParser",
    "GroupSplitConfig",
    "KaggleParser",
    "ParsedSource",
    "RacketVisionParser",
    "RoboflowParser",
    "WebDatasetParser",
]
