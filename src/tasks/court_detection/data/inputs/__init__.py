"""Source-specific Court input adapters."""

from src.tasks.court_detection.data.inputs.contract import CourtInput
from src.tasks.court_detection.data.inputs.factory import build_court_input
from src.tasks.court_detection.data.inputs.synthetic_court import SyntheticCourtInput
from src.tasks.court_detection.data.inputs.tennis_court_detector import (
    TennisCourtDetectorInput,
)

__all__ = [
    "CourtInput",
    "SyntheticCourtInput",
    "TennisCourtDetectorInput",
    "build_court_input",
]
