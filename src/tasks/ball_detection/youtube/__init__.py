"""YouTube ball dataset workflow interfaces."""

from src.tasks.ball_detection.youtube.candidate_workflow import (
    CandidatePredictionConfig,
    CandidateSelectionConfig,
    predict_candidates,
    run_candidate_selection,
)

__all__ = [
    "CandidatePredictionConfig",
    "CandidateSelectionConfig",
    "predict_candidates",
    "run_candidate_selection",
]
