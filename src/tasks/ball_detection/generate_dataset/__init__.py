"""Ball annotation interfaces."""

from src.tasks.ball_detection.generate_dataset.annotation_session import (
    BallAnnotationSessionConfig,
    FinalizeConfig,
    ZoomConfig,
    run_annotation_session,
)
from src.tasks.ball_detection.generate_dataset.candidate_workflow import (
    CandidatePredictionConfig,
    CandidateSelectionConfig,
    predict_candidates,
    run_candidate_selection,
)

__all__ = [
    "BallAnnotationSessionConfig",
    "CandidatePredictionConfig",
    "CandidateSelectionConfig",
    "FinalizeConfig",
    "ZoomConfig",
    "predict_candidates",
    "run_annotation_session",
    "run_candidate_selection",
]
