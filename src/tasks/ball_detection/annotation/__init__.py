"""Ball annotation interfaces."""

from src.tasks.ball_detection.annotation.youtube_session import (
    BallAnnotationSessionConfig,
    FinalizeConfig,
    ZoomConfig,
    run_annotation_session,
)

__all__ = [
    "BallAnnotationSessionConfig",
    "FinalizeConfig",
    "ZoomConfig",
    "run_annotation_session",
]
