"""Box labels of cross-camera person identity and the association metrics."""

from src.tasks.player_association.evaluation.labels import (
    AMBIGUOUS,
    CameraLabels,
    ClipLabels,
    LabelledPerson,
    ReviewedTrack,
    materialize,
)
from src.tasks.player_association.evaluation.metrics import (
    CameraPrediction,
    evaluate,
    match_to_labels,
)

__all__ = ["AMBIGUOUS", "CameraLabels", "CameraPrediction", "ClipLabels", "LabelledPerson", "ReviewedTrack",
           "evaluate", "match_to_labels", "materialize"]
