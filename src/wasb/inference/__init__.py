"""Inference entrypoints and trajectory completion helpers.

Consumers should prefer importing from this module instead of importing from
submodules directly, to make refactors of the internal layout less disruptive.
"""

from .ball_detection import HeatmapEnsemblePredictor, HRCNetWASBPredictor, WASBPredictor
from .event_detection import (
    EventDetectionResult,
    TrajectoryEventDetector,
    load_event_detector_from_checkpoint,
)
from .trajectory.trajectory_completion import (
    BiLSTMCompleter,
    CompletionResult,
    HybridCompleter,
    IterativeRefinementCompleter,
    PhysicsInterpolator,
    TrajectoryCompleter,
    TransformerCompleter,
    build_completer,
)

__all__ = [
    "WASBPredictor",
    "HRCNetWASBPredictor",
    "HeatmapEnsemblePredictor",
    "TrajectoryCompleter",
    "CompletionResult",
    "EventDetectionResult",
    "PhysicsInterpolator",
    "BiLSTMCompleter",
    "TransformerCompleter",
    "IterativeRefinementCompleter",
    "HybridCompleter",
    "TrajectoryEventDetector",
    "build_completer",
    "load_event_detector_from_checkpoint",
]
