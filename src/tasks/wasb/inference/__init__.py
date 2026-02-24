"""Inference entrypoints for WASB ball detection.

Consumers should prefer importing from this module instead of importing from
submodules directly, to make refactors of the internal layout less disruptive.
"""

from .ball_detection import HeatmapEnsemblePredictor, HRCNetWASBPredictor, WASBPredictor

__all__ = [
    "WASBPredictor",
    "HRCNetWASBPredictor",
    "HeatmapEnsemblePredictor",
]
