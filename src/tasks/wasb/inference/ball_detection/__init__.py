"""Ball detection inference predictors for WASB."""

from .heatmap_ensemble_predictor import HeatmapEnsemblePredictor
from .hrcnet_predictor import HRCNetWASBPredictor
from .wasb_predictor import WASBPredictor

__all__ = [
    "WASBPredictor",
    "HRCNetWASBPredictor",
    "HeatmapEnsemblePredictor",
]

