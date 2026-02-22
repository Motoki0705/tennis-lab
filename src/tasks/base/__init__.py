"""Base module for shared abstractions."""

from src.base.inference.predictor import BasePredictor
from src.base.training.lightning_module import BaseLightningModule

__all__ = ["BasePredictor", "BaseLightningModule"]
