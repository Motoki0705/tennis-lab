"""Base module for shared abstractions."""

from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.base.training.lightning_module import BaseLightningModule

__all__ = ["BasePredictor", "BaseLightningModule"]
