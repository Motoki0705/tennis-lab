"""Training entrypoints for DINOv3 tennis SSL."""

from src.tasks.dino_ssl.training.lightning_module import DinoSSLLightningModule
from src.tasks.dino_ssl.training.runner import DinoSSLTrainingRunner

__all__ = ["DinoSSLLightningModule", "DinoSSLTrainingRunner"]
