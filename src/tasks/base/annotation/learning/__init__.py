"""Task-independent learning and model adoption."""

from .adapters import LearningAdapter
from .models import ModelRegistry
from .service import Learning

__all__ = ["Learning", "LearningAdapter", "ModelRegistry"]
