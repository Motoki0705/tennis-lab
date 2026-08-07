"""Base class for pipeline modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BasePipelineModule(ABC):
    """Abstract base class for pipeline modules.

    All pipeline modules should inherit from this class and implement
    the required methods.
    """

    @abstractmethod
    def load(self) -> None:
        """Load model weights and initialize the module."""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the module is loaded."""
        pass

    @abstractmethod
    def process(self, *args: Any, **kwargs: Any) -> Any:
        """Process input data and return results."""
        pass
