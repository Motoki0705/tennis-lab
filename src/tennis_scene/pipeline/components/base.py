"""Base class for pipeline modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


def release_inference_memory(device: str) -> None:
    """Release cyclic model references and idle CUDA allocations between stages."""
    import gc

    import torch
    gc.collect()
    if torch.device(device).type == "cuda" and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


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
