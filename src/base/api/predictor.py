"""Abstract base class for inference predictors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self

import torch


class BasePredictor(ABC):
    """Abstract base class for inference predictors.

    All predictors must implement two methods:
    - load_from_checkpoint: Create instance from checkpoint file.
    - predict: Run batch inference.

    Attributes:
        model: The model used for inference.
        device: The device for inference.

    Example:
        >>> predictor = MyPredictor.load_from_checkpoint("model.ckpt", device="cuda")
        >>> results = predictor.predict(inputs)

    """

    model: torch.nn.Module
    device: torch.device

    @classmethod
    @abstractmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> Self:
        """Create a predictor instance from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file.
            device: Device for inference ("cpu" or "cuda").
            **kwargs: Implementation-specific additional arguments.

        Returns:
            Initialized predictor instance.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.

        """
        ...

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Run batch inference.

        Input/output formats vary by implementation.
        See subclass documentation for details.

        Returns:
            Dictionary of inference results. Keys and value types are
            implementation-dependent.

        """
        ...
