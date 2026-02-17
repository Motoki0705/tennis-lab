"""Base interface for model-specific temporal input adapters."""

from __future__ import annotations

from torch import Tensor


class ModelInputAdapter:
    """Build model inputs from streaming RGB frames."""

    def reset(self) -> None:
        """Reset internal temporal state."""
        raise NotImplementedError

    def step_input(self, frame_chw: Tensor) -> Tensor:
        """Consume one RGB frame [3,H,W] and return a model input batch [1,...]."""
        raise NotImplementedError

    def extract_current_logits(self, logits: Tensor) -> Tensor:
        """Extract current-frame logits as [B,H,W] from model output logits."""
        raise NotImplementedError
