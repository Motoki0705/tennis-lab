"""Independent binary cross entropy with soft event targets."""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F

from src.tasks.ball_refiner_3d.events import EVENT_CLASSES


def event_loss(logits: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    """Mean over event classes and unpadded frames, including missing observations."""
    if logits.shape != target.shape or target.shape != (
        *valid.shape,
        len(EVENT_CLASSES),
    ):
        raise ValueError(
            "Require aligned B,T,3 event logits/targets and B,T frame mask"
        )
    return F.binary_cross_entropy_with_logits(logits[valid], target[valid])
