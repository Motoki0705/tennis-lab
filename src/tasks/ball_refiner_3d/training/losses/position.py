"""All-frame position supervision in normalized court coordinates."""

from torch import Tensor


def masked_l1(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    """Mean absolute error over the ``valid (B,T)`` frames and all channels."""
    if not bool(valid.any()):
        raise ValueError("Require at least one valid frame")
    error = (prediction - target).abs().mean(dim=-1)
    return error[valid].mean()


def position_loss(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    return masked_l1(prediction, target, valid)
