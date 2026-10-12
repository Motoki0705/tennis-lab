"""Observed-position supervision for all coordinate heads."""
import torch
from torch import Tensor


def coordinate_loss(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    if prediction.shape != target.shape or valid.shape != target.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError("Coordinate loss shape/mask mismatch")
    if not bool(valid.any()):
        raise ValueError("No observed coordinate supervision")
    return torch.nn.functional.smooth_l1_loss(prediction[valid], target[valid], beta=.01)
