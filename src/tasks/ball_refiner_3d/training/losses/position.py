"""All-frame position supervision in normalized court coordinates."""

from torch import Tensor
from torch.nn import functional as F


def position_loss(prediction: Tensor, target: Tensor) -> Tensor:
    return F.l1_loss(prediction, target)
