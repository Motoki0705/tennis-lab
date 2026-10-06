"""Flow matching objective and its explicitly owned random stream."""

import torch
from torch import Tensor
from torch.nn import functional as F

from src.tasks.ball_refiner_3d.model_io.factory import bind_refiner
from src.tasks.ball_refiner_3d.models.generators.flow import FlowRefiner


def flow_matching_loss(
    model: FlowRefiner,
    coordinates: Tensor,
    missing: Tensor,
    target: Tensor,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    time = torch.rand((len(target),), device=target.device, generator=generator) * 0.95
    noise = torch.randn(
        target.shape, device=target.device, dtype=target.dtype, generator=generator
    )
    t = time[:, None, None]
    state = (1 - t) * noise + t * target
    result = bind_refiner(model).run(
        {"coordinates": coordinates, "missing": missing, "state": state, "time": time}
    )
    loss = F.mse_loss((result.coordinates - state) / (1 - t), target - noise)
    return loss, result.event_logits
