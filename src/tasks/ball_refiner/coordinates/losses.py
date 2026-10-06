"""Generator loss coefficients counted in zero-based optimizer updates."""

import torch

from src.tasks.ball_refiner.coordinates.config import ReconstructionConfig
from src.tasks.base.training.gan_schedule import gan_weight_at


def reconstruction_weight_at(index: int, config: ReconstructionConfig) -> float:
    progress = gan_weight_at(index, start=config.start_step, warmup=config.decay_steps, target=1.0)
    return float(config.initial_weight * (1.0 - progress) + config.final_weight * progress)


def generator_objective(reconstruction: torch.Tensor, adversarial: torch.Tensor, *, reconstruction_weight: float, gan_weight: float) -> torch.Tensor:
    """A zero reconstruction coefficient removes its graph from the objective."""
    if reconstruction_weight == 0:
        return gan_weight * adversarial
    return reconstruction_weight * reconstruction + gan_weight * adversarial
