"""Generator loss coefficient schedules indexed by generator updates."""

from __future__ import annotations

from src.tasks.ball_refiner_3d.configuration.training import ReconstructionConfig
from src.tasks.base.training.gan_schedule import gan_weight_at


def reconstruction_weight_at(index: int, config: ReconstructionConfig) -> float:
    if not config.enabled:
        return config.initial_weight
    progress = gan_weight_at(
        index, start=config.start_step, warmup=config.decay_steps, target=1.0
    )
    return float(
        config.initial_weight * (1.0 - progress) + config.final_weight * progress
    )
