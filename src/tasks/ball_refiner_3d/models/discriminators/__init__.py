"""Trajectory-only Transformer discriminator, using the BLCS/PLCS builder."""

from __future__ import annotations

from dataclasses import asdict

import torch
from torch import Tensor, nn

from src.tasks.ball_refiner_3d.config import DiscriminatorConfig
from src.utils.models.architectures import build_trajectory_discriminator


class TrajectoryDiscriminator(nn.Module):
    """Score each complete (B,T,D) output trajectory; no observation condition."""

    def __init__(self, dimensions: int, config: DiscriminatorConfig) -> None:
        super().__init__()
        if dimensions != 3:
            raise ValueError("Require 3D trajectories")
        self.config = config
        values = asdict(config)
        values.pop("name")
        self.network = build_trajectory_discriminator(input_dim=dimensions, disc_cfg=values)

    def forward(self, trajectory: Tensor) -> Tensor:
        # Every generated frame is a real token, including reconstructed gaps.
        padding = torch.zeros(trajectory.shape[:2], dtype=torch.bool, device=trajectory.device)
        return self.network(trajectory, padding_mask=padding)


def build_refiner_discriminator(dimensions: int, config: DiscriminatorConfig) -> TrajectoryDiscriminator:
    return TrajectoryDiscriminator(dimensions, config)


__all__ = ["TrajectoryDiscriminator", "build_refiner_discriminator"]
