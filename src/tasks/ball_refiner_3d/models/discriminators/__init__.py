"""Trajectory-only discriminator factory."""

from .trajectory import TrajectoryDiscriminator, build_refiner_discriminator

__all__ = ["TrajectoryDiscriminator", "build_refiner_discriminator"]
