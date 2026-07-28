"""Configuration-selectable BLCS asset and trajectory algorithms."""

from __future__ import annotations

from src.synthetic_data_generation.dataset.algorithms import (
    AlgorithmDefinition,
    AlgorithmRegistry,
)

BALL_ASSET_ALGORITHMS = AlgorithmRegistry(
    namespace="blcs.ball_asset",
    definitions=(
        AlgorithmDefinition(
            name="procedural_fibonacci",
            implementation="procedural_fibonacci",
            description=(
                "Generate a metric Fibonacci-shell Gaussian ball before NHT fitting."
            ),
        ),
        AlgorithmDefinition(
            name="registered_gaussian_asset",
            implementation="registered_gaussian_asset",
            description=(
                "Use one or more externally prepared Gaussian balls from a registry."
            ),
        ),
    ),
)

TRAJECTORY_ALGORITHMS = AlgorithmRegistry(
    namespace="blcs.trajectory",
    definitions=(
        AlgorithmDefinition(
            name="rally_physics",
            implementation="rally_physics",
            description=(
                "Generate deterministic single/multi-ball trajectories with the "
                "BLCS rally simulator."
            ),
        ),
    ),
)

__all__ = ["BALL_ASSET_ALGORITHMS", "TRAJECTORY_ALGORITHMS"]
