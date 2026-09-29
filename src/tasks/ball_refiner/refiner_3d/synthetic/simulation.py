"""Finite, audited sampling of the BLCS simulator's accepted physics proposals."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import GenerationPlan
from src.tasks.blcs.generate_dataset.simulation.ball_physics import PhysicsConfig
from src.tasks.blcs.generate_dataset.simulation.cell_manager import (
    NUM_CELLS_PER_SIDE,
    CellManager,
)
from src.tasks.blcs.generate_dataset.simulation.rally_simulator import (
    RallyConfig,
    RallyResult,
    RallySimulator,
)
from src.tasks.blcs.generate_dataset.simulation.targeted_velocity_sampler import (
    TargetedVelocityConfig,
    is_retryable_full_physics_rejection,
)


def accepted_rally(
    plan: GenerationPlan, *, seed: int, index: int,
) -> tuple[RallyResult, PhysicsConfig, np.random.Generator, list[dict[str, Any]]]:
    proposals: list[dict[str, Any]] = []
    for attempt in range(plan.values["simulation"]["maximum_physics_attempts_per_rally"]):
        proposal_seed = seed if attempt == 0 else int(np.random.SeedSequence([seed, attempt]).generate_state(1)[0])
        torch.manual_seed(proposal_seed)
        rng = np.random.default_rng(proposal_seed)
        physics = PhysicsConfig(**plan.physics).sample()
        simulator = RallySimulator(
            physics_config=physics, rally_config=RallyConfig(**plan.rally),
            targeted_velocity_config=TargetedVelocityConfig(**{**plan.targeted, "gravity": physics.gravity}),
            cell_manager=CellManager(), device="cpu",
        )
        try:
            result = simulator.generate_rally(from_cell=int(rng.integers(0, NUM_CELLS_PER_SIDE)), from_side="near" if index % 2 == 0 else "far")
        except RuntimeError as exc:
            if not is_retryable_full_physics_rejection(exc):
                raise
            proposals.append({"seed": proposal_seed, "status": "rejected", "reason": str(exc)})
            continue
        proposals.append({"seed": proposal_seed, "status": "accepted"})
        return result, physics, rng, proposals
    raise RuntimeError(f"Physics proposal budget exhausted: {proposals}")
