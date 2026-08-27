"""Fixed court normalization tests at the BLCS physics boundary."""

from __future__ import annotations

import torch

from src.tasks.blcs.generate_dataset.simulation.ball_physics import BallPhysics


def test_position_normalization_uses_one_scale_and_round_trips() -> None:
    physics = object.__new__(BallPhysics)
    positions_m = torch.tensor(
        [[-5.485, -11.885, 0.0], [5.485, 11.885, 1.07]],
        dtype=torch.float32,
    )

    normalized = physics.normalize_position(positions_m)
    recovered = physics.denormalize_position(normalized)

    torch.testing.assert_close(
        normalized,
        positions_m / 11.885,
        atol=1e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(recovered, positions_m, atol=1e-5, rtol=0.0)
