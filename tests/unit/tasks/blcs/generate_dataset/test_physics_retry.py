from __future__ import annotations

import pytest

from src.tasks.blcs.generate_dataset.physics_retry import (
    generate_with_bounded_physics_resampling,
)
from src.tasks.blcs.generate_dataset.simulation.targeted_velocity_sampler import (
    FULL_PHYSICS_REJECTION_PREFIX,
)


def test_bounded_physics_resampling_accepts_after_retry(
    caplog: pytest.LogCaptureFixture,
) -> None:
    attempts = 0

    def proposal() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError(f"{FULL_PHYSICS_REJECTION_PREFIX}; rejected")
        return "accepted"

    caplog.set_level("INFO")

    result = generate_with_bounded_physics_resampling(
        proposal,
        scene_id="scene_000000",
        maximum_attempts=2,
    )

    assert result == "accepted"
    assert attempts == 2
    assert "Accepted BLCS physics proposal for scene_000000" in caplog.text


def test_bounded_physics_resampling_exhaustion_preserves_cause() -> None:
    rejection = RuntimeError(f"{FULL_PHYSICS_REJECTION_PREFIX}; rejected")

    def proposal() -> None:
        raise rejection

    with pytest.raises(RuntimeError, match="exhausted 2 bounded attempts") as exc_info:
        generate_with_bounded_physics_resampling(
            proposal,
            scene_id="scene_000000",
            maximum_attempts=2,
        )

    assert exc_info.value.__cause__ is rejection


def test_bounded_physics_resampling_does_not_retry_unexpected_errors() -> None:
    attempts = 0

    def proposal() -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("unexpected implementation failure")

    with pytest.raises(RuntimeError, match="unexpected implementation failure"):
        generate_with_bounded_physics_resampling(
            proposal,
            scene_id="scene_000000",
            maximum_attempts=64,
        )

    assert attempts == 1


@pytest.mark.parametrize("maximum_attempts", (True, 0, -1))
def test_bounded_physics_resampling_rejects_invalid_budget(
    maximum_attempts: int,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        generate_with_bounded_physics_resampling(
            lambda: "unused",
            scene_id="scene_000000",
            maximum_attempts=maximum_attempts,
        )
