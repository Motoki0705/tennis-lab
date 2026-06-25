"""Global pytest fixtures shared across the whole test suite.

Keep this file dependency-light and domain-agnostic: seeding, tmp helpers and
generic dummy-tensor factories live here. Task-specific fixtures belong in a
``conftest.py`` directly under that task's test directory (for example
``tests/unit/tasks/base/conftest.py``).
"""

from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic_seed() -> None:
    """Seed Python / NumPy / Torch before every test for reproducibility.

    Autouse so individual tests never have to remember to seed; tests that need
    a specific seed can still re-seed locally.
    """
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded NumPy ``Generator`` for tests that want explicit randomness."""
    return np.random.default_rng(1234)


@pytest.fixture
def torch_generator() -> torch.Generator:
    """A seeded CPU ``torch.Generator`` for reproducible stochastic ops."""
    generator = torch.Generator()
    generator.manual_seed(1234)
    return generator


@pytest.fixture
def make_image() -> Callable[..., torch.Tensor]:
    """Factory producing dummy CHW (or BCHW) float image tensors in ``[0, 1]``."""

    def _make_image(
        *,
        channels: int = 3,
        height: int = 8,
        width: int = 8,
        batch: int | None = None,
    ) -> torch.Tensor:
        shape: tuple[int, ...] = (channels, height, width)
        if batch is not None:
            shape = (batch, *shape)
        return torch.rand(*shape)

    return _make_image
