"""Test fixtures for e2e tests."""

from __future__ import annotations

from tests.e2e.fixtures.blcs_fixtures import (
    create_minimal_blcs_checkpoint,
    create_minimal_blcs_dataset,
    make_minimal_blcs_scene,
)
from tests.e2e.fixtures.plcs_fixtures import (
    create_minimal_plcs_checkpoint,
    create_minimal_plcs_dataset,
    make_minimal_plcs_scene,
)
from tests.e2e.fixtures.wasb_fixtures import (
    create_minimal_video,
    create_minimal_wasb_checkpoint,
    create_minimal_wasb_dataset,
)

__all__ = [
    "make_minimal_plcs_scene",
    "create_minimal_plcs_dataset",
    "create_minimal_plcs_checkpoint",
    "make_minimal_blcs_scene",
    "create_minimal_blcs_dataset",
    "create_minimal_blcs_checkpoint",
    "create_minimal_wasb_dataset",
    "create_minimal_wasb_checkpoint",
    "create_minimal_video",
]
