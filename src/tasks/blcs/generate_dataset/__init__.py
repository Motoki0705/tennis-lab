"""BLCS dataset generation package.

This package contains the modular implementation of BLCS synthetic dataset generation:
- physics-based shot simulation
- distribution-controlled sampling
- camera sampling + projection
- scene serialization + split/meta writing

The Hydra CLI entrypoint is `src/tasks/blcs/scripts/generate_dataset.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.tasks.blcs.generate_dataset.source_api import (  # noqa: F401
        BLCSGeneratorConfiguration,
        BLCSPhysicsProposalExhausted,
        BLCSPhysicsProposalRejected,
        BLCSPhysicsProvenance,
        BLCSPhysicsSourceSettings,
        BLCSPhysicsTrajectorySource,
        BLCSProposalDiagnostic,
        BLCSProposalRejection,
        BLCSSourceScene,
        BLCSSourceTrack,
        BLCSTimelineSpec,
        build_blcs_generator_configuration,
    )

_SOURCE_API_EXPORTS = frozenset(
    {
        "BLCSGeneratorConfiguration",
        "BLCSPhysicsProposalExhausted",
        "BLCSPhysicsProposalRejected",
        "BLCSPhysicsProvenance",
        "BLCSPhysicsSourceSettings",
        "BLCSPhysicsTrajectorySource",
        "BLCSProposalDiagnostic",
        "BLCSProposalRejection",
        "BLCSSourceScene",
        "BLCSSourceTrack",
        "BLCSTimelineSpec",
        "build_blcs_generator_configuration",
    }
)


def __getattr__(name: str) -> Any:
    """Load the public source boundary without changing legacy import side effects."""
    if name in _SOURCE_API_EXPORTS:
        from src.tasks.blcs.generate_dataset import source_api

        return getattr(source_api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_SOURCE_API_EXPORTS)
