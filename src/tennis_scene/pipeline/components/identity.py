"""Cross-camera player identities: artifact contract only.

The node is defined by its output schema. No model or algorithm is registered
for it yet (#933), so the standard definition declares it load-only: its
artifact must be imported into the clip store before the pipeline runs, and a
request to execute it fails when the definition is built, not after the
upstream models have run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NoReturn

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.pipeline.contracts import (
    AssemblyContext,
    ComponentIO,
    InputPort,
)

PLAYER_ASSOCIATION = "player_association"
# Version 2: tennis_scene-owned contract without PLCS Re-ID embeddings.
IDENTITIES_PORT = InputPort("person_identities", 2)


@dataclass(frozen=True)
class PlayerIdentitiesOutput:
    """Clip-global player ID of every camera-local pose track.

    ``camera_ids`` are the calibrated cameras in calibration order. Row ``v`` of
    ``local_track_ids`` repeats the tracker IDs of that camera's pose carriers
    (``-1`` for padding) so a join can verify the carrier order it was built on.
    ``player_ids`` holds the clip-global ID, or ``-1`` for a track that belongs
    to no reconstructed player.
    """

    camera_ids: tuple[str, ...]
    local_track_ids: NDArray[np.int64]  # (V, D)
    player_ids: NDArray[np.int64]  # (V, D)

    def __post_init__(self) -> None:
        views = len(self.camera_ids)
        if views < 1 or len(set(self.camera_ids)) != views or any(not c for c in self.camera_ids):
            raise ValueError("Player identities require unique nonempty camera IDs")
        for name in ("local_track_ids", "player_ids"):
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.dtype != np.int64 or value.ndim != 2 or len(value) != views:
                raise ValueError(f"{name} must be int64 (V, D) aligned to camera_ids")
        if self.player_ids.shape != self.local_track_ids.shape:
            raise ValueError("player_ids must align with local_track_ids")
        if ((self.local_track_ids < 0) & (self.player_ids >= 0)).any():
            raise ValueError("A padding carrier cannot carry a player ID")
        if (self.player_ids < -1).any():
            raise ValueError("Player IDs are nonnegative, or -1 for no player")
        for row in self.player_ids:
            assigned = row[row >= 0]
            if len(np.unique(assigned)) != len(assigned):
                raise ValueError("One camera cannot observe the same player through two tracks")


@dataclass(frozen=True)
class DeclaredArtifacts:
    """The loaded dependency artifacts of an import-only node, keyed by port."""

    context: AssemblyContext
    artifacts: Mapping[str, Any]


@dataclass(frozen=True)
class DeclaredArtifactsAssembler:
    """Pass-through assembly for nodes that have no implementation-specific input."""

    version: int = 1

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> DeclaredArtifacts:
        return DeclaredArtifacts(context, dict(artifacts))


class ImportOnlyComponent:
    """A declared node whose artifact can only be loaded from the clip store."""

    def __init__(self, io: ComponentIO[Any, Any], replacement: str) -> None:
        self.io = io
        self.replacement = replacement

    def process(self, inputs: DeclaredArtifacts) -> NoReturn:
        raise RuntimeError(
            f"{self.io.name} has no model implementation ({self.replacement}); "
            "import its artifact and declare the node as load"
        )


def player_association_io(camera_ids: tuple[str, ...]) -> ComponentIO[DeclaredArtifacts, PlayerIdentitiesOutput]:
    return ComponentIO(PLAYER_ASSOCIATION, DeclaredArtifacts, PlayerIdentitiesOutput,
        {"calibration": InputPort("local_court_calibration"), **{f"pose_{c}": InputPort("person_poses") for c in camera_ids}},
        IDENTITIES_PORT.schema, IDENTITIES_PORT.version)

