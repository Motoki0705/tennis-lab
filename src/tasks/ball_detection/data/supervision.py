"""Per-frame training supervision derived from ball store labels.

Every ``point_kind`` of the store is assigned exactly one role:

* ``positive``: a located ball that becomes a heatmap peak and a metric target.
* ``absent``: the frame is still supervised, but this instance adds no peak
  (``out_of_frame``: the ball in play is outside the image).
* ``ignore``: the whole frame is excluded from the loss and the metrics,
  because its heatmap target would be wrong or unknown (an ``unresolved`` ball
  exists but has no position; an estimated position is not an observation).

A frame is **supervised** iff it is ``annotated`` and none of its instances has
an ``ignore`` kind. A supervised frame without a positive instance is a
negative. An ``unresolved`` ball is never allowed to be ``absent`` or
``positive``: it would silently become a negative or a fake peak.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.data.store import (
    LOCATED_POINT_KINDS,
    POINT_KIND_CODES,
    BallFrameStore,
)

ROLES: Final = ("positive", "absent", "ignore")


@dataclass(frozen=True, slots=True)
class FrameSupervisionPolicy:
    """A total, disjoint assignment of every point kind to one role."""

    positive: frozenset[str]
    absent: frozenset[str]
    ignore: frozenset[str]

    def __post_init__(self) -> None:
        roles = (self.positive, self.absent, self.ignore)
        assigned = [kind for role in roles for kind in role]
        if len(assigned) != len(set(assigned)):
            raise ValueError(
                "A point kind is assigned to more than one supervision role"
            )
        if set(assigned) != set(POINT_KIND_CODES):
            missing = sorted(set(POINT_KIND_CODES) - set(assigned))
            unknown = sorted(set(assigned) - set(POINT_KIND_CODES))
            raise ValueError(
                f"Supervision roles must cover every point kind exactly once "
                f"(missing {missing}, unknown {unknown})"
            )
        if not self.positive:
            raise ValueError("At least one point kind must be positive")
        if not self.positive <= LOCATED_POINT_KINDS:
            raise ValueError(
                f"Positive kinds must be located kinds {sorted(LOCATED_POINT_KINDS)}, "
                f"got {sorted(self.positive)}"
            )
        if not self.absent <= {"out_of_frame"}:
            raise ValueError("Only out_of_frame instances may be absent")
        if "unresolved" not in self.ignore:
            raise ValueError(
                "An unresolved ball must be ignored, never absent or positive"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Sequence[str]]) -> FrameSupervisionPolicy:
        if set(value) != set(ROLES):
            raise ValueError(
                f"Supervision policy needs exactly the roles {list(ROLES)}"
            )
        return cls(
            **{role: frozenset(str(kind) for kind in value[role]) for role in ROLES}
        )

    def codes(self, role: str) -> NDArray[np.uint8]:
        kinds: frozenset[str] = getattr(self, role)
        return np.asarray(
            sorted(POINT_KIND_CODES[kind] for kind in kinds), dtype=np.uint8
        )


@dataclass(frozen=True, slots=True)
class FrameSupervision:
    """Row-aligned supervision of a whole store.

    ``supervised`` is indexed like the store frame table and ``positive`` like
    its instance table.
    """

    supervised: NDArray[np.bool_]
    positive: NDArray[np.bool_]


def resolve_frame_supervision(
    store: BallFrameStore, policy: FrameSupervisionPolicy
) -> FrameSupervision:
    """Apply ``policy`` to every frame and instance of ``store``."""
    kinds = store.instances["point_kind"]
    ignored = np.isin(kinds, policy.codes("ignore"))
    counts = store.frames["inst_count"]
    frame_of_instance: NDArray[np.int64] = np.repeat(
        np.arange(len(store), dtype=np.int64), counts
    )
    has_ignored: NDArray[np.bool_] = np.zeros(len(store), dtype=np.bool_)
    has_ignored[frame_of_instance[ignored]] = True
    supervised = store.frames["annotated"] & ~has_ignored
    positive = np.isin(kinds, policy.codes("positive"))
    return FrameSupervision(supervised=supervised, positive=positive)


__all__ = [
    "ROLES",
    "FrameSupervision",
    "FrameSupervisionPolicy",
    "resolve_frame_supervision",
]


# Review metrics use the same observed-only policy as the shipped training config.
OBSERVED_ONLY = FrameSupervisionPolicy(
    positive=frozenset({"observed"}),
    absent=frozenset({"out_of_frame"}),
    ignore=frozenset({"interpolated", "occlusion_estimated", "unresolved"}),
)
