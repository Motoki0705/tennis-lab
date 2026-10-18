"""Explicit annotation masks; legacy selection keeps its existing semantics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .store import PlayerFrameStore

SCHEMA = "player.annotation-supervision.v1"
COLUMN = "teacher_supervision"
LEGACY, POSITIVE, NEGATIVE, IGNORE = range(4)
CODES = {"positive": POSITIVE, "negative": NEGATIVE, "ignore": IGNORE}


def validate_supervision(store: PlayerFrameStore) -> None:
    values = store.teacher_supervision
    if values is None:
        return
    if values.dtype != np.uint8 or values.shape != (len(store),):
        raise ValueError("Player teacher supervision must be one uint8 per frame")
    if not np.isin(values, [LEGACY, POSITIVE, NEGATIVE, IGNORE]).all():
        raise ValueError("Unknown player teacher supervision")
    selected = (values == POSITIVE) | (values == NEGATIVE)
    if (selected & (~store.frames["is_target"] | ~store.frames["reviewed"])).any():
        raise ValueError("Explicit teachers require reviewed target frames")
    positive = values == POSITIVE
    negative = values == NEGATIVE
    if (positive & (store.frames["inst_count"] == 0)).any() or (
        negative != ((values != LEGACY) & store.frames["confirmed_empty"])
    ).any():
        raise ValueError("Player supervision disagrees with positive/empty targets")
    for clip in store.clips:
        rows = store.clip_frames(clip)
        legacy = values[rows] == LEGACY
        if legacy.any() and not legacy.all():
            raise ValueError("A clip cannot mix legacy and explicit supervision")
    for row in np.flatnonzero(positive):
        if (store.instances_of(int(row)).bbox_source != 0).any():
            raise ValueError("Positive teachers require only observed boxes")
