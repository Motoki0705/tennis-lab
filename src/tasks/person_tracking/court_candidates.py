"""Offline camera-local dwell selection for the #964 development diagnostic.

All persons reach tracking. Only observed, valid box-bottom footpoints inside
the existing play region count toward dwell. A candidate must accumulate the
configured fraction of the clip; the capacity is applied AFTER that decision.
No detector score enters this selection. Cross-camera identity and ambiguity
handling belong to the existing player_association second check.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.player_association.association.associate import CameraTracks
from src.tasks.player_association.geometry.footpoints import (
    FootpointConfig,
    ground_footpoints,
)
from src.tasks.player_association.geometry.region import (
    PlayRegionConfig,
    in_play_region,
)


@dataclass(frozen=True)
class DwellConfig:
    region: PlayRegionConfig
    footpoints: FootpointConfig
    min_presence_fraction: float
    max_candidates: int = 6

    def __post_init__(self) -> None:
        if not 0 < self.min_presence_fraction <= 1 or self.max_candidates < 1:
            raise ValueError('Invalid dwell fraction or candidate capacity')


def select_candidates(tracks: CameraTracks, config: DwellConfig) -> tuple[NDArray[np.int64], list[dict[str, Any]]]:
    points, valid = ground_footpoints(tracks.boxes_xyxy, tracks.observed, tracks.camera,
                                     tracks.image_size[1], config.footpoints)
    dwell = (valid & in_play_region(points, config.region)).sum(axis=1)
    required = tracks.observed.shape[1] * config.min_presence_fraction
    eligible = [int(row) for row in np.flatnonzero(dwell >= required)]
    ranked = sorted(eligible, key=lambda row: (-int(dwell[row]), int(tracks.track_ids[row])))
    chosen = np.asarray(ranked[:config.max_candidates], np.int64)
    chosen_set = set(chosen.tolist())
    diagnostics = [{'track_id': int(track), 'observed_frames': int(tracks.observed[row].sum()),
        'valid_footpoint_frames': int(valid[row].sum()), 'in_region_frames': int(dwell[row]),
        'required_frames': required, 'selected': row in chosen_set,
        'reason': 'selected' if row in chosen_set else 'candidate_cap' if row in eligible else 'insufficient_dwell'}
        for row, track in enumerate(tracks.track_ids)]
    return chosen, diagnostics
