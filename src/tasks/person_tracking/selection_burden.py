"""Track population by the same reference-box near/far strata used in reports."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.submodules.models import PersonDetectionResult
from src.tasks.player_association.association.associate import CameraTracks
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_detection.evaluation.far_player import unit_rows


def track_burden(tracks: CameraTracks, selected: np.ndarray, labels: ClipLabels) -> list[dict[str, Any]]:
    names = ('all', 'near', 'far', 'unknown')
    counts = {s: {'track_observations': 0, 'selected_observations': 0, 'max_tracks_per_frame': 0,
                  'max_selected_per_frame': 0} for s in names}
    totals: dict[str, set[int]] = {s: set() for s in names}
    kept: dict[str, set[int]] = {s: set() for s in names}
    for frame in range(labels.num_frames):
        active = np.flatnonzero(tracks.observed[:, frame])
        rows = unit_rows(PersonDetectionResult(np.empty((0, 4), np.float32), np.empty(0, np.float32)),
            labels.cameras[tracks.camera.camera_id], labels.roles, frame)
        players = [r for r in rows if r['role'] == 'player']
        midpoint = sum(r['old_box'][3] for r in players) / 2 if len(players) == 2 and players[0]['old_box'][3] != players[1]['old_box'][3] else None
        sides = np.full(len(active), 'unknown', dtype='<U7') if midpoint is None else np.where(tracks.boxes_xyxy[active, frame, 3] < midpoint, 'far', 'near')
        for side in names:
            ids = active if side == 'all' else active[sides == side]
            selected_ids = ids[selected[ids, frame]]
            totals[side].update(ids.tolist())
            kept[side].update(selected_ids.tolist())
            count = counts[side]
            count['track_observations'] += len(ids)
            count['selected_observations'] += len(selected_ids)
            count['max_tracks_per_frame'] = max(count['max_tracks_per_frame'], len(ids))
            count['max_selected_per_frame'] = max(count['max_selected_per_frame'], len(selected_ids))
    return [{'camera': tracks.camera.camera_id, 'near_far': side, 'frames': labels.num_frames,
             **counts[side], 'tracks_total': len(totals[side]), 'selected_tracks_total': len(kept[side])} for side in names]
