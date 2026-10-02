"""Attribute selected reference-unit hits to tracks and calibrated footpoints.

This is post-hoc evaluation only. Labels never enter the selection rule.
Only selected tracks compete here, matching the selected-unit metric rather
than letting a rejected duplicate steal a labelled unit from a kept track.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.person_tracking.court_linking import LinkingConfig, exclusive_region
from src.tasks.person_tracking.selection_metrics import NONPLAYER_KIND
from src.tasks.player_association.association.associate import CameraTracks
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_association.geometry.footpoints import (
    FootpointConfig,
    ground_footpoints,
)
from src.tasks.player_association.geometry.region import (
    PlayRegionConfig,
    in_play_region,
)
from src.tasks.player_detection.evaluation.partial_labels import _unit_matches
from src.utils.geometry.bbox import pairwise_iou


def diagnose_tracks(tracks: CameraTracks, selected: NDArray[np.bool_], labels: ClipLabels) -> dict[str, Any]:
    points, valid = ground_footpoints(tracks.boxes_xyxy, tracks.observed, tracks.camera, tracks.image_size[1], FootpointConfig())
    core = valid & exclusive_region(points, LinkingConfig(), core=True)
    corridor = valid & exclusive_region(points, LinkingConfig(), core=False)
    old = valid & in_play_region(points, PlayRegionConfig(2.5, 5.))
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    evidence: list[dict[str, Any]] = []
    reference = labels.cameras[tracks.camera.camera_id]
    for frame in range(labels.num_frames):
        rows = np.flatnonzero(tracks.observed[:, frame] & selected[:, frame])
        at = reference.at(frame)
        people, boxes = reference.person_index[at], reference.boxes_xyxy[at]
        units = np.unique(people[people >= 0])
        overlap = pairwise_iou(tracks.boxes_xyxy[rows, frame], boxes)
        iou = np.column_stack([overlap[:, people == i].max(1) for i in units]) if len(units) else np.empty((len(rows), 0))
        left, right = _unit_matches(iou, .3)
        for row, unit in zip(rows[left], units[right], strict=True):
            person = labels.people[unit]
            counts[int(row)][person.person_id] += 1
            if NONPLAYER_KIND.get((labels.clip_id, person.person_id)) == 'adjacent_court':
                evidence.append({'frame': frame, 'track_id': int(tracks.track_ids[row]), 'row': int(row),
                    'person': person.person_id, 'xy': points[row, frame].tolist(), 'valid': bool(valid[row, frame]),
                    'in_old_region': bool(old[row, frame]), 'in_corridor': bool(corridor[row, frame]),
                    'in_core': bool(core[row, frame]), 'box': tracks.boxes_xyxy[row, frame].tolist()})
    player_ids = {p.person_id for p in labels.people if p.role == 'player'}
    summaries = []
    for row in sorted(counts):
        adjacent = [u for u in evidence if u['row'] == row]
        summaries.append({'track_id': int(tracks.track_ids[row]), 'row': row, 'composition': dict(counts[row]),
            'player_and_adjacent_mixed': bool(adjacent and player_ids.intersection(counts[row])),
            'adjacent_kept_units': len(adjacent),
            'adjacent_xy_min_median_max': np.quantile([u['xy'] for u in adjacent], [0, .5, 1], axis=0).tolist() if adjacent else None,
            'adjacent_inside_old': sum(u['in_old_region'] for u in adjacent),
            'adjacent_inside_corridor': sum(u['in_corridor'] for u in adjacent),
            'adjacent_inside_core': sum(u['in_core'] for u in adjacent),
            'all_observed': int(tracks.observed[row].sum()), 'all_old_dwell': int(old[row].sum()),
            'all_core_dwell': int(core[row].sum()), 'all_corridor_dwell': int(corridor[row].sum())})
    return {'tracks': summaries, 'adjacent_units': evidence,
            'interpretation': 'one-to-one IoU .3 to reviewed old COCO boxes among selected predictions; partial labels only'}
